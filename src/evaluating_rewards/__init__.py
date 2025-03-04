# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for comparing and evaluating reward models."""

from evaluating_rewards import envs  # noqa: F401
from evaluating_rewards.version import VERSION as __version__  # noqa: F401


def _register_policies():
    # import inside function so as not to pollute global namespace
    from imitation.policies import serialize  # pylint:disable=import-outside-toplevel

    serialize.policy_registry.register(
        "mixture", indirect="evaluating_rewards.policies.mixture:load_mixture"
    )
    serialize.policy_registry.register(
        key="evaluating_rewards/MCGreedy-v0",
        indirect="evaluating_rewards.policies.monte_carlo:load_monte_carlo_greedy",
    )


# _register_policies()
