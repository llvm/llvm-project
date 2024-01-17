# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Ensure flags are initialized for e.g. pytest harness case."""

import sys

from absl import flags

# When this module is loaded in an app, flags would have been parsed already
# (assuming the app's main uses directly or indirectly absl.app.main). However,
# when loaded in a test harness like pytest or unittest (e.g. python -m pytest)
# that won't happen.
# While tests shouldn't use the flags directly, some flags - like compilation
# timeout - have default values that need to be accessible.
# This makes sure flags are initialized, for this purpose.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv, known_only=True)
assert flags.FLAGS.is_parsed()
