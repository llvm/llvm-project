#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Note that the only function of this module is currently to load the
# native module and re-export its symbols. In the future, this file is
# reserved as a trampoline to handle environment specific loading needs
# and arbitrate any one-time initialization needed in various shared-library
# scenarios.

from _mlir import *
