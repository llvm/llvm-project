#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Expose the corresponding C-Extension module with a well-known name at this
# level.
from .. import _load_extension
_cextConversions = _load_extension("_mlirConversions")
