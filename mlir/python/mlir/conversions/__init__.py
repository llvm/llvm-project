#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Expose the corresponding C-Extension module with a well-known name at this
# level.
from .._mlir_libs import _mlirConversions as _cextConversions
