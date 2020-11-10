#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simply a wrapper around the extension module of the same name.
from . import _reexport_cext
_reexport_cext("passmanager", __name__)
del _reexport_cext
