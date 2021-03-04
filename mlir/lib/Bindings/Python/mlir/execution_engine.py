#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simply a wrapper around the extension module of the same name.
from . import _cext
import ctypes

class ExecutionEngine(_cext.execution_engine.ExecutionEngine):

  def lookup(self, name):
    """Lookup a function emitted with the `llvm.emit_c_interface`
    attribute and returns a ctype callable.
    Raise a RuntimeError if the function isn't found.
    """
    func = self.raw_lookup("_mlir_ciface_" + name)
    if not func:
      raise RuntimeError("Unknown function " + name)
    prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    return prototype(func)

  def invoke(self, name, *ctypes_args):
    """Invoke a function with the list of ctypes arguments.
    All arguments must be pointers.
    Raise a RuntimeError if the function isn't found.
    """
    func = self.lookup(name)
    packed_args = (ctypes.c_void_p * len(ctypes_args))()
    for argNum in range(len(ctypes_args)):
      packed_args[argNum] = ctypes.cast(ctypes_args[argNum], ctypes.c_void_p)
    func(packed_args)
