# RUN: %PYTHON %s

import doctest
import importlib

def test_module(module_name):
  print(f"--- Testing module: {module_name}")
  m = importlib.import_module(module_name)
  doctest.testmod(m, verbose=True, raise_on_error=True, report=True)


test_module("mlir.tools.linalg_opdsl.lang.affine")
test_module("mlir.tools.linalg_opdsl.lang.types")
