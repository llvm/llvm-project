# REQUIRES: bindings_python
# RUN: %PYTHON% %s | FileCheck %s

import mlir

from mlir.dialects import smt
from mlir.ir import Context, Location, Module, InsertionPoint

with Context() as ctx, Location.unknown():
    m = Module.create()
    with InsertionPoint(m.body):
        true = smt.constant(True)
        false = smt.constant(False)
    # CHECK: smt.constant true
    # CHECK: smt.constant false
    print(m)
