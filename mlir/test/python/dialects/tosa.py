# RUN: %PYTHON %s

from mlir.ir import *
import mlir.dialects.tosa as tosa


# Just make sure the dialect is populated with generated ops.
assert tosa.AddOp
