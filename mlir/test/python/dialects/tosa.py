# RUN: %PYTHON %s
# RUN: %PYTHON -m mypy %s --config-file %mlir_src_root/test/python/mypy.ini

from mlir.ir import *
import mlir.dialects.tosa as tosa


# Just make sure the dialect is populated with generated ops.
assert tosa.AddOp
