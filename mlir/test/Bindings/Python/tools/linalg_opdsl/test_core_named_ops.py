# RUN: %PYTHON -m mlir.tools.linalg_opdsl.dump_oplib .ops.core_named_ops | FileCheck %s

# Just verify that at least one known op is generated.
# CHECK: name: matmul
