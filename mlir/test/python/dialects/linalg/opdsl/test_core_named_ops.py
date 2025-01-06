# RUN: %PYTHON -m mlir.dialects.linalg.opdsl.dump_oplib .ops.core_named_ops | FileCheck %s

# Just verify that at least one known op is generated.
# CHECK: name: matmul

# verify some special cases: negf->NegFOp, powf->PowFOp
# CHECK cpp_class_name: NegFOp
# CHECK cpp_class_name: PowFOp
