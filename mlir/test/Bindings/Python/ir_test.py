# RUN: %PYTHON %s | FileCheck %s

import mlir

TEST_MLIR_ASM = r"""
module {
}
"""

ctx = mlir.ir.MlirContext()
module = ctx.parse(TEST_MLIR_ASM)
module.dump()
print(bool(module))
# CHECK: True
