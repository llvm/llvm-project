//===- SPIRVOpUtils.h - MLIR SPIR-V Dialect Op Definition Utilities -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

namespace mlir::spirv {

/// Returns the bit width of the `type`.
inline unsigned getBitWidth(Type type) {
  if (isa<spirv::PointerType>(type)) {
    // Just return 64 bits for pointer types for now.
    // TODO: Make sure not caller relies on the actual pointer width value.
    return 64;
  }

  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();

  if (auto vectorType = dyn_cast<VectorType>(type)) {
    assert(vectorType.getElementType().isIntOrFloat());
    return vectorType.getNumElements() *
           vectorType.getElementType().getIntOrFloatBitWidth();
  }
  llvm_unreachable("unhandled bit width computation for type");
}

void printVariableDecorations(Operation *op, OpAsmPrinter &printer,
                              SmallVectorImpl<StringRef> &elidedAttrs);

LogicalResult extractValueFromConstOp(Operation *op, int32_t &value);

LogicalResult verifyMemorySemantics(Operation *op,
                                    spirv::MemorySemantics memorySemantics);

} // namespace mlir::spirv
