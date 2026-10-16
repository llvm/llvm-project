//===- TileReducerTransformOps.h - transform.tr.* ---------------*- C++ -*-===//

#ifndef TILE_REDUCER_TILEREDUCERTRANSFORMOPS_H
#define TILE_REDUCER_TILEREDUCERTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TileReducer/TileReducerTransformOps.h.inc"

namespace mlir {
class DialectRegistry;
namespace tr {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace tr
} // namespace mlir

#endif // TILE_REDUCER_TILEREDUCERTRANSFORMOPS_H
