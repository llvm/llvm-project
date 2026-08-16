//===- TileReducerOps.h - TileReducer operations ----------------*- C++ -*-===//

#ifndef TILE_REDUCER_TILEREDUCEROPS_H
#define TILE_REDUCER_TILEREDUCEROPS_H

#include "TileReducer/TileReducerTypes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TileReducer/TileReducerOps.h.inc"

#endif // TILE_REDUCER_TILEREDUCEROPS_H
