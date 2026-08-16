//===- TileReducerDialect.cpp - TileReducer dialect -------------*- C++ -*-===//

#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerOps.h"

using namespace mlir;
using namespace mlir::tr;

#include "TileReducer/TileReducerOpsDialect.cpp.inc"

void TileReducerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TileReducer/TileReducerOps.cpp.inc"
      >();
}
