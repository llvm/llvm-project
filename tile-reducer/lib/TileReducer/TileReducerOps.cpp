//===- TileReducerOps.cpp - TileReducer operations --------------*- C++ -*-===//

#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerDialect.h"

using namespace mlir;
using namespace mlir::tr;

#define GET_OP_CLASSES
#include "TileReducer/TileReducerOps.cpp.inc"
