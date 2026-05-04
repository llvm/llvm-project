#include "mlir/Reducer/IR/ReducerOps.h"

using namespace mlir;
using namespace reducer;

#include "mlir/Reducer/IR/ReducerOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Reducer/IR/ReducerOps.cpp.inc"

void ReducerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Reducer/IR/ReducerOps.cpp.inc"
      >();
}
