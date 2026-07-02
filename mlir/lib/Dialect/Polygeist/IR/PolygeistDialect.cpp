#include "mlir/Dialect/Polygeist/IR/Polygeist.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::polygeist;

#include "mlir/Dialect/Polygeist/IR/PolygeistOpsDialect.cpp.inc"

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.cpp.inc"
      >();
}
