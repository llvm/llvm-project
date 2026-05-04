#ifndef ReducerOps_H
#define ReducerOps_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Reducer/IR/ReducerOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "mlir/Reducer/IR/ReducerOps.h.inc"

#endif // ReducerOps_H
