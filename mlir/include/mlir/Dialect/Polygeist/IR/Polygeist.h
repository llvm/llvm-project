#ifndef MLIR_DIALECT_POLYGEIST_IR_POLYGEIST_H_
#define MLIR_DIALECT_POLYGEIST_IR_POLYGEIST_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "mlir/Dialect/Polygeist/IR/PolygeistOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h.inc"

#endif // MLIR_DIALECT_POLYGEIST_IR_POLYGEIST_H_
