#ifndef MLIR_CAPI_PRESBURGER_H
#define MLIR_CAPI_PRESBURGER_H

#include "mlir-c/Presburger.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/CAPI/Wrap.h"
#include "llvm/ADT/DynamicAPInt.h"

DEFINE_C_API_PTR_METHODS(MlirPresburgerIntegerRelation,
                         mlir::presburger::IntegerRelation)

static inline MlirPresburgerDynamicAPInt wrap(llvm::DynamicAPInt *cpp) {
  return MlirPresburgerDynamicAPInt{cpp->getAsOpaquePointer()};
}

static inline llvm::DynamicAPInt *unwrap(MlirPresburgerDynamicAPInt c) {
  return llvm::DynamicAPInt::getFromOpaquePointer(c.ptr);
}

static inline MlirPresburgerVariableKind wrap(mlir::presburger::VarKind var) {
  return static_cast<MlirPresburgerVariableKind>(var);
}

static inline mlir::presburger::VarKind unwarp(MlirPresburgerVariableKind var) {
  return static_cast<mlir::presburger::VarKind>(var);
}

#endif /* MLIR_CAPI_PRESBURGER_H */