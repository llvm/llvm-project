//===- OMPConstants.cpp - Helpers related to OpenMP code generation ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace llvm;
using namespace omp;
using namespace types;

Directive llvm::omp::getOpenMPDirectiveKind(StringRef Str) {
  return llvm::StringSwitch<Directive>(Str)
#define OMP_DIRECTIVE(Enum, Str) .Case(Str, Enum)
#include "llvm/Frontend/OpenMP/OMPKinds.def"
      .Default(OMPD_unknown);
}

StringRef llvm::omp::getOpenMPDirectiveName(Directive Kind) {
  switch (Kind) {
#define OMP_DIRECTIVE(Enum, Str)                                               \
  case Enum:                                                                   \
    return Str;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  }
  llvm_unreachable("Invalid OpenMP directive kind");
}

/// Declarations for LLVM-IR types (simple, function and structure) are
/// generated below. Their names are defined and used in OpenMPKinds.def. Here
/// we provide the declarations, the initializeTypes function will provide the
/// values.
///
///{

#define OMP_TYPE(VarName, InitValue) Type *llvm::omp::types::VarName = nullptr;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  FunctionType *llvm::omp::types::VarName = nullptr;                           \
  PointerType *llvm::omp::types::VarName##Ptr = nullptr;
#define OMP_STRUCT_TYPE(VarName, StrName, ...)                                 \
  StructType *llvm::omp::types::VarName = nullptr;                             \
  PointerType *llvm::omp::types::VarName##Ptr = nullptr;
#include "llvm/Frontend/OpenMP/OMPKinds.def"

///}

void llvm::omp::types::initializeTypes(Module &M) {
  if (Void)
    return;

  LLVMContext &Ctx = M.getContext();
  // Create all simple and struct types exposed by the runtime and remember
  // the llvm::PointerTypes of them for easy access later.
  StructType *T;
#define OMP_TYPE(VarName, InitValue) VarName = InitValue;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  VarName = FunctionType::get(ReturnType, {__VA_ARGS__}, IsVarArg);            \
  VarName##Ptr = PointerType::getUnqual(VarName);
#define OMP_STRUCT_TYPE(VarName, StructName, ...)                              \
  T = M.getTypeByName(StructName);                                             \
  if (!T)                                                                      \
    T = StructType::create(Ctx, {__VA_ARGS__}, StructName);                    \
  VarName = T;                                                                 \
  VarName##Ptr = PointerType::getUnqual(T);
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}

void llvm::omp::types::uninitializeTypes() {
#define OMP_TYPE(VarName, InitValue) VarName = nullptr;
#define OMP_FUNCTION_TYPE(VarName, IsVarArg, ReturnType, ...)                  \
  VarName = nullptr;                                                           \
  VarName##Ptr = nullptr;
#define OMP_STRUCT_TYPE(VarName, StrName, ...)                                 \
  VarName = nullptr;                                                           \
  VarName##Ptr = nullptr;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}
