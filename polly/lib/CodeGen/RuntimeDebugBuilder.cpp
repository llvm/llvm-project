//===--- RuntimeDebugBuilder.cpp - Helper to insert prints into LLVM-IR ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "llvm/IR/Module.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

llvm::Value *RuntimeDebugBuilder::getPrintableString(PollyIRBuilder &Builder,
                                                     llvm::StringRef Str) {
  // FIXME: addressspace(4) is a marker for a string (for the %s conversion
  // specifier) but should be using the default address space. This only works
  // because CPU backends typically ignore the address space. For constant
  // strings as returned by getPrintableString, the format string should instead
  // directly spell out the string.
  return Builder.CreateGlobalStringPtr(Str, "", 4);
}

Function *RuntimeDebugBuilder::getVPrintF(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "vprintf";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(
        Builder.getInt32Ty(), {Builder.getPtrTy(), Builder.getPtrTy()}, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

void RuntimeDebugBuilder::createPrinter(PollyIRBuilder &Builder,
                                        ArrayRef<Value *> Values) {
  createCPUPrinterT(Builder, Values);
}

bool RuntimeDebugBuilder::isPrintable(Type *Ty) {
  if (Ty->isFloatingPointTy())
    return true;

  if (Ty->isIntegerTy())
    return Ty->getIntegerBitWidth() <= 64;

  if (isa<PointerType>(Ty))
    return true;

  return false;
}

static std::tuple<std::string, std::vector<Value *>>
prepareValuesForPrinting(PollyIRBuilder &Builder, ArrayRef<Value *> Values) {
  std::string FormatString;
  std::vector<Value *> ValuesToPrint;

  for (auto Val : Values) {
    Type *Ty = Val->getType();

    if (Ty->isFloatingPointTy()) {
      if (!Ty->isDoubleTy())
        Val = Builder.CreateFPExt(Val, Builder.getDoubleTy());
    } else if (Ty->isIntegerTy()) {
      if (Ty->getIntegerBitWidth() < 64)
        Val = Builder.CreateSExt(Val, Builder.getInt64Ty());
      else
        assert(Ty->getIntegerBitWidth() &&
               "Integer types larger 64 bit not supported");
    } else if (isa<PointerType>(Ty)) {
      if (Ty == Builder.getPtrTy(4)) {
        Val = Builder.CreateGEP(Builder.getInt8Ty(), Val, Builder.getInt64(0));
      } else {
        Val = Builder.CreatePtrToInt(Val, Builder.getInt64Ty());
      }
    } else {
      llvm_unreachable("Unknown type");
    }

    Ty = Val->getType();

    if (Ty->isFloatingPointTy())
      FormatString += "%f";
    else if (Ty->isIntegerTy())
      FormatString += "%ld";
    else
      FormatString += "%s";

    ValuesToPrint.push_back(Val);
  }

  return std::make_tuple(FormatString, ValuesToPrint);
}

void RuntimeDebugBuilder::createCPUPrinterT(PollyIRBuilder &Builder,
                                            ArrayRef<Value *> Values) {

  std::string FormatString;
  std::vector<Value *> ValuesToPrint;

  std::tie(FormatString, ValuesToPrint) =
      prepareValuesForPrinting(Builder, Values);

  createPrintF(Builder, FormatString, ValuesToPrint);
  createFlush(Builder);
}

Function *RuntimeDebugBuilder::getPrintF(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "printf";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(Builder.getInt32Ty(), true);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

void RuntimeDebugBuilder::createPrintF(PollyIRBuilder &Builder,
                                       std::string Format,
                                       ArrayRef<Value *> Values) {
  Value *FormatString = Builder.CreateGlobalStringPtr(Format);
  std::vector<Value *> Arguments;

  Arguments.push_back(FormatString);
  Arguments.insert(Arguments.end(), Values.begin(), Values.end());
  Builder.CreateCall(getPrintF(Builder), Arguments);
}

void RuntimeDebugBuilder::createFlush(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "fflush";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty =
        FunctionType::get(Builder.getInt32Ty(), Builder.getPtrTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  // fflush(NULL) flushes _all_ open output streams.
  //
  // fflush is declared as 'int fflush(FILE *stream)'. As we only pass on a NULL
  // pointer, the type we point to does conceptually not matter. However, if
  // fflush is already declared in this translation unit, we use the very same
  // type to ensure that LLVM does not complain about mismatching types.
  Builder.CreateCall(F, Constant::getNullValue(F->arg_begin()->getType()));
}
