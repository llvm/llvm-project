//===--- RuntimeDebugBuilder.cpp - Helper to insert prints into LLVM-IR ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

Function *RuntimeDebugBuilder::getVPrintF(PollyIRBuilder &Builder) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  const char *Name = "vprintf";
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(
        Builder.getInt32Ty(), {Builder.getInt8PtrTy(), Builder.getInt8PtrTy()},
        false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

Function *RuntimeDebugBuilder::getAddressSpaceCast(PollyIRBuilder &Builder,
                                                   unsigned Src, unsigned Dst,
                                                   unsigned SrcBits,
                                                   unsigned DstBits) {
  Module *M = Builder.GetInsertBlock()->getParent()->getParent();
  auto Name = std::string("llvm.nvvm.ptr.constant.to.gen.p") +
              std::to_string(Dst) + "i" + std::to_string(DstBits) + ".p" +
              std::to_string(Src) + "i" + std::to_string(SrcBits);
  Function *F = M->getFunction(Name);

  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    FunctionType *Ty = FunctionType::get(
        PointerType::get(Builder.getIntNTy(DstBits), Dst),
        PointerType::get(Builder.getIntNTy(SrcBits), Src), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  return F;
}

std::vector<Value *>
RuntimeDebugBuilder::getGPUThreadIdentifiers(PollyIRBuilder &Builder) {
  std::vector<Value *> Identifiers;

  auto M = Builder.GetInsertBlock()->getParent()->getParent();

  std::vector<Function *> BlockIDs = {
      Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_x),
      Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_y),
      Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_z),
  };

  Identifiers.push_back(Builder.CreateGlobalStringPtr("> block-id: ", "", 4));
  for (auto GetID : BlockIDs) {
    Value *Id = Builder.CreateCall(GetID, {});
    Id = Builder.CreateIntCast(Id, Builder.getInt64Ty(), false);
    Identifiers.push_back(Id);
    Identifiers.push_back(Builder.CreateGlobalStringPtr(" ", "", 4));
  }

  Identifiers.push_back(Builder.CreateGlobalStringPtr("| ", "", 4));

  std::vector<Function *> ThreadIDs = {
      Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_x),
      Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_y),
      Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_z),
  };

  Identifiers.push_back(Builder.CreateGlobalStringPtr("thread-id: ", "", 4));
  for (auto GetId : ThreadIDs) {
    Value *Id = Builder.CreateCall(GetId, {});
    Id = Builder.CreateIntCast(Id, Builder.getInt64Ty(), false);
    Identifiers.push_back(Id);
    Identifiers.push_back(Builder.CreateGlobalStringPtr(" ", "", 4));
  }

  return Identifiers;
}

void RuntimeDebugBuilder::createPrinter(PollyIRBuilder &Builder, bool IsGPU,
                                        ArrayRef<Value *> Values) {
  if (IsGPU)
    createGPUPrinterT(Builder, Values);
  else
    createCPUPrinterT(Builder, Values);
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
      if (Ty->getPointerElementType() == Builder.getInt8Ty() &&
          Ty->getPointerAddressSpace() == 4) {
        Val = Builder.CreateGEP(Val, Builder.getInt64(0));
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

void RuntimeDebugBuilder::createGPUPrinterT(PollyIRBuilder &Builder,
                                            ArrayRef<Value *> Values) {
  std::string str;

  auto *Zero = Builder.getInt64(0);

  auto ToPrint = getGPUThreadIdentifiers(Builder);

  ToPrint.push_back(Builder.CreateGlobalStringPtr("\n  ", "", 4));
  ToPrint.insert(ToPrint.end(), Values.begin(), Values.end());

  const DataLayout &DL = Builder.GetInsertBlock()->getModule()->getDataLayout();

  // Allocate print buffer (assuming 2*32 bit per element)
  auto T = ArrayType::get(Builder.getInt32Ty(), ToPrint.size() * 2);
  Value *Data = new AllocaInst(
      T, DL.getAllocaAddrSpace(), "polly.vprint.buffer",
      &Builder.GetInsertBlock()->getParent()->getEntryBlock().front());
  auto *DataPtr = Builder.CreateGEP(Data, {Zero, Zero});

  int Offset = 0;
  for (auto Val : ToPrint) {
    auto Ptr = Builder.CreateGEP(DataPtr, Builder.getInt64(Offset));
    Type *Ty = Val->getType();

    if (Ty->isFloatingPointTy()) {
      if (!Ty->isDoubleTy())
        Val = Builder.CreateFPExt(Val, Builder.getDoubleTy());
    } else if (Ty->isIntegerTy()) {
      if (Ty->getIntegerBitWidth() < 64) {
        Val = Builder.CreateSExt(Val, Builder.getInt64Ty());
      } else {
        assert(Ty->getIntegerBitWidth() == 64 &&
               "Integer types larger 64 bit not supported");
        // fallthrough
      }
    } else if (auto PtTy = dyn_cast<PointerType>(Ty)) {
      if (PtTy->getAddressSpace() == 4) {
        // Pointers in constant address space are printed as strings
        Val = Builder.CreateGEP(Val, Builder.getInt64(0));
        auto F = RuntimeDebugBuilder::getAddressSpaceCast(Builder, 4, 0);
        Val = Builder.CreateCall(F, Val);
      } else {
        Val = Builder.CreatePtrToInt(Val, Builder.getInt64Ty());
      }
    } else {
      llvm_unreachable("Unknown type");
    }

    Ty = Val->getType();
    Ptr = Builder.CreatePointerBitCastOrAddrSpaceCast(Ptr, Ty->getPointerTo(5));
    Builder.CreateAlignedStore(Val, Ptr, 4);

    if (Ty->isFloatingPointTy())
      str += "%f";
    else if (Ty->isIntegerTy())
      str += "%ld";
    else
      str += "%s";

    Offset += 2;
  }

  Value *Format = Builder.CreateGlobalStringPtr(str, "polly.vprintf.buffer", 4);
  Format = Builder.CreateCall(getAddressSpaceCast(Builder, 4, 0), Format);

  Data = Builder.CreateBitCast(Data, Builder.getInt8PtrTy());

  Builder.CreateCall(getVPrintF(Builder), {Format, Data});
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
        FunctionType::get(Builder.getInt32Ty(), Builder.getInt8PtrTy(), false);
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
