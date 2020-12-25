//=== AMDGPUPrintfRuntimeBinding.cpp - OpenCL printf implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
//
// The pass bind printfs to a kernel arg pointer that will be bound to a buffer
// later by the runtime.
//
// This pass traverses the functions in the module and converts
// each call to printf to a sequence of operations that
// store the following into the printf buffer:
// - format string (passed as a module's metadata unique ID)
// - bitwise copies of printf arguments
// The backend passes will need to store metadata in the kernel
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "printfToRuntime"
#define DWORD_ALIGN 4

namespace {
class AMDGPUPrintfRuntimeBinding final : public ModulePass {

public:
  static char ID;

  explicit AMDGPUPrintfRuntimeBinding();

private:
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
};

class AMDGPUPrintfRuntimeBindingImpl {
public:
  AMDGPUPrintfRuntimeBindingImpl(
      function_ref<const DominatorTree &(Function &)> GetDT,
      function_ref<const TargetLibraryInfo &(Function &)> GetTLI)
      : GetDT(GetDT), GetTLI(GetTLI) {}
  bool run(Module &M);

private:
  void getConversionSpecifiers(SmallVectorImpl<char> &OpConvSpecifiers,
                               StringRef fmt, size_t num_ops) const;

  bool shouldPrintAsStr(char Specifier, Type *OpType) const;
  bool lowerPrintfForGpu(Module &M);

  Value *simplify(Instruction *I, const TargetLibraryInfo *TLI,
                  const DominatorTree *DT) {
    return SimplifyInstruction(I, {*TD, TLI, DT});
  }

  const DataLayout *TD;
  function_ref<const DominatorTree &(Function &)> GetDT;
  function_ref<const TargetLibraryInfo &(Function &)> GetTLI;
  SmallVector<CallInst *, 32> Printfs;
};
} // namespace

char AMDGPUPrintfRuntimeBinding::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPUPrintfRuntimeBinding,
                      "amdgpu-printf-runtime-binding", "AMDGPU Printf lowering",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(AMDGPUPrintfRuntimeBinding, "amdgpu-printf-runtime-binding",
                    "AMDGPU Printf lowering", false, false)

char &llvm::AMDGPUPrintfRuntimeBindingID = AMDGPUPrintfRuntimeBinding::ID;

namespace llvm {
ModulePass *createAMDGPUPrintfRuntimeBinding() {
  return new AMDGPUPrintfRuntimeBinding();
}
} // namespace llvm

AMDGPUPrintfRuntimeBinding::AMDGPUPrintfRuntimeBinding() : ModulePass(ID) {
  initializeAMDGPUPrintfRuntimeBindingPass(*PassRegistry::getPassRegistry());
}

void AMDGPUPrintfRuntimeBindingImpl::getConversionSpecifiers(
    SmallVectorImpl<char> &OpConvSpecifiers, StringRef Fmt,
    size_t NumOps) const {
  // not all format characters are collected.
  // At this time the format characters of interest
  // are %p and %s, which use to know if we
  // are either storing a literal string or a
  // pointer to the printf buffer.
  static const char ConvSpecifiers[] = "cdieEfgGaosuxXp";
  size_t CurFmtSpecifierIdx = 0;
  size_t PrevFmtSpecifierIdx = 0;

  while ((CurFmtSpecifierIdx = Fmt.find_first_of(
              ConvSpecifiers, CurFmtSpecifierIdx)) != StringRef::npos) {
    bool ArgDump = false;
    StringRef CurFmt = Fmt.substr(PrevFmtSpecifierIdx,
                                  CurFmtSpecifierIdx - PrevFmtSpecifierIdx);
    size_t pTag = CurFmt.find_last_of("%");
    if (pTag != StringRef::npos) {
      ArgDump = true;
      while (pTag && CurFmt[--pTag] == '%') {
        ArgDump = !ArgDump;
      }
    }

    if (ArgDump)
      OpConvSpecifiers.push_back(Fmt[CurFmtSpecifierIdx]);

    PrevFmtSpecifierIdx = ++CurFmtSpecifierIdx;
  }
}

bool AMDGPUPrintfRuntimeBindingImpl::shouldPrintAsStr(char Specifier,
                                                      Type *OpType) const {
  if (Specifier != 's')
    return false;
  const PointerType *PT = dyn_cast<PointerType>(OpType);
  if (!PT || PT->getAddressSpace() != AMDGPUAS::CONSTANT_ADDRESS)
    return false;
  Type *ElemType = PT->getContainedType(0);
  if (ElemType->getTypeID() != Type::IntegerTyID)
    return false;
  IntegerType *ElemIType = cast<IntegerType>(ElemType);
  return ElemIType->getBitWidth() == 8;
}

bool AMDGPUPrintfRuntimeBindingImpl::lowerPrintfForGpu(Module &M) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> Builder(Ctx);
  Type *I32Ty = Type::getInt32Ty(Ctx);
  unsigned UniqID = 0;
  // NB: This is important for this string size to be divizable by 4
  const char NonLiteralStr[4] = "???";

  for (auto CI : Printfs) {
    unsigned NumOps = CI->getNumArgOperands();

    SmallString<16> OpConvSpecifiers;
    Value *Op = CI->getArgOperand(0);

    if (auto LI = dyn_cast<LoadInst>(Op)) {
      Op = LI->getPointerOperand();
      for (auto Use : Op->users()) {
        if (auto SI = dyn_cast<StoreInst>(Use)) {
          Op = SI->getValueOperand();
          break;
        }
      }
    }

    if (auto I = dyn_cast<Instruction>(Op)) {
      Value *Op_simplified =
          simplify(I, &GetTLI(*I->getFunction()), &GetDT(*I->getFunction()));
      if (Op_simplified)
        Op = Op_simplified;
    }

    ConstantExpr *ConstExpr = dyn_cast<ConstantExpr>(Op);

    if (ConstExpr) {
      GlobalVariable *GVar = dyn_cast<GlobalVariable>(ConstExpr->getOperand(0));

      StringRef Str("unknown");
      if (GVar && GVar->hasInitializer()) {
        auto Init = GVar->getInitializer();
        if (auto CA = dyn_cast<ConstantDataArray>(Init)) {
          if (CA->isString())
            Str = CA->getAsCString();
        } else if (isa<ConstantAggregateZero>(Init)) {
          Str = "";
        }
        //
        // we need this call to ascertain
        // that we are printing a string
        // or a pointer. It takes out the
        // specifiers and fills up the first
        // arg
        getConversionSpecifiers(OpConvSpecifiers, Str, NumOps - 1);
      }
      // Add metadata for the string
      std::string AStreamHolder;
      raw_string_ostream Sizes(AStreamHolder);
      int Sum = DWORD_ALIGN;
      Sizes << CI->getNumArgOperands() - 1;
      Sizes << ':';
      for (unsigned ArgCount = 1; ArgCount < CI->getNumArgOperands() &&
                                  ArgCount <= OpConvSpecifiers.size();
           ArgCount++) {
        Value *Arg = CI->getArgOperand(ArgCount);
        Type *ArgType = Arg->getType();
        unsigned ArgSize = TD->getTypeAllocSizeInBits(ArgType);
        ArgSize = ArgSize / 8;
        //
        // ArgSize by design should be a multiple of DWORD_ALIGN,
        // expand the arguments that do not follow this rule.
        //
        if (ArgSize % DWORD_ALIGN != 0) {
          llvm::Type *ResType = llvm::Type::getInt32Ty(Ctx);
          auto *LLVMVecType = llvm::dyn_cast<llvm::FixedVectorType>(ArgType);
          int NumElem = LLVMVecType ? LLVMVecType->getNumElements() : 1;
          if (LLVMVecType && NumElem > 1)
            ResType = llvm::FixedVectorType::get(ResType, NumElem);
          Builder.SetInsertPoint(CI);
          Builder.SetCurrentDebugLocation(CI->getDebugLoc());
          if (OpConvSpecifiers[ArgCount - 1] == 'x' ||
              OpConvSpecifiers[ArgCount - 1] == 'X' ||
              OpConvSpecifiers[ArgCount - 1] == 'u' ||
              OpConvSpecifiers[ArgCount - 1] == 'o')
            Arg = Builder.CreateZExt(Arg, ResType);
          else
            Arg = Builder.CreateSExt(Arg, ResType);
          ArgType = Arg->getType();
          ArgSize = TD->getTypeAllocSizeInBits(ArgType);
          ArgSize = ArgSize / 8;
          CI->setOperand(ArgCount, Arg);
        }
        if (OpConvSpecifiers[ArgCount - 1] == 'f') {
          ConstantFP *FpCons = dyn_cast<ConstantFP>(Arg);
          if (FpCons)
            ArgSize = 4;
          else {
            FPExtInst *FpExt = dyn_cast<FPExtInst>(Arg);
            if (FpExt && FpExt->getType()->isDoubleTy() &&
                FpExt->getOperand(0)->getType()->isFloatTy())
              ArgSize = 4;
          }
        }
        if (shouldPrintAsStr(OpConvSpecifiers[ArgCount - 1], ArgType)) {
          if (ConstantExpr *ConstExpr = dyn_cast<ConstantExpr>(Arg)) {
            GlobalVariable *GV =
                dyn_cast<GlobalVariable>(ConstExpr->getOperand(0));
            if (GV && GV->hasInitializer()) {
              Constant *Init = GV->getInitializer();
              ConstantDataArray *CA = dyn_cast<ConstantDataArray>(Init);
              if (Init->isZeroValue() || CA->isString()) {
                size_t SizeStr = Init->isZeroValue()
                                     ? 1
                                     : (strlen(CA->getAsCString().data()) + 1);
                size_t Rem = SizeStr % DWORD_ALIGN;
                size_t NSizeStr = 0;
                LLVM_DEBUG(dbgs() << "Printf string original size = " << SizeStr
                                  << '\n');
                if (Rem) {
                  NSizeStr = SizeStr + (DWORD_ALIGN - Rem);
                } else {
                  NSizeStr = SizeStr;
                }
                ArgSize = NSizeStr;
              }
            } else {
              ArgSize = sizeof(NonLiteralStr);
            }
          } else {
            ArgSize = sizeof(NonLiteralStr);
          }
        }
        LLVM_DEBUG(dbgs() << "Printf ArgSize (in buffer) = " << ArgSize
                          << " for type: " << *ArgType << '\n');
        Sizes << ArgSize << ':';
        Sum += ArgSize;
      }
      LLVM_DEBUG(dbgs() << "Printf format string in source = " << Str.str()
                        << '\n');
      for (size_t I = 0; I < Str.size(); ++I) {
        // Rest of the C escape sequences (e.g. \') are handled correctly
        // by the MDParser
        switch (Str[I]) {
        case '\a':
          Sizes << "\\a";
          break;
        case '\b':
          Sizes << "\\b";
          break;
        case '\f':
          Sizes << "\\f";
          break;
        case '\n':
          Sizes << "\\n";
          break;
        case '\r':
          Sizes << "\\r";
          break;
        case '\v':
          Sizes << "\\v";
          break;
        case ':':
          // ':' cannot be scanned by Flex, as it is defined as a delimiter
          // Replace it with it's octal representation \72
          Sizes << "\\72";
          break;
        default:
          Sizes << Str[I];
          break;
        }
      }

      // Insert the printf_alloc call
      Builder.SetInsertPoint(CI);
      Builder.SetCurrentDebugLocation(CI->getDebugLoc());

      AttributeList Attr = AttributeList::get(Ctx, AttributeList::FunctionIndex,
                                              Attribute::NoUnwind);

      Type *SizetTy = Type::getInt32Ty(Ctx);

      Type *Tys_alloc[1] = {SizetTy};
      Type *I8Ptr = PointerType::get(Type::getInt8Ty(Ctx), 1);
      FunctionType *FTy_alloc = FunctionType::get(I8Ptr, Tys_alloc, false);
      FunctionCallee PrintfAllocFn =
          M.getOrInsertFunction(StringRef("__printf_alloc"), FTy_alloc, Attr);

      LLVM_DEBUG(dbgs() << "Printf metadata = " << Sizes.str() << '\n');
      std::string fmtstr = itostr(++UniqID) + ":" + Sizes.str().c_str();
      MDString *fmtStrArray = MDString::get(Ctx, fmtstr);

      // Instead of creating global variables, the
      // printf format strings are extracted
      // and passed as metadata. This avoids
      // polluting llvm's symbol tables in this module.
      // Metadata is going to be extracted
      // by the backend passes and inserted
      // into the OpenCL binary as appropriate.
      StringRef amd("llvm.printf.fmts");
      NamedMDNode *metaD = M.getOrInsertNamedMetadata(amd);
      MDNode *myMD = MDNode::get(Ctx, fmtStrArray);
      metaD->addOperand(myMD);
      Value *sumC = ConstantInt::get(SizetTy, Sum, false);
      SmallVector<Value *, 1> alloc_args;
      alloc_args.push_back(sumC);
      CallInst *pcall =
          CallInst::Create(PrintfAllocFn, alloc_args, "printf_alloc_fn", CI);

      //
      // Insert code to split basicblock with a
      // piece of hammock code.
      // basicblock splits after buffer overflow check
      //
      ConstantPointerNull *zeroIntPtr =
          ConstantPointerNull::get(PointerType::get(Type::getInt8Ty(Ctx), 1));
      ICmpInst *cmp =
          dyn_cast<ICmpInst>(Builder.CreateICmpNE(pcall, zeroIntPtr, ""));
      if (!CI->use_empty()) {
        Value *result =
            Builder.CreateSExt(Builder.CreateNot(cmp), I32Ty, "printf_res");
        CI->replaceAllUsesWith(result);
      }
      SplitBlock(CI->getParent(), cmp);
      Instruction *Brnch =
          SplitBlockAndInsertIfThen(cmp, cmp->getNextNode(), false);

      Builder.SetInsertPoint(Brnch);

      // store unique printf id in the buffer
      //
      SmallVector<Value *, 1> ZeroIdxList;
      ConstantInt *zeroInt =
          ConstantInt::get(Ctx, APInt(32, StringRef("0"), 10));
      ZeroIdxList.push_back(zeroInt);

      GetElementPtrInst *BufferIdx = GetElementPtrInst::Create(
          nullptr, pcall, ZeroIdxList, "PrintBuffID", Brnch);

      Type *idPointer = PointerType::get(I32Ty, AMDGPUAS::GLOBAL_ADDRESS);
      Value *id_gep_cast =
          new BitCastInst(BufferIdx, idPointer, "PrintBuffIdCast", Brnch);

      new StoreInst(ConstantInt::get(I32Ty, UniqID), id_gep_cast, Brnch);

      SmallVector<Value *, 2> FourthIdxList;
      ConstantInt *fourInt =
          ConstantInt::get(Ctx, APInt(32, StringRef("4"), 10));

      FourthIdxList.push_back(fourInt); // 1st 4 bytes hold the printf_id
      // the following GEP is the buffer pointer
      BufferIdx = GetElementPtrInst::Create(nullptr, pcall, FourthIdxList,
                                            "PrintBuffGep", Brnch);

      Type *Int32Ty = Type::getInt32Ty(Ctx);
      Type *Int64Ty = Type::getInt64Ty(Ctx);
      for (unsigned ArgCount = 1; ArgCount < CI->getNumArgOperands() &&
                                  ArgCount <= OpConvSpecifiers.size();
           ArgCount++) {
        Value *Arg = CI->getArgOperand(ArgCount);
        Type *ArgType = Arg->getType();
        SmallVector<Value *, 32> WhatToStore;
        if (ArgType->isFPOrFPVectorTy() && !isa<VectorType>(ArgType)) {
          Type *IType = (ArgType->isFloatTy()) ? Int32Ty : Int64Ty;
          if (OpConvSpecifiers[ArgCount - 1] == 'f') {
            if (auto *FpCons = dyn_cast<ConstantFP>(Arg)) {
              APFloat Val(FpCons->getValueAPF());
              bool Lost = false;
              Val.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                          &Lost);
              Arg = ConstantFP::get(Ctx, Val);
              IType = Int32Ty;
            } else if (auto *FpExt = dyn_cast<FPExtInst>(Arg)) {
              if (FpExt->getType()->isDoubleTy() &&
                  FpExt->getOperand(0)->getType()->isFloatTy()) {
                Arg = FpExt->getOperand(0);
                IType = Int32Ty;
              }
            }
          }
          Arg = new BitCastInst(Arg, IType, "PrintArgFP", Brnch);
          WhatToStore.push_back(Arg);
        } else if (ArgType->getTypeID() == Type::PointerTyID) {
          if (shouldPrintAsStr(OpConvSpecifiers[ArgCount - 1], ArgType)) {
            const char *S = NonLiteralStr;
            if (auto *ConstExpr = dyn_cast<ConstantExpr>(Arg)) {
              auto *GV = dyn_cast<GlobalVariable>(ConstExpr->getOperand(0));
              if (GV && GV->hasInitializer()) {
                Constant *Init = GV->getInitializer();
                ConstantDataArray *CA = dyn_cast<ConstantDataArray>(Init);
                if (Init->isZeroValue() || CA->isString()) {
                  S = Init->isZeroValue() ? "" : CA->getAsCString().data();
                }
              }
            }
            size_t SizeStr = strlen(S) + 1;
            size_t Rem = SizeStr % DWORD_ALIGN;
            size_t NSizeStr = 0;
            if (Rem) {
              NSizeStr = SizeStr + (DWORD_ALIGN - Rem);
            } else {
              NSizeStr = SizeStr;
            }
            if (S[0]) {
              char *MyNewStr = new char[NSizeStr]();
              strcpy(MyNewStr, S);
              int NumInts = NSizeStr / 4;
              int CharC = 0;
              while (NumInts) {
                int ANum = *(int *)(MyNewStr + CharC);
                CharC += 4;
                NumInts--;
                Value *ANumV = ConstantInt::get(Int32Ty, ANum, false);
                WhatToStore.push_back(ANumV);
              }
              delete[] MyNewStr;
            } else {
              // Empty string, give a hint to RT it is no NULL
              Value *ANumV = ConstantInt::get(Int32Ty, 0xFFFFFF00, false);
              WhatToStore.push_back(ANumV);
            }
          } else {
            uint64_t Size = TD->getTypeAllocSizeInBits(ArgType);
            assert((Size == 32 || Size == 64) && "unsupported size");
            Type *DstType = (Size == 32) ? Int32Ty : Int64Ty;
            Arg = new PtrToIntInst(Arg, DstType, "PrintArgPtr", Brnch);
            WhatToStore.push_back(Arg);
          }
        } else if (isa<FixedVectorType>(ArgType)) {
          Type *IType = NULL;
          uint32_t EleCount = cast<FixedVectorType>(ArgType)->getNumElements();
          uint32_t EleSize = ArgType->getScalarSizeInBits();
          uint32_t TotalSize = EleCount * EleSize;
          if (EleCount == 3) {
            ShuffleVectorInst *Shuffle =
                new ShuffleVectorInst(Arg, Arg, ArrayRef<int>{0, 1, 2, 2});
            Shuffle->insertBefore(Brnch);
            Arg = Shuffle;
            ArgType = Arg->getType();
            TotalSize += EleSize;
          }
          switch (EleSize) {
          default:
            EleCount = TotalSize / 64;
            IType = Type::getInt64Ty(ArgType->getContext());
            break;
          case 8:
            if (EleCount >= 8) {
              EleCount = TotalSize / 64;
              IType = Type::getInt64Ty(ArgType->getContext());
            } else if (EleCount >= 3) {
              EleCount = 1;
              IType = Type::getInt32Ty(ArgType->getContext());
            } else {
              EleCount = 1;
              IType = Type::getInt16Ty(ArgType->getContext());
            }
            break;
          case 16:
            if (EleCount >= 3) {
              EleCount = TotalSize / 64;
              IType = Type::getInt64Ty(ArgType->getContext());
            } else {
              EleCount = 1;
              IType = Type::getInt32Ty(ArgType->getContext());
            }
            break;
          }
          if (EleCount > 1) {
            IType = FixedVectorType::get(IType, EleCount);
          }
          Arg = new BitCastInst(Arg, IType, "PrintArgVect", Brnch);
          WhatToStore.push_back(Arg);
        } else {
          WhatToStore.push_back(Arg);
        }
        for (unsigned I = 0, E = WhatToStore.size(); I != E; ++I) {
          Value *TheBtCast = WhatToStore[I];
          unsigned ArgSize =
              TD->getTypeAllocSizeInBits(TheBtCast->getType()) / 8;
          SmallVector<Value *, 1> BuffOffset;
          BuffOffset.push_back(ConstantInt::get(I32Ty, ArgSize));

          Type *ArgPointer = PointerType::get(TheBtCast->getType(), 1);
          Value *CastedGEP =
              new BitCastInst(BufferIdx, ArgPointer, "PrintBuffPtrCast", Brnch);
          StoreInst *StBuff = new StoreInst(TheBtCast, CastedGEP, Brnch);
          LLVM_DEBUG(dbgs() << "inserting store to printf buffer:\n"
                            << *StBuff << '\n');
          (void)StBuff;
          if (I + 1 == E && ArgCount + 1 == CI->getNumArgOperands())
            break;
          BufferIdx = GetElementPtrInst::Create(nullptr, BufferIdx, BuffOffset,
                                                "PrintBuffNextPtr", Brnch);
          LLVM_DEBUG(dbgs() << "inserting gep to the printf buffer:\n"
                            << *BufferIdx << '\n');
        }
      }
    }
  }

  // erase the printf calls
  for (auto CI : Printfs)
    CI->eraseFromParent();

  Printfs.clear();
  return true;
}

bool AMDGPUPrintfRuntimeBindingImpl::run(Module &M) {
  Triple TT(M.getTargetTriple());
  if (TT.getArch() == Triple::r600)
    return false;

  auto PrintfFunction = M.getFunction("printf");
  if (!PrintfFunction)
    return false;

  for (auto &U : PrintfFunction->uses()) {
    if (auto *CI = dyn_cast<CallInst>(U.getUser())) {
      if (CI->isCallee(&U))
        Printfs.push_back(CI);
    }
  }

  if (Printfs.empty())
    return false;

  if (auto HostcallFunction = M.getFunction("__ockl_hostcall_internal")) {
    for (auto &U : HostcallFunction->uses()) {
      if (auto *CI = dyn_cast<CallInst>(U.getUser())) {
        M.getContext().emitError(
            CI, "Cannot use both printf and hostcall in the same module");
      }
    }
  }

  TD = &M.getDataLayout();

  return lowerPrintfForGpu(M);
}

bool AMDGPUPrintfRuntimeBinding::runOnModule(Module &M) {
  auto GetDT = [this](Function &F) -> DominatorTree & {
    return this->getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  };
  auto GetTLI = [this](Function &F) -> TargetLibraryInfo & {
    return this->getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  };

  return AMDGPUPrintfRuntimeBindingImpl(GetDT, GetTLI).run(M);
}

PreservedAnalyses
AMDGPUPrintfRuntimeBindingPass::run(Module &M, ModuleAnalysisManager &AM) {
  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetDT = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  bool Changed = AMDGPUPrintfRuntimeBindingImpl(GetDT, GetTLI).run(M);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
