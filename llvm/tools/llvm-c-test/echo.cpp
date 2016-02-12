//===-- echo.cpp - tool for testing libLLVM and llvm-c API ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the --echo commands in llvm-c-test.
//
// This command uses the C API to read a module and output an exact copy of it
// as output. It is used to check that the resulting module matches the input
// to validate that the C API can read and write modules properly.
//
//===----------------------------------------------------------------------===//

#include "llvm-c-test.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"

#include <stdio.h>
#include <stdlib.h>

using namespace llvm;

// Provide DenseMapInfo for C API opaque types.
template<typename T>
struct CAPIDenseMap {};

// The default DenseMapInfo require to know about pointer alignement.
// Because the C API uses opaques pointer types, their alignement is unknown.
// As a result, we need to roll out our own implementation.
template<typename T>
struct CAPIDenseMap<T*> {
  struct CAPIDenseMapInfo {
    static inline T* getEmptyKey() {
      uintptr_t Val = static_cast<uintptr_t>(-1);
      return reinterpret_cast<T*>(Val);
    }
    static inline T* getTombstoneKey() {
      uintptr_t Val = static_cast<uintptr_t>(-2);
      return reinterpret_cast<T*>(Val);
    }
    static unsigned getHashValue(const T *PtrVal) {
      return hash_value(PtrVal);
    }
    static bool isEqual(const T *LHS, const T *RHS) { return LHS == RHS; }
  };

  typedef DenseMap<T*, T*, CAPIDenseMapInfo> Map;
};

typedef CAPIDenseMap<LLVMValueRef>::Map ValueMap;
typedef CAPIDenseMap<LLVMBasicBlockRef>::Map BasicBlockMap;

struct TypeCloner {
  LLVMModuleRef M;
  LLVMContextRef Ctx;

  TypeCloner(LLVMModuleRef M): M(M), Ctx(LLVMGetModuleContext(M)) {}

  LLVMTypeRef Clone(LLVMTypeRef Src) {
    LLVMTypeKind Kind = LLVMGetTypeKind(Src);
    switch (Kind) {
      case LLVMVoidTypeKind:
        return LLVMVoidTypeInContext(Ctx);
      case LLVMHalfTypeKind:
        return LLVMHalfTypeInContext(Ctx);
      case LLVMFloatTypeKind:
        return LLVMFloatTypeInContext(Ctx);
      case LLVMDoubleTypeKind:
        return LLVMDoubleTypeInContext(Ctx);
      case LLVMX86_FP80TypeKind:
        return LLVMX86FP80TypeInContext(Ctx);
      case LLVMFP128TypeKind:
        return LLVMFP128TypeInContext(Ctx);
      case LLVMPPC_FP128TypeKind:
        return LLVMPPCFP128TypeInContext(Ctx);
      case LLVMLabelTypeKind:
        return LLVMLabelTypeInContext(Ctx);
      case LLVMIntegerTypeKind:
        return LLVMIntTypeInContext(Ctx, LLVMGetIntTypeWidth(Src));
      case LLVMFunctionTypeKind: {
        unsigned ParamCount = LLVMCountParamTypes(Src);
        LLVMTypeRef* Params = nullptr;
        if (ParamCount > 0) {
          Params = (LLVMTypeRef*) malloc(ParamCount * sizeof(LLVMTypeRef));
          LLVMGetParamTypes(Src, Params);
          for (unsigned i = 0; i < ParamCount; i++)
            Params[i] = Clone(Params[i]);
        }

        LLVMTypeRef FunTy = LLVMFunctionType(Clone(LLVMGetReturnType(Src)),
                                             Params, ParamCount,
                                             LLVMIsFunctionVarArg(Src));
        if (ParamCount > 0)
          free(Params);
        return FunTy;
      }
      case LLVMStructTypeKind: {
        LLVMTypeRef S = nullptr;
        const char *Name = LLVMGetStructName(Src);
        if (Name) {
          S = LLVMGetTypeByName(M, Name);
          if (S)
            return S;
          S = LLVMStructCreateNamed(Ctx, Name);
          if (LLVMIsOpaqueStruct(Src))
            return S;
        }

        unsigned EltCount = LLVMCountStructElementTypes(Src);
        SmallVector<LLVMTypeRef, 8> Elts;
        for (unsigned i = 0; i < EltCount; i++)
          Elts.push_back(Clone(LLVMStructGetTypeAtIndex(Src, i)));
        if (Name)
          LLVMStructSetBody(S, Elts.data(), EltCount, LLVMIsPackedStruct(Src));
        else
          S = LLVMStructTypeInContext(Ctx, Elts.data(), EltCount,
                                      LLVMIsPackedStruct(Src));
        return S;
      }
      case LLVMArrayTypeKind:
        return LLVMArrayType(
          Clone(LLVMGetElementType(Src)),
          LLVMGetArrayLength(Src)
        );
      case LLVMPointerTypeKind:
        return LLVMPointerType(
          Clone(LLVMGetElementType(Src)),
          LLVMGetPointerAddressSpace(Src)
        );
      case LLVMVectorTypeKind:
        return LLVMVectorType(
          Clone(LLVMGetElementType(Src)),
          LLVMGetVectorSize(Src)
        );
      case LLVMMetadataTypeKind:
        break;
      case LLVMX86_MMXTypeKind:
        return LLVMX86MMXTypeInContext(Ctx);
      default:
        break;
    }

    fprintf(stderr, "%d is not a supported typekind\n", Kind);
    exit(-1);
  }
};

static ValueMap clone_params(LLVMValueRef Src, LLVMValueRef Dst);

struct FunCloner {
  LLVMValueRef Fun;
  LLVMModuleRef M;

  ValueMap VMap;
  BasicBlockMap BBMap;

  FunCloner(LLVMValueRef Src, LLVMValueRef Dst): Fun(Dst),
    M(LLVMGetGlobalParent(Fun)), VMap(clone_params(Src, Dst)) {}

  LLVMTypeRef CloneType(LLVMTypeRef Src) {
    return TypeCloner(M).Clone(Src);
  }

  LLVMTypeRef CloneType(LLVMValueRef Src) {
    return CloneType(LLVMTypeOf(Src));
  }

  // Try to clone everything in the llvm::Value hierarchy.
  LLVMValueRef CloneValue(LLVMValueRef Src) {
    const char *Name = LLVMGetValueName(Src);

    // First, the value may be constant.
    if (LLVMIsAConstant(Src)) {
      // Maybe it is a symbol
      if (LLVMIsAGlobalValue(Src)) {
        // Try function
        LLVMValueRef Dst = LLVMGetNamedFunction(M, Name);
        if (Dst != nullptr)
          return Dst;

        // Try global variable
        Dst = LLVMGetNamedGlobal(M, Name);
        if (Dst != nullptr)
          return Dst;

        fprintf(stderr, "Could not find @%s\n", Name);
        exit(-1);
      }

      // Try literal
      if (LLVMIsAConstantInt(Src)) {
        LLVMTypeRef Ty = CloneType(Src);
        return LLVMConstInt(Ty, LLVMConstIntGetZExtValue(Src), false);
      }

      // Try undef
      if (LLVMIsUndef(Src))
        return LLVMGetUndef(CloneType(Src));

      // This kind of constant is not supported.
      report_fatal_error("Unsupported contant type");
    }

    // Function argument should always be in the map already.
    auto i = VMap.find(Src);
    if (i != VMap.end())
      return i->second;

    if (!LLVMIsAInstruction(Src))
      report_fatal_error("Expected an instruction");

    auto Ctx = LLVMGetModuleContext(M);
    auto Builder = LLVMCreateBuilderInContext(Ctx);
    auto BB = DeclareBB(LLVMGetInstructionParent(Src));
    LLVMPositionBuilderAtEnd(Builder, BB);
    auto Dst = CloneInstruction(Src, Builder);
    LLVMDisposeBuilder(Builder);
    return Dst;
  }

  LLVMValueRef CloneInstruction(LLVMValueRef Src, LLVMBuilderRef Builder) {
    const char *Name = LLVMGetValueName(Src);
    if (!LLVMIsAInstruction(Src))
      report_fatal_error("Expected an instruction");

    // Check if this is something we already computed.
    {
      auto i = VMap.find(Src);
      if (i != VMap.end()) {
        // If we have a hit, it means we already generated the instruction
        // as a dependancy to somethign else. We need to make sure
        // it is ordered properly.
        auto I = i->second;
        LLVMInstructionRemoveFromParent(I);
        LLVMInsertIntoBuilderWithName(Builder, I, Name);
        return I;
      }
    }

    // We tried everything, it must be an instruction
    // that hasn't been generated already.
    LLVMValueRef Dst = nullptr;

    LLVMOpcode Op = LLVMGetInstructionOpcode(Src);
    switch(Op) {
      case LLVMRet: {
        int OpCount = LLVMGetNumOperands(Src);
        if (OpCount == 0)
          Dst = LLVMBuildRetVoid(Builder);
        else
          Dst = LLVMBuildRet(Builder, CloneValue(LLVMGetOperand(Src, 0)));
        break;
      }
      case LLVMBr: {
        if (!LLVMIsConditional(Src)) {
          LLVMValueRef SrcOp = LLVMGetOperand(Src, 0);
          LLVMBasicBlockRef SrcBB = LLVMValueAsBasicBlock(SrcOp);
          Dst = LLVMBuildBr(Builder, DeclareBB(SrcBB));
          break;
        }

        LLVMValueRef Cond = LLVMGetCondition(Src);
        LLVMValueRef Else = LLVMGetOperand(Src, 1);
        LLVMBasicBlockRef ElseBB = DeclareBB(LLVMValueAsBasicBlock(Else));
        LLVMValueRef Then = LLVMGetOperand(Src, 2);
        LLVMBasicBlockRef ThenBB = DeclareBB(LLVMValueAsBasicBlock(Then));
        Dst = LLVMBuildCondBr(Builder, Cond, ThenBB, ElseBB);
        break;
      }
      case LLVMSwitch:
      case LLVMIndirectBr:
      case LLVMInvoke:
        break;
      case LLVMUnreachable:
        Dst = LLVMBuildUnreachable(Builder);
        break;
      case LLVMAdd: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildAdd(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMSub: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildSub(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMMul: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildMul(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMUDiv: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildUDiv(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMSDiv: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildSDiv(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMURem: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildURem(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMSRem: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildSRem(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMShl: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildShl(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMLShr: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildLShr(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMAShr: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildAShr(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMAnd: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildAnd(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMOr: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildOr(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMXor: {
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildXor(Builder, LHS, RHS, Name);
        break;
      }
      case LLVMAlloca: {
        LLVMTypeRef Ty = CloneType(LLVMGetAllocatedType(Src));
        Dst = LLVMBuildAlloca(Builder, Ty, Name);
        break;
      }
      case LLVMICmp: {
        LLVMIntPredicate Pred = LLVMGetICmpPredicate(Src);
        LLVMValueRef LHS = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef RHS = CloneValue(LLVMGetOperand(Src, 1));
        Dst = LLVMBuildICmp(Builder, Pred, LHS, RHS, Name);
        break;
      }
      case LLVMPHI: {
        // We need to agressively set things here because of loops.
        VMap[Src] = Dst = LLVMBuildPhi(Builder, CloneType(Src), Name);

        SmallVector<LLVMValueRef, 8> Values;
        SmallVector<LLVMBasicBlockRef, 8> Blocks;

        unsigned IncomingCount = LLVMCountIncoming(Src);
        for (unsigned i = 0; i < IncomingCount; ++i) {
          Blocks.push_back(DeclareBB(LLVMGetIncomingBlock(Src, i)));
          Values.push_back(CloneValue(LLVMGetIncomingValue(Src, i)));
        }

        LLVMAddIncoming(Dst, Values.data(), Blocks.data(), IncomingCount);
        return Dst;
      }
      case LLVMCall: {
        SmallVector<LLVMValueRef, 8> Args;
        int ArgCount = LLVMGetNumArgOperands(Src);
        for (int i = 0; i < ArgCount; i++)
          Args.push_back(CloneValue(LLVMGetOperand(Src, i)));
        LLVMValueRef Fn = CloneValue(LLVMGetCalledValue(Src));
        Dst = LLVMBuildCall(Builder, Fn, Args.data(), ArgCount, Name);
        break;
      }
      case LLVMExtractValue: {
        LLVMValueRef Agg = CloneValue(LLVMGetOperand(Src, 0));
        if (LLVMGetNumIndices(Src) != 1)
          report_fatal_error("Expected only one indice");
        auto I = LLVMGetIndices(Src)[0];
        Dst = LLVMBuildExtractValue(Builder, Agg, I, Name);
        break;
      }
      case LLVMInsertValue: {
        LLVMValueRef Agg = CloneValue(LLVMGetOperand(Src, 0));
        LLVMValueRef V = CloneValue(LLVMGetOperand(Src, 1));
        if (LLVMGetNumIndices(Src) != 1)
          report_fatal_error("Expected only one indice");
        auto I = LLVMGetIndices(Src)[0];
        Dst = LLVMBuildInsertValue(Builder, Agg, V, I, Name);
        break;
      }
      default:
        break;
    }

    if (Dst == nullptr) {
      fprintf(stderr, "%d is not a supported opcode\n", Op);
      exit(-1);
    }

    return VMap[Src] = Dst;
  }

  LLVMBasicBlockRef DeclareBB(LLVMBasicBlockRef Src) {
    // Check if this is something we already computed.
    {
      auto i = BBMap.find(Src);
      if (i != BBMap.end()) {
        return i->second;
      }
    }

    const char *Name = LLVMGetBasicBlockName(Src);

    LLVMValueRef V = LLVMBasicBlockAsValue(Src);
    if (!LLVMValueIsBasicBlock(V) || LLVMValueAsBasicBlock(V) != Src)
      report_fatal_error("Basic block is not a basic block");

    const char *VName = LLVMGetValueName(V);
    if (Name != VName)
      report_fatal_error("Basic block name mismatch");

    LLVMBasicBlockRef BB = LLVMAppendBasicBlock(Fun, Name);
    return BBMap[Src] = BB;
  }

  LLVMBasicBlockRef CloneBB(LLVMBasicBlockRef Src) {
    LLVMBasicBlockRef BB = DeclareBB(Src);

    // Make sure ordering is correct.
    LLVMBasicBlockRef Prev = LLVMGetPreviousBasicBlock(Src);
    if (Prev)
      LLVMMoveBasicBlockAfter(BB, DeclareBB(Prev));

    LLVMValueRef First = LLVMGetFirstInstruction(Src);
    LLVMValueRef Last = LLVMGetLastInstruction(Src);

    if (First == nullptr) {
      if (Last != nullptr) {
        fprintf(stderr, "Has no first instruction, but last one\n");
        exit(-1);
      }

      return BB;
    }

    auto Ctx = LLVMGetModuleContext(M);
    LLVMBuilderRef Builder = LLVMCreateBuilderInContext(Ctx);
    LLVMPositionBuilderAtEnd(Builder, BB);

    LLVMValueRef Cur = First;
    LLVMValueRef Next = nullptr;
    while(true) {
      CloneInstruction(Cur, Builder);
      Next = LLVMGetNextInstruction(Cur);
      if (Next == nullptr) {
        if (Cur != Last) {
          fprintf(stderr, "Final instruction does not match Last\n");
          exit(-1);
        }

        break;
      }

      LLVMValueRef Prev = LLVMGetPreviousInstruction(Next);
      if (Prev != Cur) {
        fprintf(stderr, "Next.Previous instruction is not Current\n");
        exit(-1);
      }

      Cur = Next;
    }

    LLVMDisposeBuilder(Builder);
    return BB;
  }

  void CloneBBs(LLVMValueRef Src) {
    unsigned Count = LLVMCountBasicBlocks(Src);
    if (Count == 0)
      return;

    LLVMBasicBlockRef First = LLVMGetFirstBasicBlock(Src);
    LLVMBasicBlockRef Last = LLVMGetLastBasicBlock(Src);

    LLVMBasicBlockRef Cur = First;
    LLVMBasicBlockRef Next = nullptr;
    while(true) {
      CloneBB(Cur);
      Count--;
      Next = LLVMGetNextBasicBlock(Cur);
      if (Next == nullptr) {
        if (Cur != Last) {
          fprintf(stderr, "Final basic block does not match Last\n");
          exit(-1);
        }

        break;
      }

      LLVMBasicBlockRef Prev = LLVMGetPreviousBasicBlock(Next);
      if (Prev != Cur) {
        fprintf(stderr, "Next.Previous basic bloc is not Current\n");
        exit(-1);
      }

      Cur = Next;
    }

    if (Count != 0) {
      fprintf(stderr, "Basic block count does not match iterration\n");
      exit(-1);
    }
  }
};

static ValueMap clone_params(LLVMValueRef Src, LLVMValueRef Dst) {
  unsigned Count = LLVMCountParams(Src);
  if (Count != LLVMCountParams(Dst)) {
    fprintf(stderr, "Parameter count mismatch\n");
    exit(-1);
  }

  ValueMap VMap;
  if (Count == 0)
    return VMap;

  LLVMValueRef SrcFirst = LLVMGetFirstParam(Src);
  LLVMValueRef DstFirst = LLVMGetFirstParam(Dst);
  LLVMValueRef SrcLast = LLVMGetLastParam(Src);
  LLVMValueRef DstLast = LLVMGetLastParam(Dst);

  LLVMValueRef SrcCur = SrcFirst;
  LLVMValueRef DstCur = DstFirst;
  LLVMValueRef SrcNext = nullptr;
  LLVMValueRef DstNext = nullptr;
  while (true) {
    const char *Name = LLVMGetValueName(SrcCur);
    LLVMSetValueName(DstCur, Name);

    VMap[SrcCur] = DstCur;

    Count--;
    SrcNext = LLVMGetNextParam(SrcCur);
    DstNext = LLVMGetNextParam(DstCur);
    if (SrcNext == nullptr && DstNext == nullptr) {
      if (SrcCur != SrcLast) {
        fprintf(stderr, "SrcLast param does not match End\n");
        exit(-1);
      }

      if (DstCur != DstLast) {
        fprintf(stderr, "DstLast param does not match End\n");
        exit(-1);
      }

      break;
    }

    if (SrcNext == nullptr) {
      fprintf(stderr, "SrcNext was unexpectedly null\n");
      exit(-1);
    }

    if (DstNext == nullptr) {
      fprintf(stderr, "DstNext was unexpectedly null\n");
      exit(-1);
    }

    LLVMValueRef SrcPrev = LLVMGetPreviousParam(SrcNext);
    if (SrcPrev != SrcCur) {
      fprintf(stderr, "SrcNext.Previous param is not Current\n");
      exit(-1);
    }

    LLVMValueRef DstPrev = LLVMGetPreviousParam(DstNext);
    if (DstPrev != DstCur) {
      fprintf(stderr, "DstNext.Previous param is not Current\n");
      exit(-1);
    }

    SrcCur = SrcNext;
    DstCur = DstNext;
  }

  if (Count != 0) {
    fprintf(stderr, "Parameter count does not match iteration\n");
    exit(-1);
  }

  return VMap;
}

static LLVMValueRef clone_function(LLVMValueRef Src, LLVMModuleRef M) {
  const char *Name = LLVMGetValueName(Src);
  LLVMValueRef Fun = LLVMGetNamedFunction(M, Name);
  if (Fun != nullptr)
    return Fun;

  LLVMTypeRef DstTy = TypeCloner(M).Clone(LLVMTypeOf(Src));
  LLVMTypeRef FunTy = LLVMGetElementType(DstTy);

  Fun = LLVMAddFunction(M, Name, FunTy);
  FunCloner FC(Src, Fun);
  FC.CloneBBs(Src);

  return Fun;
}

static void clone_functions(LLVMModuleRef Src, LLVMModuleRef Dst) {
  LLVMValueRef Begin = LLVMGetFirstFunction(Src);
  LLVMValueRef End = LLVMGetLastFunction(Src);

  LLVMValueRef Cur = Begin;
  LLVMValueRef Next = nullptr;
  while (true) {
    clone_function(Cur, Dst);
    Next = LLVMGetNextFunction(Cur);
    if (Next == nullptr) {
      if (Cur != End) {
        fprintf(stderr, "Last function does not match End\n");
        exit(-1);
      }

      break;
    }

    LLVMValueRef Prev = LLVMGetPreviousFunction(Next);
    if (Prev != Cur) {
      fprintf(stderr, "Next.Previous function is not Current\n");
      exit(-1);
    }

    Cur = Next;
  }
}

int llvm_echo(void) {
  LLVMEnablePrettyStackTrace();

  LLVMModuleRef Src = llvm_load_module(false, true);

  LLVMContextRef Ctx = LLVMContextCreate();
  LLVMModuleRef Dst = LLVMModuleCreateWithNameInContext("<stdin>", Ctx);

  clone_functions(Src, Dst);
  char *Str = LLVMPrintModuleToString(Dst);
  fputs(Str, stdout);

  LLVMDisposeMessage(Str);
  LLVMDisposeModule(Dst);
  LLVMContextDispose(Ctx);

  return 0;
}
