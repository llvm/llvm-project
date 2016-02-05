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

static LLVMTypeRef clone_type(LLVMTypeRef Src, LLVMContextRef Ctx) {
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
        for (unsigned i = 0; i < ParamCount; i++) {
          Params[i] = clone_type(Params[i], Ctx);
        }
      }

      LLVMTypeRef FunTy = LLVMFunctionType(
        clone_type(LLVMGetReturnType(Src), Ctx),
        Params, ParamCount,
        LLVMIsFunctionVarArg(Src)
      );

      if (ParamCount > 0)
        free(Params);

      return FunTy;
    }
    case LLVMStructTypeKind:
      break;
    case LLVMArrayTypeKind:
      return LLVMArrayType(
        clone_type(LLVMGetElementType(Src), Ctx),
        LLVMGetArrayLength(Src)
      );
    case LLVMPointerTypeKind:
      return LLVMPointerType(
        clone_type(LLVMGetElementType(Src), Ctx),
        LLVMGetPointerAddressSpace(Src)
      );
    case LLVMVectorTypeKind:
      return LLVMVectorType(
        clone_type(LLVMGetElementType(Src), Ctx),
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

static LLVMValueRef clone_literal(LLVMValueRef Src, LLVMContextRef Ctx) {
  LLVMTypeRef Ty = clone_type(LLVMTypeOf(Src), Ctx);

  LLVMTypeKind Kind = LLVMGetTypeKind(Ty);
  switch (Kind) {
    case LLVMIntegerTypeKind:
      return LLVMConstInt(Ty, LLVMConstIntGetZExtValue(Src), false);
    default:
      break;
  }

  fprintf(stderr, "%d is not a supported constant typekind\n", Kind);
  exit(-1);
}

static LLVMModuleRef get_module(LLVMBuilderRef Builder) {
  LLVMBasicBlockRef BB = LLVMGetInsertBlock(Builder);
  LLVMValueRef Fn = LLVMGetBasicBlockParent(BB);
  return LLVMGetGlobalParent(Fn);
}

// Try to clone everything in the llvm::Value hierarchy.
static LLVMValueRef clone_value(LLVMValueRef Src, LLVMBuilderRef Builder, ValueMap &VMap) {
  const char *Name = LLVMGetValueName(Src);

  // First, the value may be constant.
  if (LLVMIsAConstant(Src)) {
    LLVMModuleRef M = get_module(Builder);

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
    LLVMContextRef Ctx = LLVMGetModuleContext(M);
    return clone_literal(Src, Ctx);
  }

  // Try undef
  if (LLVMIsUndef(Src)) {
    LLVMContextRef Ctx = LLVMGetModuleContext(get_module(Builder));
    LLVMTypeRef Ty = clone_type(LLVMTypeOf(Src), Ctx);
    return LLVMGetUndef(Ty);
  }

  // Check if this is something we already computed.
  {
    auto i = VMap.find(Src);
    if (i != VMap.end())
      return i->second;
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
        Dst = LLVMBuildRet(Builder, clone_value(LLVMGetOperand(Src, 0),
                                                Builder, VMap));
      break;
    }
    case LLVMBr:
    case LLVMSwitch:
    case LLVMIndirectBr:
    case LLVMInvoke:
    case LLVMUnreachable:
      break;
    case LLVMAdd: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildAdd(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMSub: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildSub(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMMul: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildMul(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMUDiv: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildUDiv(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMSDiv: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildSDiv(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMURem: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildURem(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMSRem: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildSRem(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMShl: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildShl(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMLShr: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildLShr(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMAShr: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildAShr(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMAnd: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildAnd(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMOr: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildOr(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMXor: {
      LLVMValueRef LHS = clone_value(LLVMGetOperand(Src, 0), Builder, VMap);
      LLVMValueRef RHS = clone_value(LLVMGetOperand(Src, 1), Builder, VMap);
      Dst = LLVMBuildXor(Builder, LHS, RHS, Name);
      break;
    }
    case LLVMAlloca: {
      LLVMTypeRef Ty = LLVMGetElementType(LLVMTypeOf(Src));
      Dst = LLVMBuildAlloca(Builder, Ty, Name);
      break;
    }
    case LLVMCall: {
      int ArgCount = LLVMGetNumOperands(Src) - 1;
      SmallVector<LLVMValueRef, 8> Args;
      for (int i = 0; i < ArgCount; i++)
        Args.push_back(clone_value(LLVMGetOperand(Src, i), Builder, VMap));
      LLVMValueRef Fn = clone_value(LLVMGetOperand(Src, ArgCount), Builder, VMap);
      Dst = LLVMBuildCall(Builder, Fn, Args.data(), ArgCount, Name);
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

static LLVMBasicBlockRef clone_bb(LLVMBasicBlockRef Src, LLVMValueRef Dst, ValueMap &VMap) {
  LLVMBasicBlockRef BB = LLVMAppendBasicBlock(Dst, "");

  LLVMValueRef First = LLVMGetFirstInstruction(Src);
  LLVMValueRef Last = LLVMGetLastInstruction(Src);

  if (First == nullptr) {
    if (Last != nullptr) {
      fprintf(stderr, "Has no first instruction, but last one\n");
      exit(-1);
    }

    return BB;
  }

  LLVMContextRef Ctx = LLVMGetModuleContext(LLVMGetGlobalParent(Dst));
  LLVMBuilderRef Builder = LLVMCreateBuilderInContext(Ctx);
  LLVMPositionBuilderAtEnd(Builder, BB);

  LLVMValueRef Cur = First;
  LLVMValueRef Next = nullptr;
  while(true) {
    clone_value(Cur, Builder, VMap);
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

static void clone_bbs(LLVMValueRef Src, LLVMValueRef Dst, ValueMap &VMap) {
  unsigned Count = LLVMCountBasicBlocks(Src);
  if (Count == 0)
    return;

  LLVMBasicBlockRef First = LLVMGetFirstBasicBlock(Src);
  LLVMBasicBlockRef Last = LLVMGetLastBasicBlock(Src);

  LLVMBasicBlockRef Cur = First;
  LLVMBasicBlockRef Next = nullptr;
  while(true) {
    clone_bb(Cur, Dst, VMap);
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

static LLVMValueRef clone_function(LLVMValueRef Src, LLVMModuleRef Dst) {
  const char *Name = LLVMGetValueName(Src);
  LLVMValueRef Fun = LLVMGetNamedFunction(Dst, Name);
  if (Fun != nullptr)
    return Fun;

  LLVMTypeRef SrcTy = LLVMTypeOf(Src);
  LLVMTypeRef DstTy = clone_type(SrcTy, LLVMGetModuleContext(Dst));
  LLVMTypeRef FunTy = LLVMGetElementType(DstTy);

  Fun = LLVMAddFunction(Dst, Name, FunTy);

  ValueMap VMap = clone_params(Src, Fun);
  clone_bbs(Src, Fun, VMap);

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

int echo(void) {
  LLVMEnablePrettyStackTrace();

  LLVMModuleRef Src = load_module(false, true);

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
