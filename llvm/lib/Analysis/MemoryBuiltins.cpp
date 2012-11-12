//===------ MemoryBuiltins.cpp - Identify calls to memory builtins --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.  
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "memory-builtins"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Metadata.h"
#include "llvm/Module.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

enum AllocType {
  MallocLike         = 1<<0, // allocates
  CallocLike         = 1<<1, // allocates + bzero
  ReallocLike        = 1<<2, // reallocates
  StrDupLike         = 1<<3,
  AllocLike          = MallocLike | CallocLike | StrDupLike,
  AnyAlloc           = MallocLike | CallocLike | ReallocLike | StrDupLike
};

struct AllocFnsTy {
  LibFunc::Func Func;
  AllocType AllocTy;
  unsigned char NumParams;
  // First and Second size parameters (or -1 if unused)
  signed char FstParam, SndParam;
};

// FIXME: certain users need more information. E.g., SimplifyLibCalls needs to
// know which functions are nounwind, noalias, nocapture parameters, etc.
static const AllocFnsTy AllocationFnData[] = {
  {LibFunc::malloc,              MallocLike,  1, 0,  -1},
  {LibFunc::valloc,              MallocLike,  1, 0,  -1},
  {LibFunc::Znwj,                MallocLike,  1, 0,  -1}, // new(unsigned int)
  {LibFunc::ZnwjRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new(unsigned int, nothrow)
  {LibFunc::Znwm,                MallocLike,  1, 0,  -1}, // new(unsigned long)
  {LibFunc::ZnwmRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new(unsigned long, nothrow)
  {LibFunc::Znaj,                MallocLike,  1, 0,  -1}, // new[](unsigned int)
  {LibFunc::ZnajRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new[](unsigned int, nothrow)
  {LibFunc::Znam,                MallocLike,  1, 0,  -1}, // new[](unsigned long)
  {LibFunc::ZnamRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new[](unsigned long, nothrow)
  {LibFunc::posix_memalign,      MallocLike,  3, 2,  -1},
  {LibFunc::calloc,              CallocLike,  2, 0,   1},
  {LibFunc::realloc,             ReallocLike, 2, 1,  -1},
  {LibFunc::reallocf,            ReallocLike, 2, 1,  -1},
  {LibFunc::strdup,              StrDupLike,  1, -1, -1},
  {LibFunc::strndup,             StrDupLike,  2, 1,  -1}
};


static Function *getCalledFunction(const Value *V, bool LookThroughBitCast) {
  if (LookThroughBitCast)
    V = V->stripPointerCasts();

  CallSite CS(const_cast<Value*>(V));
  if (!CS.getInstruction())
    return 0;

  Function *Callee = CS.getCalledFunction();
  if (!Callee || !Callee->isDeclaration())
    return 0;
  return Callee;
}

/// \brief Returns the allocation data for the given value if it is a call to a
/// known allocation function, and NULL otherwise.
static const AllocFnsTy *getAllocationData(const Value *V, AllocType AllocTy,
                                           const TargetLibraryInfo *TLI,
                                           bool LookThroughBitCast = false) {
  Function *Callee = getCalledFunction(V, LookThroughBitCast);
  if (!Callee)
    return 0;

  // Make sure that the function is available.
  StringRef FnName = Callee->getName();
  LibFunc::Func TLIFn;
  if (!TLI || !TLI->getLibFunc(FnName, TLIFn) || !TLI->has(TLIFn))
    return 0;

  unsigned i = 0;
  bool found = false;
  for ( ; i < array_lengthof(AllocationFnData); ++i) {
    if (AllocationFnData[i].Func == TLIFn) {
      found = true;
      break;
    }
  }
  if (!found)
    return 0;

  const AllocFnsTy *FnData = &AllocationFnData[i];
  if ((FnData->AllocTy & AllocTy) == 0)
    return 0;

  // Check function prototype.
  int FstParam = FnData->FstParam;
  int SndParam = FnData->SndParam;
  FunctionType *FTy = Callee->getFunctionType();

  if (FTy->getReturnType() == Type::getInt8PtrTy(FTy->getContext()) &&
      FTy->getNumParams() == FnData->NumParams &&
      (FstParam < 0 ||
       (FTy->getParamType(FstParam)->isIntegerTy(32) ||
        FTy->getParamType(FstParam)->isIntegerTy(64))) &&
      (SndParam < 0 ||
       FTy->getParamType(SndParam)->isIntegerTy(32) ||
       FTy->getParamType(SndParam)->isIntegerTy(64)))
    return FnData;
  return 0;
}

static bool hasNoAliasAttr(const Value *V, bool LookThroughBitCast) {
  ImmutableCallSite CS(LookThroughBitCast ? V->stripPointerCasts() : V);
  return CS && CS.hasFnAttr(Attributes::NoAlias);
}


/// \brief Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool llvm::isAllocationFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast) {
  return getAllocationData(V, AnyAlloc, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a function that returns a
/// NoAlias pointer (including malloc/calloc/realloc/strdup-like functions).
bool llvm::isNoAliasFn(const Value *V, const TargetLibraryInfo *TLI,
                       bool LookThroughBitCast) {
  // it's safe to consider realloc as noalias since accessing the original
  // pointer is undefined behavior
  return isAllocationFn(V, TLI, LookThroughBitCast) ||
         hasNoAliasAttr(V, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory (such as malloc).
bool llvm::isMallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast) {
  return getAllocationData(V, MallocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates zero-filled memory (such as calloc).
bool llvm::isCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast) {
  return getAllocationData(V, CallocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool llvm::isAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                         bool LookThroughBitCast) {
  return getAllocationData(V, AllocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// reallocates memory (such as realloc).
bool llvm::isReallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                           bool LookThroughBitCast) {
  return getAllocationData(V, ReallocLike, TLI, LookThroughBitCast);
}

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst *llvm::extractMallocCall(const Value *I,
                                        const TargetLibraryInfo *TLI) {
  return isMallocLikeFn(I, TLI) ? dyn_cast<CallInst>(I) : 0;
}

static Value *computeArraySize(const CallInst *CI, const DataLayout *TD,
                               const TargetLibraryInfo *TLI,
                               bool LookThroughSExt = false) {
  if (!CI)
    return NULL;

  // The size of the malloc's result type must be known to determine array size.
  Type *T = getMallocAllocatedType(CI, TLI);
  if (!T || !T->isSized() || !TD)
    return NULL;

  unsigned ElementSize = TD->getTypeAllocSize(T);
  if (StructType *ST = dyn_cast<StructType>(T))
    ElementSize = TD->getStructLayout(ST)->getSizeInBytes();

  // If malloc call's arg can be determined to be a multiple of ElementSize,
  // return the multiple.  Otherwise, return NULL.
  Value *MallocArg = CI->getArgOperand(0);
  Value *Multiple = NULL;
  if (ComputeMultiple(MallocArg, ElementSize, Multiple,
                      LookThroughSExt))
    return Multiple;

  return NULL;
}

/// isArrayMalloc - Returns the corresponding CallInst if the instruction 
/// is a call to malloc whose array size can be determined and the array size
/// is not constant 1.  Otherwise, return NULL.
const CallInst *llvm::isArrayMalloc(const Value *I,
                                    const DataLayout *TD,
                                    const TargetLibraryInfo *TLI) {
  const CallInst *CI = extractMallocCall(I, TLI);
  Value *ArraySize = computeArraySize(CI, TD, TLI);

  if (ArraySize &&
      ArraySize != ConstantInt::get(CI->getArgOperand(0)->getType(), 1))
    return CI;

  // CI is a non-array malloc or we can't figure out that it is an array malloc.
  return NULL;
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
PointerType *llvm::getMallocType(const CallInst *CI,
                                 const TargetLibraryInfo *TLI) {
  assert(isMallocLikeFn(CI, TLI) && "getMallocType and not malloc call");
  
  PointerType *MallocType = NULL;
  unsigned NumOfBitCastUses = 0;

  // Determine if CallInst has a bitcast use.
  for (Value::const_use_iterator UI = CI->use_begin(), E = CI->use_end();
       UI != E; )
    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(*UI++)) {
      MallocType = cast<PointerType>(BCI->getDestTy());
      NumOfBitCastUses++;
    }

  // Malloc call has 1 bitcast use, so type is the bitcast's destination type.
  if (NumOfBitCastUses == 1)
    return MallocType;

  // Malloc call was not bitcast, so type is the malloc function's return type.
  if (NumOfBitCastUses == 0)
    return cast<PointerType>(CI->getType());

  // Type could not be determined.
  return NULL;
}

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
Type *llvm::getMallocAllocatedType(const CallInst *CI,
                                   const TargetLibraryInfo *TLI) {
  PointerType *PT = getMallocType(CI, TLI);
  return PT ? PT->getElementType() : NULL;
}

/// getMallocArraySize - Returns the array size of a malloc call.  If the 
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value *llvm::getMallocArraySize(CallInst *CI, const DataLayout *TD,
                                const TargetLibraryInfo *TLI,
                                bool LookThroughSExt) {
  assert(isMallocLikeFn(CI, TLI) && "getMallocArraySize and not malloc call");
  return computeArraySize(CI, TD, TLI, LookThroughSExt);
}


/// extractCallocCall - Returns the corresponding CallInst if the instruction
/// is a calloc call.
const CallInst *llvm::extractCallocCall(const Value *I,
                                        const TargetLibraryInfo *TLI) {
  return isCallocLikeFn(I, TLI) ? cast<CallInst>(I) : 0;
}


/// isFreeCall - Returns non-null if the value is a call to the builtin free()
const CallInst *llvm::isFreeCall(const Value *I, const TargetLibraryInfo *TLI) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI)
    return 0;
  Function *Callee = CI->getCalledFunction();
  if (Callee == 0 || !Callee->isDeclaration())
    return 0;

  StringRef FnName = Callee->getName();
  LibFunc::Func TLIFn;
  if (!TLI || !TLI->getLibFunc(FnName, TLIFn) || !TLI->has(TLIFn))
    return 0;

  if (TLIFn != LibFunc::free &&
      TLIFn != LibFunc::ZdlPv && // operator delete(void*)
      TLIFn != LibFunc::ZdaPv)   // operator delete[](void*)
    return 0;

  // Check free prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin 
  // attribute will exist.
  FunctionType *FTy = Callee->getFunctionType();
  if (!FTy->getReturnType()->isVoidTy())
    return 0;
  if (FTy->getNumParams() != 1)
    return 0;
  if (FTy->getParamType(0) != Type::getInt8PtrTy(Callee->getContext()))
    return 0;

  return CI;
}



//===----------------------------------------------------------------------===//
//  Utility functions to compute size of objects.
//


/// \brief Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise.
/// If RoundToAlign is true, then Size is rounded up to the aligment of allocas,
/// byval arguments, and global variables.
bool llvm::getObjectSize(const Value *Ptr, uint64_t &Size, const DataLayout *TD,
                         const TargetLibraryInfo *TLI, bool RoundToAlign) {
  if (!TD)
    return false;

  ObjectSizeOffsetVisitor Visitor(TD, TLI, Ptr->getContext(), RoundToAlign);
  SizeOffsetType Data = Visitor.compute(const_cast<Value*>(Ptr));
  if (!Visitor.bothKnown(Data))
    return false;

  APInt ObjSize = Data.first, Offset = Data.second;
  // check for overflow
  if (Offset.slt(0) || ObjSize.ult(Offset))
    Size = 0;
  else
    Size = (ObjSize - Offset).getZExtValue();
  return true;
}


STATISTIC(ObjectVisitorArgument,
          "Number of arguments with unsolved size and offset");
STATISTIC(ObjectVisitorLoad,
          "Number of load instructions with unsolved size and offset");


APInt ObjectSizeOffsetVisitor::align(APInt Size, uint64_t Align) {
  if (RoundToAlign && Align)
    return APInt(IntTyBits, RoundUpToAlignment(Size.getZExtValue(), Align));
  return Size;
}

ObjectSizeOffsetVisitor::ObjectSizeOffsetVisitor(const DataLayout *TD,
                                                 const TargetLibraryInfo *TLI,
                                                 LLVMContext &Context,
                                                 bool RoundToAlign)
: TD(TD), TLI(TLI), RoundToAlign(RoundToAlign) {
  IntegerType *IntTy = TD->getIntPtrType(Context);
  IntTyBits = IntTy->getBitWidth();
  Zero = APInt::getNullValue(IntTyBits);
}

SizeOffsetType ObjectSizeOffsetVisitor::compute(Value *V) {
  V = V->stripPointerCasts();
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If we have already seen this instruction, bail out. Cycles can happen in
    // unreachable code after constant propagation.
    if (!SeenInsts.insert(I))
      return unknown();

    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V))
      return visitGEPOperator(*GEP);
    return visit(*I);
  }
  if (Argument *A = dyn_cast<Argument>(V))
    return visitArgument(*A);
  if (ConstantPointerNull *P = dyn_cast<ConstantPointerNull>(V))
    return visitConstantPointerNull(*P);
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return visitGlobalVariable(*GV);
  if (UndefValue *UV = dyn_cast<UndefValue>(V))
    return visitUndefValue(*UV);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::IntToPtr)
      return unknown(); // clueless
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return visitGEPOperator(cast<GEPOperator>(*CE));
  }

  DEBUG(dbgs() << "ObjectSizeOffsetVisitor::compute() unhandled value: " << *V
        << '\n');
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitAllocaInst(AllocaInst &I) {
  if (!I.getAllocatedType()->isSized())
    return unknown();

  APInt Size(IntTyBits, TD->getTypeAllocSize(I.getAllocatedType()));
  if (!I.isArrayAllocation())
    return std::make_pair(align(Size, I.getAlignment()), Zero);

  Value *ArraySize = I.getArraySize();
  if (const ConstantInt *C = dyn_cast<ConstantInt>(ArraySize)) {
    Size *= C->getValue().zextOrSelf(IntTyBits);
    return std::make_pair(align(Size, I.getAlignment()), Zero);
  }
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitArgument(Argument &A) {
  // no interprocedural analysis is done at the moment
  if (!A.hasByValAttr()) {
    ++ObjectVisitorArgument;
    return unknown();
  }
  PointerType *PT = cast<PointerType>(A.getType());
  APInt Size(IntTyBits, TD->getTypeAllocSize(PT->getElementType()));
  return std::make_pair(align(Size, A.getParamAlignment()), Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitCallSite(CallSite CS) {
  const AllocFnsTy *FnData = getAllocationData(CS.getInstruction(), AnyAlloc,
                                               TLI);
  if (!FnData)
    return unknown();

  // handle strdup-like functions separately
  if (FnData->AllocTy == StrDupLike) {
    APInt Size(IntTyBits, GetStringLength(CS.getArgument(0)));
    if (!Size)
      return unknown();

    // strndup limits strlen
    if (FnData->FstParam > 0) {
      ConstantInt *Arg= dyn_cast<ConstantInt>(CS.getArgument(FnData->FstParam));
      if (!Arg)
        return unknown();

      APInt MaxSize = Arg->getValue().zextOrSelf(IntTyBits);
      if (Size.ugt(MaxSize))
        Size = MaxSize + 1;
    }
    return std::make_pair(Size, Zero);
  }

  ConstantInt *Arg = dyn_cast<ConstantInt>(CS.getArgument(FnData->FstParam));
  if (!Arg)
    return unknown();

  APInt Size = Arg->getValue().zextOrSelf(IntTyBits);
  // size determined by just 1 parameter
  if (FnData->SndParam < 0)
    return std::make_pair(Size, Zero);

  Arg = dyn_cast<ConstantInt>(CS.getArgument(FnData->SndParam));
  if (!Arg)
    return unknown();

  Size *= Arg->getValue().zextOrSelf(IntTyBits);
  return std::make_pair(Size, Zero);

  // TODO: handle more standard functions (+ wchar cousins):
  // - strdup / strndup
  // - strcpy / strncpy
  // - strcat / strncat
  // - memcpy / memmove
  // - strcat / strncat
  // - memset
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitConstantPointerNull(ConstantPointerNull&) {
  return std::make_pair(Zero, Zero);
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitExtractElementInst(ExtractElementInst&) {
  return unknown();
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitExtractValueInst(ExtractValueInst&) {
  // Easy cases were already folded by previous passes.
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGEPOperator(GEPOperator &GEP) {
  SizeOffsetType PtrData = compute(GEP.getPointerOperand());
  if (!bothKnown(PtrData) || !GEP.hasAllConstantIndices())
    return unknown();

  SmallVector<Value*, 8> Ops(GEP.idx_begin(), GEP.idx_end());
  APInt Offset(IntTyBits,TD->getIndexedOffset(GEP.getPointerOperandType(),Ops));
  return std::make_pair(PtrData.first, PtrData.second + Offset);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGlobalVariable(GlobalVariable &GV){
  if (!GV.hasDefinitiveInitializer())
    return unknown();

  APInt Size(IntTyBits, TD->getTypeAllocSize(GV.getType()->getElementType()));
  return std::make_pair(align(Size, GV.getAlignment()), Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitIntToPtrInst(IntToPtrInst&) {
  // clueless
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitLoadInst(LoadInst&) {
  ++ObjectVisitorLoad;
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitPHINode(PHINode&) {
  // too complex to analyze statically.
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitSelectInst(SelectInst &I) {
  SizeOffsetType TrueSide  = compute(I.getTrueValue());
  SizeOffsetType FalseSide = compute(I.getFalseValue());
  if (bothKnown(TrueSide) && bothKnown(FalseSide) && TrueSide == FalseSide)
    return TrueSide;
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitUndefValue(UndefValue&) {
  return std::make_pair(Zero, Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitInstruction(Instruction &I) {
  DEBUG(dbgs() << "ObjectSizeOffsetVisitor unknown instruction:" << I << '\n');
  return unknown();
}


ObjectSizeOffsetEvaluator::ObjectSizeOffsetEvaluator(const DataLayout *TD,
                                                   const TargetLibraryInfo *TLI,
                                                     LLVMContext &Context)
: TD(TD), TLI(TLI), Context(Context), Builder(Context, TargetFolder(TD)) {
  IntTy = TD->getIntPtrType(Context);
  Zero = ConstantInt::get(IntTy, 0);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::compute(Value *V) {
  SizeOffsetEvalType Result = compute_(V);

  if (!bothKnown(Result)) {
    // erase everything that was computed in this iteration from the cache, so
    // that no dangling references are left behind. We could be a bit smarter if
    // we kept a dependency graph. It's probably not worth the complexity.
    for (PtrSetTy::iterator I=SeenVals.begin(), E=SeenVals.end(); I != E; ++I) {
      CacheMapTy::iterator CacheIt = CacheMap.find(*I);
      // non-computable results can be safely cached
      if (CacheIt != CacheMap.end() && anyKnown(CacheIt->second))
        CacheMap.erase(CacheIt);
    }
  }

  SeenVals.clear();
  return Result;
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::compute_(Value *V) {
  ObjectSizeOffsetVisitor Visitor(TD, TLI, Context);
  SizeOffsetType Const = Visitor.compute(V);
  if (Visitor.bothKnown(Const))
    return std::make_pair(ConstantInt::get(Context, Const.first),
                          ConstantInt::get(Context, Const.second));

  V = V->stripPointerCasts();

  // check cache
  CacheMapTy::iterator CacheIt = CacheMap.find(V);
  if (CacheIt != CacheMap.end())
    return CacheIt->second;

  // always generate code immediately before the instruction being
  // processed, so that the generated code dominates the same BBs
  Instruction *PrevInsertPoint = Builder.GetInsertPoint();
  if (Instruction *I = dyn_cast<Instruction>(V))
    Builder.SetInsertPoint(I);

  // record the pointers that were handled in this run, so that they can be
  // cleaned later if something fails
  SeenVals.insert(V);

  // now compute the size and offset
  SizeOffsetEvalType Result;
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    Result = visitGEPOperator(*GEP);
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    Result = visit(*I);
  } else if (isa<Argument>(V) ||
             (isa<ConstantExpr>(V) &&
              cast<ConstantExpr>(V)->getOpcode() == Instruction::IntToPtr) ||
             isa<GlobalVariable>(V)) {
    // ignore values where we cannot do more than what ObjectSizeVisitor can
    Result = unknown();
  } else {
    DEBUG(dbgs() << "ObjectSizeOffsetEvaluator::compute() unhandled value: "
          << *V << '\n');
    Result = unknown();
  }

  if (PrevInsertPoint)
    Builder.SetInsertPoint(PrevInsertPoint);

  // Don't reuse CacheIt since it may be invalid at this point.
  CacheMap[V] = Result;
  return Result;
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitAllocaInst(AllocaInst &I) {
  if (!I.getAllocatedType()->isSized())
    return unknown();

  // must be a VLA
  assert(I.isArrayAllocation());
  Value *ArraySize = I.getArraySize();
  Value *Size = ConstantInt::get(ArraySize->getType(),
                                 TD->getTypeAllocSize(I.getAllocatedType()));
  Size = Builder.CreateMul(Size, ArraySize);
  return std::make_pair(Size, Zero);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitCallSite(CallSite CS) {
  const AllocFnsTy *FnData = getAllocationData(CS.getInstruction(), AnyAlloc,
                                               TLI);
  if (!FnData)
    return unknown();

  // handle strdup-like functions separately
  if (FnData->AllocTy == StrDupLike) {
    // TODO
    return unknown();
  }

  Value *FirstArg = CS.getArgument(FnData->FstParam);
  FirstArg = Builder.CreateZExt(FirstArg, IntTy);
  if (FnData->SndParam < 0)
    return std::make_pair(FirstArg, Zero);

  Value *SecondArg = CS.getArgument(FnData->SndParam);
  SecondArg = Builder.CreateZExt(SecondArg, IntTy);
  Value *Size = Builder.CreateMul(FirstArg, SecondArg);
  return std::make_pair(Size, Zero);

  // TODO: handle more standard functions (+ wchar cousins):
  // - strdup / strndup
  // - strcpy / strncpy
  // - strcat / strncat
  // - memcpy / memmove
  // - strcat / strncat
  // - memset
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitExtractElementInst(ExtractElementInst&) {
  return unknown();
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitExtractValueInst(ExtractValueInst&) {
  return unknown();
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitGEPOperator(GEPOperator &GEP) {
  SizeOffsetEvalType PtrData = compute_(GEP.getPointerOperand());
  if (!bothKnown(PtrData))
    return unknown();

  Value *Offset = EmitGEPOffset(&Builder, *TD, &GEP, /*NoAssumptions=*/true);
  Offset = Builder.CreateAdd(PtrData.second, Offset);
  return std::make_pair(PtrData.first, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitIntToPtrInst(IntToPtrInst&) {
  // clueless
  return unknown();
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitLoadInst(LoadInst&) {
  return unknown();
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitPHINode(PHINode &PHI) {
  // create 2 PHIs: one for size and another for offset
  PHINode *SizePHI   = Builder.CreatePHI(IntTy, PHI.getNumIncomingValues());
  PHINode *OffsetPHI = Builder.CreatePHI(IntTy, PHI.getNumIncomingValues());

  // insert right away in the cache to handle recursive PHIs
  CacheMap[&PHI] = std::make_pair(SizePHI, OffsetPHI);

  // compute offset/size for each PHI incoming pointer
  for (unsigned i = 0, e = PHI.getNumIncomingValues(); i != e; ++i) {
    Builder.SetInsertPoint(PHI.getIncomingBlock(i)->getFirstInsertionPt());
    SizeOffsetEvalType EdgeData = compute_(PHI.getIncomingValue(i));

    if (!bothKnown(EdgeData)) {
      OffsetPHI->replaceAllUsesWith(UndefValue::get(IntTy));
      OffsetPHI->eraseFromParent();
      SizePHI->replaceAllUsesWith(UndefValue::get(IntTy));
      SizePHI->eraseFromParent();
      return unknown();
    }
    SizePHI->addIncoming(EdgeData.first, PHI.getIncomingBlock(i));
    OffsetPHI->addIncoming(EdgeData.second, PHI.getIncomingBlock(i));
  }

  Value *Size = SizePHI, *Offset = OffsetPHI, *Tmp;
  if ((Tmp = SizePHI->hasConstantValue())) {
    Size = Tmp;
    SizePHI->replaceAllUsesWith(Size);
    SizePHI->eraseFromParent();
  }
  if ((Tmp = OffsetPHI->hasConstantValue())) {
    Offset = Tmp;
    OffsetPHI->replaceAllUsesWith(Offset);
    OffsetPHI->eraseFromParent();
  }
  return std::make_pair(Size, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitSelectInst(SelectInst &I) {
  SizeOffsetEvalType TrueSide  = compute_(I.getTrueValue());
  SizeOffsetEvalType FalseSide = compute_(I.getFalseValue());

  if (!bothKnown(TrueSide) || !bothKnown(FalseSide))
    return unknown();
  if (TrueSide == FalseSide)
    return TrueSide;

  Value *Size = Builder.CreateSelect(I.getCondition(), TrueSide.first,
                                     FalseSide.first);
  Value *Offset = Builder.CreateSelect(I.getCondition(), TrueSide.second,
                                       FalseSide.second);
  return std::make_pair(Size, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitInstruction(Instruction &I) {
  DEBUG(dbgs() << "ObjectSizeOffsetEvaluator unknown instruction:" << I <<'\n');
  return unknown();
}
