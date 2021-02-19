//===- BasicAliasAnalysis.cpp - Stateless Alias Analysis Impl -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the primary stateless implementation of the
// Alias Analysis interface that implements identities (two different
// globals cannot alias, etc), but does no stateful analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/PhiValues.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/KnownBits.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <utility>

#define DEBUG_TYPE "basicaa"

using namespace llvm;

/// Enable analysis of recursive PHI nodes.
static cl::opt<bool> EnableRecPhiAnalysis("basic-aa-recphi", cl::Hidden,
                                          cl::init(true));

/// By default, even on 32-bit architectures we use 64-bit integers for
/// calculations. This will allow us to more-aggressively decompose indexing
/// expressions calculated using i64 values (e.g., long long in C) which is
/// common enough to worry about.
static cl::opt<bool> ForceAtLeast64Bits("basic-aa-force-at-least-64b",
                                        cl::Hidden, cl::init(true));
static cl::opt<bool> DoubleCalcBits("basic-aa-double-calc-bits",
                                    cl::Hidden, cl::init(false));

/// SearchLimitReached / SearchTimes shows how often the limit of
/// to decompose GEPs is reached. It will affect the precision
/// of basic alias analysis.
STATISTIC(SearchLimitReached, "Number of times the limit to "
                              "decompose GEPs is reached");
STATISTIC(SearchTimes, "Number of times a GEP is decomposed");

/// Cutoff after which to stop analysing a set of phi nodes potentially involved
/// in a cycle. Because we are analysing 'through' phi nodes, we need to be
/// careful with value equivalence. We use reachability to make sure a value
/// cannot be involved in a cycle.
const unsigned MaxNumPhiBBsValueReachabilityCheck = 20;

// The max limit of the search depth in DecomposeGEPExpression() and
// getUnderlyingObject(), both functions need to use the same search
// depth otherwise the algorithm in aliasGEP will assert.
static const unsigned MaxLookupSearchDepth = 6;

bool BasicAAResult::invalidate(Function &Fn, const PreservedAnalyses &PA,
                               FunctionAnalysisManager::Invalidator &Inv) {
  // We don't care if this analysis itself is preserved, it has no state. But
  // we need to check that the analyses it depends on have been. Note that we
  // may be created without handles to some analyses and in that case don't
  // depend on them.
  if (Inv.invalidate<AssumptionAnalysis>(Fn, PA) ||
      (DT && Inv.invalidate<DominatorTreeAnalysis>(Fn, PA)) ||
      (LI && Inv.invalidate<LoopAnalysis>(Fn, PA)) ||
      (PV && Inv.invalidate<PhiValuesAnalysis>(Fn, PA)))
    return true;

  // Otherwise this analysis result remains valid.
  return false;
}

//===----------------------------------------------------------------------===//
// Useful predicates
//===----------------------------------------------------------------------===//

/// Returns true if the pointer is one which would have been considered an
/// escape by isNonEscapingLocalObject.
static bool isEscapeSource(const Value *V) {
  if (isa<CallBase>(V))
    return true;

  if (isa<Argument>(V))
    return true;

  // The load case works because isNonEscapingLocalObject considers all
  // stores to be escapes (it passes true for the StoreCaptures argument
  // to PointerMayBeCaptured).
  if (isa<LoadInst>(V))
    return true;

  return false;
}

/// Returns the size of the object specified by V or UnknownSize if unknown.
static uint64_t getObjectSize(const Value *V, const DataLayout &DL,
                              const TargetLibraryInfo &TLI,
                              bool NullIsValidLoc,
                              bool RoundToAlign = false) {
  uint64_t Size;
  ObjectSizeOpts Opts;
  Opts.RoundToAlign = RoundToAlign;
  Opts.NullIsUnknownSize = NullIsValidLoc;
  if (getObjectSize(V, Size, DL, &TLI, Opts))
    return Size;
  return MemoryLocation::UnknownSize;
}

/// Returns true if we can prove that the object specified by V is smaller than
/// Size.
static bool isObjectSmallerThan(const Value *V, uint64_t Size,
                                const DataLayout &DL,
                                const TargetLibraryInfo &TLI,
                                bool NullIsValidLoc) {
  // Note that the meanings of the "object" are slightly different in the
  // following contexts:
  //    c1: llvm::getObjectSize()
  //    c2: llvm.objectsize() intrinsic
  //    c3: isObjectSmallerThan()
  // c1 and c2 share the same meaning; however, the meaning of "object" in c3
  // refers to the "entire object".
  //
  //  Consider this example:
  //     char *p = (char*)malloc(100)
  //     char *q = p+80;
  //
  //  In the context of c1 and c2, the "object" pointed by q refers to the
  // stretch of memory of q[0:19]. So, getObjectSize(q) should return 20.
  //
  //  However, in the context of c3, the "object" refers to the chunk of memory
  // being allocated. So, the "object" has 100 bytes, and q points to the middle
  // the "object". In case q is passed to isObjectSmallerThan() as the 1st
  // parameter, before the llvm::getObjectSize() is called to get the size of
  // entire object, we should:
  //    - either rewind the pointer q to the base-address of the object in
  //      question (in this case rewind to p), or
  //    - just give up. It is up to caller to make sure the pointer is pointing
  //      to the base address the object.
  //
  // We go for 2nd option for simplicity.
  if (!isIdentifiedObject(V))
    return false;

  // This function needs to use the aligned object size because we allow
  // reads a bit past the end given sufficient alignment.
  uint64_t ObjectSize = getObjectSize(V, DL, TLI, NullIsValidLoc,
                                      /*RoundToAlign*/ true);

  return ObjectSize != MemoryLocation::UnknownSize && ObjectSize < Size;
}

/// Return the minimal extent from \p V to the end of the underlying object,
/// assuming the result is used in an aliasing query. E.g., we do use the query
/// location size and the fact that null pointers cannot alias here.
static uint64_t getMinimalExtentFrom(const Value &V,
                                     const LocationSize &LocSize,
                                     const DataLayout &DL,
                                     bool NullIsValidLoc) {
  // If we have dereferenceability information we know a lower bound for the
  // extent as accesses for a lower offset would be valid. We need to exclude
  // the "or null" part if null is a valid pointer.
  bool CanBeNull;
  uint64_t DerefBytes = V.getPointerDereferenceableBytes(DL, CanBeNull);
  DerefBytes = (CanBeNull && NullIsValidLoc) ? 0 : DerefBytes;
  // If queried with a precise location size, we assume that location size to be
  // accessed, thus valid.
  if (LocSize.isPrecise())
    DerefBytes = std::max(DerefBytes, LocSize.getValue());
  return DerefBytes;
}

/// Returns true if we can prove that the object specified by V has size Size.
static bool isObjectSize(const Value *V, uint64_t Size, const DataLayout &DL,
                         const TargetLibraryInfo &TLI, bool NullIsValidLoc) {
  uint64_t ObjectSize = getObjectSize(V, DL, TLI, NullIsValidLoc);
  return ObjectSize != MemoryLocation::UnknownSize && ObjectSize == Size;
}

//===----------------------------------------------------------------------===//
// GetElementPtr Instruction Decomposition and Analysis
//===----------------------------------------------------------------------===//

/// Analyzes the specified value as a linear expression: "A*V + B", where A and
/// B are constant integers.
///
/// Returns the scale and offset values as APInts and return V as a Value*, and
/// return whether we looked through any sign or zero extends.  The incoming
/// Value is known to have IntegerType, and it may already be sign or zero
/// extended.
///
/// Note that this looks through extends, so the high bits may not be
/// represented in the result.
/*static*/ const Value *BasicAAResult::GetLinearExpression(
    const Value *V, APInt &Scale, APInt &Offset, unsigned &ZExtBits,
    unsigned &SExtBits, const DataLayout &DL, unsigned Depth,
    AssumptionCache *AC, DominatorTree *DT, bool &NSW, bool &NUW) {
  assert(V->getType()->isIntegerTy() && "Not an integer value");

  // Limit our recursion depth.
  if (Depth == 6) {
    Scale = 1;
    Offset = 0;
    return V;
  }

  if (const ConstantInt *Const = dyn_cast<ConstantInt>(V)) {
    // If it's a constant, just convert it to an offset and remove the variable.
    // If we've been called recursively, the Offset bit width will be greater
    // than the constant's (the Offset's always as wide as the outermost call),
    // so we'll zext here and process any extension in the isa<SExtInst> &
    // isa<ZExtInst> cases below.
    Offset += Const->getValue().zextOrSelf(Offset.getBitWidth());
    assert(Scale == 0 && "Constant values don't have a scale");
    return V;
  }

  if (const BinaryOperator *BOp = dyn_cast<BinaryOperator>(V)) {
    if (ConstantInt *RHSC = dyn_cast<ConstantInt>(BOp->getOperand(1))) {
      // If we've been called recursively, then Offset and Scale will be wider
      // than the BOp operands. We'll always zext it here as we'll process sign
      // extensions below (see the isa<SExtInst> / isa<ZExtInst> cases).
      APInt RHS = RHSC->getValue().zextOrSelf(Offset.getBitWidth());

      switch (BOp->getOpcode()) {
      default:
        // We don't understand this instruction, so we can't decompose it any
        // further.
        Scale = 1;
        Offset = 0;
        return V;
      case Instruction::Or:
        // X|C == X+C if all the bits in C are unset in X.  Otherwise we can't
        // analyze it.
        if (!MaskedValueIsZero(BOp->getOperand(0), RHSC->getValue(), DL, 0, AC,
                               BOp, DT)) {
          Scale = 1;
          Offset = 0;
          return V;
        }
        LLVM_FALLTHROUGH;
      case Instruction::Add:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset += RHS;
        break;
      case Instruction::Sub:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset -= RHS;
        break;
      case Instruction::Mul:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);
        Offset *= RHS;
        Scale *= RHS;
        break;
      case Instruction::Shl:
        V = GetLinearExpression(BOp->getOperand(0), Scale, Offset, ZExtBits,
                                SExtBits, DL, Depth + 1, AC, DT, NSW, NUW);

        // We're trying to linearize an expression of the kind:
        //   shl i8 -128, 36
        // where the shift count exceeds the bitwidth of the type.
        // We can't decompose this further (the expression would return
        // a poison value).
        if (Offset.getBitWidth() < RHS.getLimitedValue() ||
            Scale.getBitWidth() < RHS.getLimitedValue()) {
          Scale = 1;
          Offset = 0;
          return V;
        }

        Offset <<= RHS.getLimitedValue();
        Scale <<= RHS.getLimitedValue();
        // the semantics of nsw and nuw for left shifts don't match those of
        // multiplications, so we won't propagate them.
        NSW = NUW = false;
        return V;
      }

      if (isa<OverflowingBinaryOperator>(BOp)) {
        NUW &= BOp->hasNoUnsignedWrap();
        NSW &= BOp->hasNoSignedWrap();
      }
      return V;
    }
  }

  // Since GEP indices are sign extended anyway, we don't care about the high
  // bits of a sign or zero extended value - just scales and offsets.  The
  // extensions have to be consistent though.
  if (isa<SExtInst>(V) || isa<ZExtInst>(V)) {
    Value *CastOp = cast<CastInst>(V)->getOperand(0);
    unsigned NewWidth = V->getType()->getPrimitiveSizeInBits();
    unsigned SmallWidth = CastOp->getType()->getPrimitiveSizeInBits();
    unsigned OldZExtBits = ZExtBits, OldSExtBits = SExtBits;
    const Value *Result =
        GetLinearExpression(CastOp, Scale, Offset, ZExtBits, SExtBits, DL,
                            Depth + 1, AC, DT, NSW, NUW);

    // zext(zext(%x)) == zext(%x), and similarly for sext; we'll handle this
    // by just incrementing the number of bits we've extended by.
    unsigned ExtendedBy = NewWidth - SmallWidth;

    if (isa<SExtInst>(V) && ZExtBits == 0) {
      // sext(sext(%x, a), b) == sext(%x, a + b)

      if (NSW) {
        // We haven't sign-wrapped, so it's valid to decompose sext(%x + c)
        // into sext(%x) + sext(c). We'll sext the Offset ourselves:
        unsigned OldWidth = Offset.getBitWidth();
        Offset = Offset.trunc(SmallWidth).sext(NewWidth).zextOrSelf(OldWidth);
      } else {
        // We may have signed-wrapped, so don't decompose sext(%x + c) into
        // sext(%x) + sext(c)
        Scale = 1;
        Offset = 0;
        Result = CastOp;
        ZExtBits = OldZExtBits;
        SExtBits = OldSExtBits;
      }
      SExtBits += ExtendedBy;
    } else {
      // sext(zext(%x, a), b) = zext(zext(%x, a), b) = zext(%x, a + b)

      if (!NUW) {
        // We may have unsigned-wrapped, so don't decompose zext(%x + c) into
        // zext(%x) + zext(c)
        Scale = 1;
        Offset = 0;
        Result = CastOp;
        ZExtBits = OldZExtBits;
        SExtBits = OldSExtBits;
      }
      ZExtBits += ExtendedBy;
    }

    return Result;
  }

  Scale = 1;
  Offset = 0;
  return V;
}

/// To ensure a pointer offset fits in an integer of size PointerSize
/// (in bits) when that size is smaller than the maximum pointer size. This is
/// an issue, for example, in particular for 32b pointers with negative indices
/// that rely on two's complement wrap-arounds for precise alias information
/// where the maximum pointer size is 64b.
static APInt adjustToPointerSize(const APInt &Offset, unsigned PointerSize) {
  assert(PointerSize <= Offset.getBitWidth() && "Invalid PointerSize!");
  unsigned ShiftBits = Offset.getBitWidth() - PointerSize;
  return (Offset << ShiftBits).ashr(ShiftBits);
}

static unsigned getMaxPointerSize(const DataLayout &DL) {
  unsigned MaxPointerSize = DL.getMaxPointerSizeInBits();
  if (MaxPointerSize < 64 && ForceAtLeast64Bits) MaxPointerSize = 64;
  if (DoubleCalcBits) MaxPointerSize *= 2;

  return MaxPointerSize;
}

/// If V is a symbolic pointer expression, decompose it into a base pointer
/// with a constant offset and a number of scaled symbolic offsets.
///
/// The scaled symbolic offsets (represented by pairs of a Value* and a scale
/// in the VarIndices vector) are Value*'s that are known to be scaled by the
/// specified amount, but which may have other unrepresented high bits. As
/// such, the gep cannot necessarily be reconstructed from its decomposed form.
///
/// This function is capable of analyzing everything that getUnderlyingObject
/// can look through. To be able to do that getUnderlyingObject and
/// DecomposeGEPExpression must use the same search depth
/// (MaxLookupSearchDepth).
BasicAAResult::DecomposedGEP
BasicAAResult::DecomposeGEPExpression(const Value *V, const DataLayout &DL,
                                      AssumptionCache *AC, DominatorTree *DT) {
  // Limit recursion depth to limit compile time in crazy cases.
  unsigned MaxLookup = MaxLookupSearchDepth;
  SearchTimes++;
  const Instruction *CxtI = dyn_cast<Instruction>(V);

  unsigned MaxPointerSize = getMaxPointerSize(DL);
  DecomposedGEP Decomposed;
  Decomposed.Offset = APInt(MaxPointerSize, 0);
  Decomposed.HasCompileTimeConstantScale = true;
  do {
    // See if this is a bitcast or GEP.
    const Operator *Op = dyn_cast<Operator>(V);
    if (!Op) {
      // The only non-operator case we can handle are GlobalAliases.
      if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(V)) {
        if (!GA->isInterposable()) {
          V = GA->getAliasee();
          continue;
        }
      }
      Decomposed.Base = V;
      return Decomposed;
    }

    if (Op->getOpcode() == Instruction::BitCast ||
        Op->getOpcode() == Instruction::AddrSpaceCast) {
      V = Op->getOperand(0);
      continue;
    }

    const GEPOperator *GEPOp = dyn_cast<GEPOperator>(Op);
    if (!GEPOp) {
      if (const auto *PHI = dyn_cast<PHINode>(V)) {
        // Look through single-arg phi nodes created by LCSSA.
        if (PHI->getNumIncomingValues() == 1) {
          V = PHI->getIncomingValue(0);
          continue;
        }
      } else if (const auto *Call = dyn_cast<CallBase>(V)) {
        // CaptureTracking can know about special capturing properties of some
        // intrinsics like launder.invariant.group, that can't be expressed with
        // the attributes, but have properties like returning aliasing pointer.
        // Because some analysis may assume that nocaptured pointer is not
        // returned from some special intrinsic (because function would have to
        // be marked with returns attribute), it is crucial to use this function
        // because it should be in sync with CaptureTracking. Not using it may
        // cause weird miscompilations where 2 aliasing pointers are assumed to
        // noalias.
        if (auto *RP = getArgumentAliasingToReturnedPointer(Call, false)) {
          V = RP;
          continue;
        }
      }

      Decomposed.Base = V;
      return Decomposed;
    }

    // Don't attempt to analyze GEPs over unsized objects.
    if (!GEPOp->getSourceElementType()->isSized()) {
      Decomposed.Base = V;
      return Decomposed;
    }

    // Don't attempt to analyze GEPs if index scale is not a compile-time
    // constant.
    if (isa<ScalableVectorType>(GEPOp->getSourceElementType())) {
      Decomposed.Base = V;
      Decomposed.HasCompileTimeConstantScale = false;
      return Decomposed;
    }

    unsigned AS = GEPOp->getPointerAddressSpace();
    // Walk the indices of the GEP, accumulating them into BaseOff/VarIndices.
    gep_type_iterator GTI = gep_type_begin(GEPOp);
    unsigned PointerSize = DL.getPointerSizeInBits(AS);
    // Assume all GEP operands are constants until proven otherwise.
    bool GepHasConstantOffset = true;
    for (User::const_op_iterator I = GEPOp->op_begin() + 1, E = GEPOp->op_end();
         I != E; ++I, ++GTI) {
      const Value *Index = *I;
      // Compute the (potentially symbolic) offset in bytes for this index.
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        // For a struct, add the member offset.
        unsigned FieldNo = cast<ConstantInt>(Index)->getZExtValue();
        if (FieldNo == 0)
          continue;

        Decomposed.Offset += DL.getStructLayout(STy)->getElementOffset(FieldNo);
        continue;
      }

      // For an array/pointer, add the element offset, explicitly scaled.
      if (const ConstantInt *CIdx = dyn_cast<ConstantInt>(Index)) {
        if (CIdx->isZero())
          continue;
        Decomposed.Offset +=
            DL.getTypeAllocSize(GTI.getIndexedType()).getFixedSize() *
            CIdx->getValue().sextOrTrunc(MaxPointerSize);
        continue;
      }

      GepHasConstantOffset = false;

      APInt Scale(MaxPointerSize,
                  DL.getTypeAllocSize(GTI.getIndexedType()).getFixedSize());
      unsigned ZExtBits = 0, SExtBits = 0;

      // If the integer type is smaller than the pointer size, it is implicitly
      // sign extended to pointer size.
      unsigned Width = Index->getType()->getIntegerBitWidth();
      if (PointerSize > Width)
        SExtBits += PointerSize - Width;

      // Use GetLinearExpression to decompose the index into a C1*V+C2 form.
      APInt IndexScale(Width, 0), IndexOffset(Width, 0);
      bool NSW = true, NUW = true;
      const Value *OrigIndex = Index;
      Index = GetLinearExpression(Index, IndexScale, IndexOffset, ZExtBits,
                                  SExtBits, DL, 0, AC, DT, NSW, NUW);

      // The GEP index scale ("Scale") scales C1*V+C2, yielding (C1*V+C2)*Scale.
      // This gives us an aggregate computation of (C1*Scale)*V + C2*Scale.

      // It can be the case that, even through C1*V+C2 does not overflow for
      // relevant values of V, (C2*Scale) can overflow. In that case, we cannot
      // decompose the expression in this way.
      //
      // FIXME: C1*Scale and the other operations in the decomposed
      // (C1*Scale)*V+C2*Scale can also overflow. We should check for this
      // possibility.
      bool Overflow;
      APInt ScaledOffset = IndexOffset.sextOrTrunc(MaxPointerSize)
                           .smul_ov(Scale, Overflow);
      if (Overflow) {
        Index = OrigIndex;
        IndexScale = 1;
        IndexOffset = 0;

        ZExtBits = SExtBits = 0;
        if (PointerSize > Width)
          SExtBits += PointerSize - Width;
      } else {
        Decomposed.Offset += ScaledOffset;
        Scale *= IndexScale.sextOrTrunc(MaxPointerSize);
      }

      // If we already had an occurrence of this index variable, merge this
      // scale into it.  For example, we want to handle:
      //   A[x][x] -> x*16 + x*4 -> x*20
      // This also ensures that 'x' only appears in the index list once.
      for (unsigned i = 0, e = Decomposed.VarIndices.size(); i != e; ++i) {
        if (Decomposed.VarIndices[i].V == Index &&
            Decomposed.VarIndices[i].ZExtBits == ZExtBits &&
            Decomposed.VarIndices[i].SExtBits == SExtBits) {
          Scale += Decomposed.VarIndices[i].Scale;
          Decomposed.VarIndices.erase(Decomposed.VarIndices.begin() + i);
          break;
        }
      }

      // Make sure that we have a scale that makes sense for this target's
      // pointer size.
      Scale = adjustToPointerSize(Scale, PointerSize);

      if (!!Scale) {
        VariableGEPIndex Entry = {Index, ZExtBits, SExtBits, Scale, CxtI};
        Decomposed.VarIndices.push_back(Entry);
      }
    }

    // Take care of wrap-arounds
    if (GepHasConstantOffset)
      Decomposed.Offset = adjustToPointerSize(Decomposed.Offset, PointerSize);

    // Analyze the base pointer next.
    V = GEPOp->getOperand(0);
  } while (--MaxLookup);

  // If the chain of expressions is too deep, just return early.
  Decomposed.Base = V;
  SearchLimitReached++;
  return Decomposed;
}

/// Returns whether the given pointer value points to memory that is local to
/// the function, with global constants being considered local to all
/// functions.
bool BasicAAResult::pointsToConstantMemory(const MemoryLocation &Loc,
                                           AAQueryInfo &AAQI, bool OrLocal) {
  assert(Visited.empty() && "Visited must be cleared after use!");

  unsigned MaxLookup = 8;
  SmallVector<const Value *, 16> Worklist;
  Worklist.push_back(Loc.Ptr);
  do {
    const Value *V = getUnderlyingObject(Worklist.pop_back_val());
    if (!Visited.insert(V).second) {
      Visited.clear();
      return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
    }

    // An alloca instruction defines local memory.
    if (OrLocal && isa<AllocaInst>(V))
      continue;

    // A global constant counts as local memory for our purposes.
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      // Note: this doesn't require GV to be "ODR" because it isn't legal for a
      // global to be marked constant in some modules and non-constant in
      // others.  GV may even be a declaration, not a definition.
      if (!GV->isConstant()) {
        Visited.clear();
        return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
      }
      continue;
    }

    // If both select values point to local memory, then so does the select.
    if (const SelectInst *SI = dyn_cast<SelectInst>(V)) {
      Worklist.push_back(SI->getTrueValue());
      Worklist.push_back(SI->getFalseValue());
      continue;
    }

    // If all values incoming to a phi node point to local memory, then so does
    // the phi.
    if (const PHINode *PN = dyn_cast<PHINode>(V)) {
      // Don't bother inspecting phi nodes with many operands.
      if (PN->getNumIncomingValues() > MaxLookup) {
        Visited.clear();
        return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
      }
      append_range(Worklist, PN->incoming_values());
      continue;
    }

    // Otherwise be conservative.
    Visited.clear();
    return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
  } while (!Worklist.empty() && --MaxLookup);

  Visited.clear();
  return Worklist.empty();
}

/// Returns the behavior when calling the given call site.
FunctionModRefBehavior BasicAAResult::getModRefBehavior(const CallBase *Call) {
  if (Call->doesNotAccessMemory())
    // Can't do better than this.
    return FMRB_DoesNotAccessMemory;

  FunctionModRefBehavior Min = FMRB_UnknownModRefBehavior;

  // If the callsite knows it only reads memory, don't return worse
  // than that.
  if (Call->onlyReadsMemory())
    Min = FMRB_OnlyReadsMemory;
  else if (Call->doesNotReadMemory())
    Min = FMRB_OnlyWritesMemory;

  if (Call->onlyAccessesArgMemory())
    Min = FunctionModRefBehavior(Min & FMRB_OnlyAccessesArgumentPointees);
  else if (Call->onlyAccessesInaccessibleMemory())
    Min = FunctionModRefBehavior(Min & FMRB_OnlyAccessesInaccessibleMem);
  else if (Call->onlyAccessesInaccessibleMemOrArgMem())
    Min = FunctionModRefBehavior(Min & FMRB_OnlyAccessesInaccessibleOrArgMem);

  // If the call has operand bundles then aliasing attributes from the function
  // it calls do not directly apply to the call.  This can be made more precise
  // in the future.
  if (!Call->hasOperandBundles())
    if (const Function *F = Call->getCalledFunction())
      Min =
          FunctionModRefBehavior(Min & getBestAAResults().getModRefBehavior(F));

  return Min;
}

/// Returns the behavior when calling the given function. For use when the call
/// site is not known.
FunctionModRefBehavior BasicAAResult::getModRefBehavior(const Function *F) {
  // If the function declares it doesn't access memory, we can't do better.
  if (F->doesNotAccessMemory())
    return FMRB_DoesNotAccessMemory;

  FunctionModRefBehavior Min = FMRB_UnknownModRefBehavior;

  // If the function declares it only reads memory, go with that.
  if (F->onlyReadsMemory())
    Min = FMRB_OnlyReadsMemory;
  else if (F->doesNotReadMemory())
    Min = FMRB_OnlyWritesMemory;

  if (F->onlyAccessesArgMemory())
    Min = FunctionModRefBehavior(Min & FMRB_OnlyAccessesArgumentPointees);
  else if (F->onlyAccessesInaccessibleMemory())
    Min = FunctionModRefBehavior(Min & FMRB_OnlyAccessesInaccessibleMem);
  else if (F->onlyAccessesInaccessibleMemOrArgMem())
    Min = FunctionModRefBehavior(Min & FMRB_OnlyAccessesInaccessibleOrArgMem);

  return Min;
}

/// Returns true if this is a writeonly (i.e Mod only) parameter.
static bool isWriteOnlyParam(const CallBase *Call, unsigned ArgIdx,
                             const TargetLibraryInfo &TLI) {
  if (Call->paramHasAttr(ArgIdx, Attribute::WriteOnly))
    return true;

  // We can bound the aliasing properties of memset_pattern16 just as we can
  // for memcpy/memset.  This is particularly important because the
  // LoopIdiomRecognizer likes to turn loops into calls to memset_pattern16
  // whenever possible.
  // FIXME Consider handling this in InferFunctionAttr.cpp together with other
  // attributes.
  LibFunc F;
  if (Call->getCalledFunction() &&
      TLI.getLibFunc(*Call->getCalledFunction(), F) &&
      F == LibFunc_memset_pattern16 && TLI.has(F))
    if (ArgIdx == 0)
      return true;

  // TODO: memset_pattern4, memset_pattern8
  // TODO: _chk variants
  // TODO: strcmp, strcpy

  return false;
}

ModRefInfo BasicAAResult::getArgModRefInfo(const CallBase *Call,
                                           unsigned ArgIdx) {
  // Checking for known builtin intrinsics and target library functions.
  if (isWriteOnlyParam(Call, ArgIdx, TLI))
    return ModRefInfo::Mod;

  if (Call->paramHasAttr(ArgIdx, Attribute::ReadOnly))
    return ModRefInfo::Ref;

  if (Call->paramHasAttr(ArgIdx, Attribute::ReadNone))
    return ModRefInfo::NoModRef;

  return AAResultBase::getArgModRefInfo(Call, ArgIdx);
}

static bool isIntrinsicCall(const CallBase *Call, Intrinsic::ID IID) {
  const IntrinsicInst *II = dyn_cast<IntrinsicInst>(Call);
  return II && II->getIntrinsicID() == IID;
}

#ifndef NDEBUG
static const Function *getParent(const Value *V) {
  if (const Instruction *inst = dyn_cast<Instruction>(V)) {
    if (!inst->getParent())
      return nullptr;
    return inst->getParent()->getParent();
  }

  if (const Argument *arg = dyn_cast<Argument>(V))
    return arg->getParent();

  return nullptr;
}

static bool notDifferentParent(const Value *O1, const Value *O2) {

  const Function *F1 = getParent(O1);
  const Function *F2 = getParent(O2);

  return !F1 || !F2 || F1 == F2;
}
#endif

AliasResult BasicAAResult::alias(const MemoryLocation &LocA,
                                 const MemoryLocation &LocB,
                                 AAQueryInfo &AAQI) {
  assert(notDifferentParent(LocA.Ptr, LocB.Ptr) &&
         "BasicAliasAnalysis doesn't support interprocedural queries.");
  return aliasCheck(LocA.Ptr, LocA.Size, LocA.AATags, LocB.Ptr, LocB.Size,
                    LocB.AATags, AAQI);
}

/// Checks to see if the specified callsite can clobber the specified memory
/// object.
///
/// Since we only look at local properties of this function, we really can't
/// say much about this query.  We do, however, use simple "address taken"
/// analysis on local objects.
ModRefInfo BasicAAResult::getModRefInfo(const CallBase *Call,
                                        const MemoryLocation &Loc,
                                        AAQueryInfo &AAQI) {
  assert(notDifferentParent(Call, Loc.Ptr) &&
         "AliasAnalysis query involving multiple functions!");

  const Value *Object = getUnderlyingObject(Loc.Ptr);

  // Calls marked 'tail' cannot read or write allocas from the current frame
  // because the current frame might be destroyed by the time they run. However,
  // a tail call may use an alloca with byval. Calling with byval copies the
  // contents of the alloca into argument registers or stack slots, so there is
  // no lifetime issue.
  if (isa<AllocaInst>(Object))
    if (const CallInst *CI = dyn_cast<CallInst>(Call))
      if (CI->isTailCall() &&
          !CI->getAttributes().hasAttrSomewhere(Attribute::ByVal))
        return ModRefInfo::NoModRef;

  // Stack restore is able to modify unescaped dynamic allocas. Assume it may
  // modify them even though the alloca is not escaped.
  if (auto *AI = dyn_cast<AllocaInst>(Object))
    if (!AI->isStaticAlloca() && isIntrinsicCall(Call, Intrinsic::stackrestore))
      return ModRefInfo::Mod;

  // If the pointer is to a locally allocated object that does not escape,
  // then the call can not mod/ref the pointer unless the call takes the pointer
  // as an argument, and itself doesn't capture it.
  if (!isa<Constant>(Object) && Call != Object &&
      isNonEscapingLocalObject(Object, &AAQI.IsCapturedCache)) {

    // Optimistically assume that call doesn't touch Object and check this
    // assumption in the following loop.
    ModRefInfo Result = ModRefInfo::NoModRef;
    bool IsMustAlias = true;

    unsigned OperandNo = 0;
    for (auto CI = Call->data_operands_begin(), CE = Call->data_operands_end();
         CI != CE; ++CI, ++OperandNo) {
      // Only look at the no-capture or byval pointer arguments.  If this
      // pointer were passed to arguments that were neither of these, then it
      // couldn't be no-capture.
      if (!(*CI)->getType()->isPointerTy() ||
          (!Call->doesNotCapture(OperandNo) &&
           OperandNo < Call->getNumArgOperands() &&
           !Call->isByValArgument(OperandNo)))
        continue;

      // Call doesn't access memory through this operand, so we don't care
      // if it aliases with Object.
      if (Call->doesNotAccessMemory(OperandNo))
        continue;

      // If this is a no-capture pointer argument, see if we can tell that it
      // is impossible to alias the pointer we're checking.
      AliasResult AR = getBestAAResults().alias(
          MemoryLocation::getBeforeOrAfter(*CI),
          MemoryLocation::getBeforeOrAfter(Object), AAQI);
      if (AR != MustAlias)
        IsMustAlias = false;
      // Operand doesn't alias 'Object', continue looking for other aliases
      if (AR == NoAlias)
        continue;
      // Operand aliases 'Object', but call doesn't modify it. Strengthen
      // initial assumption and keep looking in case if there are more aliases.
      if (Call->onlyReadsMemory(OperandNo)) {
        Result = setRef(Result);
        continue;
      }
      // Operand aliases 'Object' but call only writes into it.
      if (Call->doesNotReadMemory(OperandNo)) {
        Result = setMod(Result);
        continue;
      }
      // This operand aliases 'Object' and call reads and writes into it.
      // Setting ModRef will not yield an early return below, MustAlias is not
      // used further.
      Result = ModRefInfo::ModRef;
      break;
    }

    // No operand aliases, reset Must bit. Add below if at least one aliases
    // and all aliases found are MustAlias.
    if (isNoModRef(Result))
      IsMustAlias = false;

    // Early return if we improved mod ref information
    if (!isModAndRefSet(Result)) {
      if (isNoModRef(Result))
        return ModRefInfo::NoModRef;
      return IsMustAlias ? setMust(Result) : clearMust(Result);
    }
  }

  // If the call is malloc/calloc like, we can assume that it doesn't
  // modify any IR visible value.  This is only valid because we assume these
  // routines do not read values visible in the IR.  TODO: Consider special
  // casing realloc and strdup routines which access only their arguments as
  // well.  Or alternatively, replace all of this with inaccessiblememonly once
  // that's implemented fully.
  if (isMallocOrCallocLikeFn(Call, &TLI)) {
    // Be conservative if the accessed pointer may alias the allocation -
    // fallback to the generic handling below.
    if (getBestAAResults().alias(MemoryLocation::getBeforeOrAfter(Call),
                                 Loc, AAQI) == NoAlias)
      return ModRefInfo::NoModRef;
  }

  // The semantics of memcpy intrinsics either exactly overlap or do not
  // overlap, i.e., source and destination of any given memcpy are either
  // no-alias or must-alias.
  if (auto *Inst = dyn_cast<AnyMemCpyInst>(Call)) {
    AliasResult SrcAA =
        getBestAAResults().alias(MemoryLocation::getForSource(Inst), Loc, AAQI);
    AliasResult DestAA =
        getBestAAResults().alias(MemoryLocation::getForDest(Inst), Loc, AAQI);
    // It's also possible for Loc to alias both src and dest, or neither.
    ModRefInfo rv = ModRefInfo::NoModRef;
    if (SrcAA != NoAlias)
      rv = setRef(rv);
    if (DestAA != NoAlias)
      rv = setMod(rv);
    return rv;
  }

  // While the assume intrinsic is marked as arbitrarily writing so that
  // proper control dependencies will be maintained, it never aliases any
  // particular memory location.
  if (isIntrinsicCall(Call, Intrinsic::assume))
    return ModRefInfo::NoModRef;

  // Like assumes, guard intrinsics are also marked as arbitrarily writing so
  // that proper control dependencies are maintained but they never mods any
  // particular memory location.
  //
  // *Unlike* assumes, guard intrinsics are modeled as reading memory since the
  // heap state at the point the guard is issued needs to be consistent in case
  // the guard invokes the "deopt" continuation.
  if (isIntrinsicCall(Call, Intrinsic::experimental_guard))
    return ModRefInfo::Ref;
  // The same applies to deoptimize which is essentially a guard(false).
  if (isIntrinsicCall(Call, Intrinsic::experimental_deoptimize))
    return ModRefInfo::Ref;

  // Like assumes, invariant.start intrinsics were also marked as arbitrarily
  // writing so that proper control dependencies are maintained but they never
  // mod any particular memory location visible to the IR.
  // *Unlike* assumes (which are now modeled as NoModRef), invariant.start
  // intrinsic is now modeled as reading memory. This prevents hoisting the
  // invariant.start intrinsic over stores. Consider:
  // *ptr = 40;
  // *ptr = 50;
  // invariant_start(ptr)
  // int val = *ptr;
  // print(val);
  //
  // This cannot be transformed to:
  //
  // *ptr = 40;
  // invariant_start(ptr)
  // *ptr = 50;
  // int val = *ptr;
  // print(val);
  //
  // The transformation will cause the second store to be ignored (based on
  // rules of invariant.start)  and print 40, while the first program always
  // prints 50.
  if (isIntrinsicCall(Call, Intrinsic::invariant_start))
    return ModRefInfo::Ref;

  // The AAResultBase base class has some smarts, lets use them.
  return AAResultBase::getModRefInfo(Call, Loc, AAQI);
}

ModRefInfo BasicAAResult::getModRefInfo(const CallBase *Call1,
                                        const CallBase *Call2,
                                        AAQueryInfo &AAQI) {
  // While the assume intrinsic is marked as arbitrarily writing so that
  // proper control dependencies will be maintained, it never aliases any
  // particular memory location.
  if (isIntrinsicCall(Call1, Intrinsic::assume) ||
      isIntrinsicCall(Call2, Intrinsic::assume))
    return ModRefInfo::NoModRef;

  // Like assumes, guard intrinsics are also marked as arbitrarily writing so
  // that proper control dependencies are maintained but they never mod any
  // particular memory location.
  //
  // *Unlike* assumes, guard intrinsics are modeled as reading memory since the
  // heap state at the point the guard is issued needs to be consistent in case
  // the guard invokes the "deopt" continuation.

  // NB! This function is *not* commutative, so we special case two
  // possibilities for guard intrinsics.

  if (isIntrinsicCall(Call1, Intrinsic::experimental_guard))
    return isModSet(createModRefInfo(getModRefBehavior(Call2)))
               ? ModRefInfo::Ref
               : ModRefInfo::NoModRef;

  if (isIntrinsicCall(Call2, Intrinsic::experimental_guard))
    return isModSet(createModRefInfo(getModRefBehavior(Call1)))
               ? ModRefInfo::Mod
               : ModRefInfo::NoModRef;

  // The AAResultBase base class has some smarts, lets use them.
  return AAResultBase::getModRefInfo(Call1, Call2, AAQI);
}

// If a we have (a) a GEP and (b) a pointer based on an alloca, and the
// beginning of the object the GEP points would have a negative offset with
// repsect to the alloca, that means the GEP can not alias pointer (b).
// Note that the pointer based on the alloca may not be a GEP. For
// example, it may be the alloca itself.
// The same applies if (b) is based on a GlobalVariable. Note that just being
// based on isIdentifiedObject() is not enough - we need an identified object
// that does not permit access to negative offsets. For example, a negative
// offset from a noalias argument or call can be inbounds w.r.t the actual
// underlying object.
//
// For example, consider:
//
//   struct { int f0, int f1, ...} foo;
//   foo alloca;
//   foo* random = bar(alloca);
//   int *f0 = &alloca.f0
//   int *f1 = &random->f1;
//
// Which is lowered, approximately, to:
//
//  %alloca = alloca %struct.foo
//  %random = call %struct.foo* @random(%struct.foo* %alloca)
//  %f0 = getelementptr inbounds %struct, %struct.foo* %alloca, i32 0, i32 0
//  %f1 = getelementptr inbounds %struct, %struct.foo* %random, i32 0, i32 1
//
// Assume %f1 and %f0 alias. Then %f1 would point into the object allocated
// by %alloca. Since the %f1 GEP is inbounds, that means %random must also
// point into the same object. But since %f0 points to the beginning of %alloca,
// the highest %f1 can be is (%alloca + 3). This means %random can not be higher
// than (%alloca - 1), and so is not inbounds, a contradiction.
bool BasicAAResult::isGEPBaseAtNegativeOffset(const GEPOperator *GEPOp,
      const DecomposedGEP &DecompGEP, const DecomposedGEP &DecompObject,
      LocationSize MaybeObjectAccessSize) {
  // If the object access size is unknown, or the GEP isn't inbounds, bail.
  if (!MaybeObjectAccessSize.hasValue() || !GEPOp->isInBounds())
    return false;

  const uint64_t ObjectAccessSize = MaybeObjectAccessSize.getValue();

  // We need the object to be an alloca or a globalvariable, and want to know
  // the offset of the pointer from the object precisely, so no variable
  // indices are allowed.
  if (!(isa<AllocaInst>(DecompObject.Base) ||
        isa<GlobalVariable>(DecompObject.Base)) ||
      !DecompObject.VarIndices.empty())
    return false;

  // If the GEP has no variable indices, we know the precise offset
  // from the base, then use it. If the GEP has variable indices,
  // we can't get exact GEP offset to identify pointer alias. So return
  // false in that case.
  if (!DecompGEP.VarIndices.empty())
    return false;

  return DecompGEP.Offset.sge(DecompObject.Offset + (int64_t)ObjectAccessSize);
}

/// Provides a bunch of ad-hoc rules to disambiguate a GEP instruction against
/// another pointer.
///
/// We know that V1 is a GEP, but we don't know anything about V2.
/// UnderlyingV1 is getUnderlyingObject(GEP1), UnderlyingV2 is the same for
/// V2.
AliasResult BasicAAResult::aliasGEP(
    const GEPOperator *GEP1, LocationSize V1Size, const AAMDNodes &V1AAInfo,
    const Value *V2, LocationSize V2Size, const AAMDNodes &V2AAInfo,
    const Value *UnderlyingV1, const Value *UnderlyingV2, AAQueryInfo &AAQI) {
  DecomposedGEP DecompGEP1 = DecomposeGEPExpression(GEP1, DL, &AC, DT);
  DecomposedGEP DecompGEP2 = DecomposeGEPExpression(V2, DL, &AC, DT);

  // Don't attempt to analyze the decomposed GEP if index scale is not a
  // compile-time constant.
  if (!DecompGEP1.HasCompileTimeConstantScale ||
      !DecompGEP2.HasCompileTimeConstantScale)
    return MayAlias;

  assert(DecompGEP1.Base == UnderlyingV1 && DecompGEP2.Base == UnderlyingV2 &&
         "DecomposeGEPExpression returned a result different from "
         "getUnderlyingObject");

  // If the GEP's offset relative to its base is such that the base would
  // fall below the start of the object underlying V2, then the GEP and V2
  // cannot alias.
  if (isGEPBaseAtNegativeOffset(GEP1, DecompGEP1, DecompGEP2, V2Size))
    return NoAlias;
  // If we have two gep instructions with must-alias or not-alias'ing base
  // pointers, figure out if the indexes to the GEP tell us anything about the
  // derived pointer.
  if (const GEPOperator *GEP2 = dyn_cast<GEPOperator>(V2)) {
    // Check for the GEP base being at a negative offset, this time in the other
    // direction.
    if (isGEPBaseAtNegativeOffset(GEP2, DecompGEP2, DecompGEP1, V1Size))
      return NoAlias;

    // Subtract the GEP2 pointer from the GEP1 pointer to find out their
    // symbolic difference.
    DecompGEP1.Offset -= DecompGEP2.Offset;
    GetIndexDifference(DecompGEP1.VarIndices, DecompGEP2.VarIndices);

    // For GEPs with identical offsets, we can preserve the size and AAInfo
    // when performing the alias check on the underlying objects.
    if (DecompGEP1.Offset == 0 && DecompGEP1.VarIndices.empty())
      return getBestAAResults().alias(
          MemoryLocation(UnderlyingV1, V1Size, V1AAInfo),
          MemoryLocation(UnderlyingV2, V2Size, V2AAInfo), AAQI);

    // Do the base pointers alias?
    AliasResult BaseAlias = getBestAAResults().alias(
        MemoryLocation::getBeforeOrAfter(UnderlyingV1),
        MemoryLocation::getBeforeOrAfter(UnderlyingV2), AAQI);

    // If we get a No or May, then return it immediately, no amount of analysis
    // will improve this situation.
    if (BaseAlias != MustAlias) {
      assert(BaseAlias == NoAlias || BaseAlias == MayAlias);
      return BaseAlias;
    }
  } else {
    // Check to see if these two pointers are related by the getelementptr
    // instruction.  If one pointer is a GEP with a non-zero index of the other
    // pointer, we know they cannot alias.

    // If both accesses are unknown size, we can't do anything useful here.
    if (!V1Size.hasValue() && !V2Size.hasValue())
      return MayAlias;

    AliasResult R = getBestAAResults().alias(
        MemoryLocation::getBeforeOrAfter(UnderlyingV1),
        MemoryLocation(V2, V2Size, V2AAInfo), AAQI);
    if (R != MustAlias) {
      // If V2 may alias GEP base pointer, conservatively returns MayAlias.
      // If V2 is known not to alias GEP base pointer, then the two values
      // cannot alias per GEP semantics: "Any memory access must be done through
      // a pointer value associated with an address range of the memory access,
      // otherwise the behavior is undefined.".
      assert(R == NoAlias || R == MayAlias);
      return R;
    }
  }

  // In the two GEP Case, if there is no difference in the offsets of the
  // computed pointers, the resultant pointers are a must alias.  This
  // happens when we have two lexically identical GEP's (for example).
  //
  // In the other case, if we have getelementptr <ptr>, 0, 0, 0, 0, ... and V2
  // must aliases the GEP, the end result is a must alias also.
  if (DecompGEP1.Offset == 0 && DecompGEP1.VarIndices.empty())
    return MustAlias;

  // If there is a constant difference between the pointers, but the difference
  // is less than the size of the associated memory object, then we know
  // that the objects are partially overlapping.  If the difference is
  // greater, we know they do not overlap.
  if (DecompGEP1.Offset != 0 && DecompGEP1.VarIndices.empty()) {
    if (DecompGEP1.Offset.sge(0)) {
      if (V2Size.hasValue()) {
        if (DecompGEP1.Offset.ult(V2Size.getValue()))
          return PartialAlias;
        return NoAlias;
      }
    } else {
      // We have the situation where:
      // +                +
      // | BaseOffset     |
      // ---------------->|
      // |-->V1Size       |-------> V2Size
      // GEP1             V2
      if (V1Size.hasValue()) {
        if ((-DecompGEP1.Offset).ult(V1Size.getValue()))
          return PartialAlias;
        return NoAlias;
      }
    }
  }

  if (!DecompGEP1.VarIndices.empty()) {
    APInt GCD;
    bool AllNonNegative = DecompGEP1.Offset.isNonNegative();
    bool AllNonPositive = DecompGEP1.Offset.isNonPositive();
    for (unsigned i = 0, e = DecompGEP1.VarIndices.size(); i != e; ++i) {
      const APInt &Scale = DecompGEP1.VarIndices[i].Scale;
      if (i == 0)
        GCD = Scale.abs();
      else
        GCD = APIntOps::GreatestCommonDivisor(GCD, Scale.abs());

      if (AllNonNegative || AllNonPositive) {
        // If the Value could change between cycles, then any reasoning about
        // the Value this cycle may not hold in the next cycle. We'll just
        // give up if we can't determine conditions that hold for every cycle:
        const Value *V = DecompGEP1.VarIndices[i].V;
        const Instruction *CxtI = DecompGEP1.VarIndices[i].CxtI;

        KnownBits Known = computeKnownBits(V, DL, 0, &AC, CxtI, DT);
        bool SignKnownZero = Known.isNonNegative();
        bool SignKnownOne = Known.isNegative();

        // Zero-extension widens the variable, and so forces the sign
        // bit to zero.
        bool IsZExt = DecompGEP1.VarIndices[i].ZExtBits > 0 || isa<ZExtInst>(V);
        SignKnownZero |= IsZExt;
        SignKnownOne &= !IsZExt;

        AllNonNegative &= (SignKnownZero && Scale.isNonNegative()) ||
                          (SignKnownOne && Scale.isNonPositive());
        AllNonPositive &= (SignKnownZero && Scale.isNonPositive()) ||
                          (SignKnownOne && Scale.isNonNegative());
      }
    }

    // We now have accesses at two offsets from the same base:
    //  1. (...)*GCD + DecompGEP1.Offset with size V1Size
    //  2. 0 with size V2Size
    // Using arithmetic modulo GCD, the accesses are at
    // [ModOffset..ModOffset+V1Size) and [0..V2Size). If the first access fits
    // into the range [V2Size..GCD), then we know they cannot overlap.
    APInt ModOffset = DecompGEP1.Offset.srem(GCD);
    if (ModOffset.isNegative())
      ModOffset += GCD; // We want mod, not rem.
    if (V1Size.hasValue() && V2Size.hasValue() &&
        ModOffset.uge(V2Size.getValue()) &&
        (GCD - ModOffset).uge(V1Size.getValue()))
      return NoAlias;

    // If we know all the variables are non-negative, then the total offset is
    // also non-negative and >= DecompGEP1.Offset. We have the following layout:
    // [0, V2Size) ... [TotalOffset, TotalOffer+V1Size]
    // If DecompGEP1.Offset >= V2Size, the accesses don't alias.
    if (AllNonNegative && V2Size.hasValue() &&
        DecompGEP1.Offset.uge(V2Size.getValue()))
      return NoAlias;
    // Similarly, if the variables are non-positive, then the total offset is
    // also non-positive and <= DecompGEP1.Offset. We have the following layout:
    // [TotalOffset, TotalOffset+V1Size) ... [0, V2Size)
    // If -DecompGEP1.Offset >= V1Size, the accesses don't alias.
    if (AllNonPositive && V1Size.hasValue() &&
        (-DecompGEP1.Offset).uge(V1Size.getValue()))
      return NoAlias;

    if (V1Size.hasValue() && V2Size.hasValue()) {
      // Try to determine whether abs(VarIndex) > 0.
      Optional<APInt> MinAbsVarIndex;
      if (DecompGEP1.VarIndices.size() == 1) {
        // VarIndex = Scale*V. If V != 0 then abs(VarIndex) >= abs(Scale).
        const VariableGEPIndex &Var = DecompGEP1.VarIndices[0];
        if (isKnownNonZero(Var.V, DL, 0, &AC, Var.CxtI, DT))
          MinAbsVarIndex = Var.Scale.abs();
      } else if (DecompGEP1.VarIndices.size() == 2) {
        // VarIndex = Scale*V0 + (-Scale)*V1.
        // If V0 != V1 then abs(VarIndex) >= abs(Scale).
        // Check that VisitedPhiBBs is empty, to avoid reasoning about
        // inequality of values across loop iterations.
        const VariableGEPIndex &Var0 = DecompGEP1.VarIndices[0];
        const VariableGEPIndex &Var1 = DecompGEP1.VarIndices[1];
        if (Var0.Scale == -Var1.Scale && Var0.ZExtBits == Var1.ZExtBits &&
            Var0.SExtBits == Var1.SExtBits && VisitedPhiBBs.empty() &&
            isKnownNonEqual(Var0.V, Var1.V, DL, &AC, /* CxtI */ nullptr, DT))
          MinAbsVarIndex = Var0.Scale.abs();
      }

      if (MinAbsVarIndex) {
        // The constant offset will have added at least +/-MinAbsVarIndex to it.
        APInt OffsetLo = DecompGEP1.Offset - *MinAbsVarIndex;
        APInt OffsetHi = DecompGEP1.Offset + *MinAbsVarIndex;
        // Check that an access at OffsetLo or lower, and an access at OffsetHi
        // or higher both do not alias.
        if (OffsetLo.isNegative() && (-OffsetLo).uge(V1Size.getValue()) &&
            OffsetHi.isNonNegative() && OffsetHi.uge(V2Size.getValue()))
          return NoAlias;
      }
    }

    if (constantOffsetHeuristic(DecompGEP1.VarIndices, V1Size, V2Size,
                                DecompGEP1.Offset, &AC, DT))
      return NoAlias;
  }

  // Statically, we can see that the base objects are the same, but the
  // pointers have dynamic offsets which we can't resolve. And none of our
  // little tricks above worked.
  return MayAlias;
}

static AliasResult MergeAliasResults(AliasResult A, AliasResult B) {
  // If the results agree, take it.
  if (A == B)
    return A;
  // A mix of PartialAlias and MustAlias is PartialAlias.
  if ((A == PartialAlias && B == MustAlias) ||
      (B == PartialAlias && A == MustAlias))
    return PartialAlias;
  // Otherwise, we don't know anything.
  return MayAlias;
}

/// Provides a bunch of ad-hoc rules to disambiguate a Select instruction
/// against another.
AliasResult
BasicAAResult::aliasSelect(const SelectInst *SI, LocationSize SISize,
                           const AAMDNodes &SIAAInfo, const Value *V2,
                           LocationSize V2Size, const AAMDNodes &V2AAInfo,
                           AAQueryInfo &AAQI) {
  // If the values are Selects with the same condition, we can do a more precise
  // check: just check for aliases between the values on corresponding arms.
  if (const SelectInst *SI2 = dyn_cast<SelectInst>(V2))
    if (SI->getCondition() == SI2->getCondition()) {
      AliasResult Alias = getBestAAResults().alias(
          MemoryLocation(SI->getTrueValue(), SISize, SIAAInfo),
          MemoryLocation(SI2->getTrueValue(), V2Size, V2AAInfo), AAQI);
      if (Alias == MayAlias)
        return MayAlias;
      AliasResult ThisAlias = getBestAAResults().alias(
          MemoryLocation(SI->getFalseValue(), SISize, SIAAInfo),
          MemoryLocation(SI2->getFalseValue(), V2Size, V2AAInfo), AAQI);
      return MergeAliasResults(ThisAlias, Alias);
    }

  // If both arms of the Select node NoAlias or MustAlias V2, then returns
  // NoAlias / MustAlias. Otherwise, returns MayAlias.
  AliasResult Alias = getBestAAResults().alias(
      MemoryLocation(V2, V2Size, V2AAInfo),
      MemoryLocation(SI->getTrueValue(), SISize, SIAAInfo), AAQI);
  if (Alias == MayAlias)
    return MayAlias;

  AliasResult ThisAlias = getBestAAResults().alias(
      MemoryLocation(V2, V2Size, V2AAInfo),
      MemoryLocation(SI->getFalseValue(), SISize, SIAAInfo), AAQI);
  return MergeAliasResults(ThisAlias, Alias);
}

/// Provide a bunch of ad-hoc rules to disambiguate a PHI instruction against
/// another.
AliasResult BasicAAResult::aliasPHI(const PHINode *PN, LocationSize PNSize,
                                    const AAMDNodes &PNAAInfo, const Value *V2,
                                    LocationSize V2Size,
                                    const AAMDNodes &V2AAInfo,
                                    AAQueryInfo &AAQI) {
  // If the values are PHIs in the same block, we can do a more precise
  // as well as efficient check: just check for aliases between the values
  // on corresponding edges.
  if (const PHINode *PN2 = dyn_cast<PHINode>(V2))
    if (PN2->getParent() == PN->getParent()) {
      Optional<AliasResult> Alias;
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        AliasResult ThisAlias = getBestAAResults().alias(
            MemoryLocation(PN->getIncomingValue(i), PNSize, PNAAInfo),
            MemoryLocation(
                PN2->getIncomingValueForBlock(PN->getIncomingBlock(i)), V2Size,
                V2AAInfo),
            AAQI);
        if (Alias)
          *Alias = MergeAliasResults(*Alias, ThisAlias);
        else
          Alias = ThisAlias;
        if (*Alias == MayAlias)
          break;
      }
      return *Alias;
    }

  SmallVector<Value *, 4> V1Srcs;
  // If a phi operand recurses back to the phi, we can still determine NoAlias
  // if we don't alias the underlying objects of the other phi operands, as we
  // know that the recursive phi needs to be based on them in some way.
  bool isRecursive = false;
  auto CheckForRecPhi = [&](Value *PV) {
    if (!EnableRecPhiAnalysis)
      return false;
    if (getUnderlyingObject(PV) == PN) {
      isRecursive = true;
      return true;
    }
    return false;
  };

  if (PV) {
    // If we have PhiValues then use it to get the underlying phi values.
    const PhiValues::ValueSet &PhiValueSet = PV->getValuesForPhi(PN);
    // If we have more phi values than the search depth then return MayAlias
    // conservatively to avoid compile time explosion. The worst possible case
    // is if both sides are PHI nodes. In which case, this is O(m x n) time
    // where 'm' and 'n' are the number of PHI sources.
    if (PhiValueSet.size() > MaxLookupSearchDepth)
      return MayAlias;
    // Add the values to V1Srcs
    for (Value *PV1 : PhiValueSet) {
      if (CheckForRecPhi(PV1))
        continue;
      V1Srcs.push_back(PV1);
    }
  } else {
    // If we don't have PhiInfo then just look at the operands of the phi itself
    // FIXME: Remove this once we can guarantee that we have PhiInfo always
    SmallPtrSet<Value *, 4> UniqueSrc;
    for (Value *PV1 : PN->incoming_values()) {
      if (isa<PHINode>(PV1))
        // If any of the source itself is a PHI, return MayAlias conservatively
        // to avoid compile time explosion. The worst possible case is if both
        // sides are PHI nodes. In which case, this is O(m x n) time where 'm'
        // and 'n' are the number of PHI sources.
        return MayAlias;

      if (CheckForRecPhi(PV1))
        continue;

      if (UniqueSrc.insert(PV1).second)
        V1Srcs.push_back(PV1);
    }
  }

  // If V1Srcs is empty then that means that the phi has no underlying non-phi
  // value. This should only be possible in blocks unreachable from the entry
  // block, but return MayAlias just in case.
  if (V1Srcs.empty())
    return MayAlias;

  // If this PHI node is recursive, indicate that the pointer may be moved
  // across iterations. We can only prove NoAlias if different underlying
  // objects are involved.
  if (isRecursive)
    PNSize = LocationSize::beforeOrAfterPointer();

  // In the recursive alias queries below, we may compare values from two
  // different loop iterations. Keep track of visited phi blocks, which will
  // be used when determining value equivalence.
  bool BlockInserted = VisitedPhiBBs.insert(PN->getParent()).second;
  auto _ = make_scope_exit([&]() {
    if (BlockInserted)
      VisitedPhiBBs.erase(PN->getParent());
  });

  // If we inserted a block into VisitedPhiBBs, alias analysis results that
  // have been cached earlier may no longer be valid. Perform recursive queries
  // with a new AAQueryInfo.
  AAQueryInfo NewAAQI = AAQI.withEmptyCache();
  AAQueryInfo *UseAAQI = BlockInserted ? &NewAAQI : &AAQI;

  AliasResult Alias = getBestAAResults().alias(
      MemoryLocation(V2, V2Size, V2AAInfo),
      MemoryLocation(V1Srcs[0], PNSize, PNAAInfo), *UseAAQI);

  // Early exit if the check of the first PHI source against V2 is MayAlias.
  // Other results are not possible.
  if (Alias == MayAlias)
    return MayAlias;
  // With recursive phis we cannot guarantee that MustAlias/PartialAlias will
  // remain valid to all elements and needs to conservatively return MayAlias.
  if (isRecursive && Alias != NoAlias)
    return MayAlias;

  // If all sources of the PHI node NoAlias or MustAlias V2, then returns
  // NoAlias / MustAlias. Otherwise, returns MayAlias.
  for (unsigned i = 1, e = V1Srcs.size(); i != e; ++i) {
    Value *V = V1Srcs[i];

    AliasResult ThisAlias = getBestAAResults().alias(
        MemoryLocation(V2, V2Size, V2AAInfo),
        MemoryLocation(V, PNSize, PNAAInfo), *UseAAQI);
    Alias = MergeAliasResults(ThisAlias, Alias);
    if (Alias == MayAlias)
      break;
  }

  return Alias;
}

/// Provides a bunch of ad-hoc rules to disambiguate in common cases, such as
/// array references.
AliasResult BasicAAResult::aliasCheck(const Value *V1, LocationSize V1Size,
                                      const AAMDNodes &V1AAInfo,
                                      const Value *V2, LocationSize V2Size,
                                      const AAMDNodes &V2AAInfo,
                                      AAQueryInfo &AAQI) {
  // If either of the memory references is empty, it doesn't matter what the
  // pointer values are.
  if (V1Size.isZero() || V2Size.isZero())
    return NoAlias;

  // Strip off any casts if they exist.
  V1 = V1->stripPointerCastsAndInvariantGroups();
  V2 = V2->stripPointerCastsAndInvariantGroups();

  // If V1 or V2 is undef, the result is NoAlias because we can always pick a
  // value for undef that aliases nothing in the program.
  if (isa<UndefValue>(V1) || isa<UndefValue>(V2))
    return NoAlias;

  // Are we checking for alias of the same value?
  // Because we look 'through' phi nodes, we could look at "Value" pointers from
  // different iterations. We must therefore make sure that this is not the
  // case. The function isValueEqualInPotentialCycles ensures that this cannot
  // happen by looking at the visited phi nodes and making sure they cannot
  // reach the value.
  if (isValueEqualInPotentialCycles(V1, V2))
    return MustAlias;

  if (!V1->getType()->isPointerTy() || !V2->getType()->isPointerTy())
    return NoAlias; // Scalars cannot alias each other

  // Figure out what objects these things are pointing to if we can.
  const Value *O1 = getUnderlyingObject(V1, MaxLookupSearchDepth);
  const Value *O2 = getUnderlyingObject(V2, MaxLookupSearchDepth);

  // Null values in the default address space don't point to any object, so they
  // don't alias any other pointer.
  if (const ConstantPointerNull *CPN = dyn_cast<ConstantPointerNull>(O1))
    if (!NullPointerIsDefined(&F, CPN->getType()->getAddressSpace()))
      return NoAlias;
  if (const ConstantPointerNull *CPN = dyn_cast<ConstantPointerNull>(O2))
    if (!NullPointerIsDefined(&F, CPN->getType()->getAddressSpace()))
      return NoAlias;

  if (O1 != O2) {
    // If V1/V2 point to two different objects, we know that we have no alias.
    if (isIdentifiedObject(O1) && isIdentifiedObject(O2))
      return NoAlias;

    // Constant pointers can't alias with non-const isIdentifiedObject objects.
    if ((isa<Constant>(O1) && isIdentifiedObject(O2) && !isa<Constant>(O2)) ||
        (isa<Constant>(O2) && isIdentifiedObject(O1) && !isa<Constant>(O1)))
      return NoAlias;

    // Function arguments can't alias with things that are known to be
    // unambigously identified at the function level.
    if ((isa<Argument>(O1) && isIdentifiedFunctionLocal(O2)) ||
        (isa<Argument>(O2) && isIdentifiedFunctionLocal(O1)))
      return NoAlias;

    // If one pointer is the result of a call/invoke or load and the other is a
    // non-escaping local object within the same function, then we know the
    // object couldn't escape to a point where the call could return it.
    //
    // Note that if the pointers are in different functions, there are a
    // variety of complications. A call with a nocapture argument may still
    // temporary store the nocapture argument's value in a temporary memory
    // location if that memory location doesn't escape. Or it may pass a
    // nocapture value to other functions as long as they don't capture it.
    if (isEscapeSource(O1) &&
        isNonEscapingLocalObject(O2, &AAQI.IsCapturedCache))
      return NoAlias;
    if (isEscapeSource(O2) &&
        isNonEscapingLocalObject(O1, &AAQI.IsCapturedCache))
      return NoAlias;
  }

  // If the size of one access is larger than the entire object on the other
  // side, then we know such behavior is undefined and can assume no alias.
  bool NullIsValidLocation = NullPointerIsDefined(&F);
  if ((isObjectSmallerThan(
          O2, getMinimalExtentFrom(*V1, V1Size, DL, NullIsValidLocation), DL,
          TLI, NullIsValidLocation)) ||
      (isObjectSmallerThan(
          O1, getMinimalExtentFrom(*V2, V2Size, DL, NullIsValidLocation), DL,
          TLI, NullIsValidLocation)))
    return NoAlias;

  // If one the accesses may be before the accessed pointer, canonicalize this
  // by using unknown after-pointer sizes for both accesses. This is
  // equivalent, because regardless of which pointer is lower, one of them
  // will always came after the other, as long as the underlying objects aren't
  // disjoint. We do this so that the rest of BasicAA does not have to deal
  // with accesses before the base pointer, and to improve cache utilization by
  // merging equivalent states.
  if (V1Size.mayBeBeforePointer() || V2Size.mayBeBeforePointer()) {
    V1Size = LocationSize::afterPointer();
    V2Size = LocationSize::afterPointer();
  }

  // Check the cache before climbing up use-def chains. This also terminates
  // otherwise infinitely recursive queries.
  AAQueryInfo::LocPair Locs(MemoryLocation(V1, V1Size, V1AAInfo),
                            MemoryLocation(V2, V2Size, V2AAInfo));
  if (V1 > V2)
    std::swap(Locs.first, Locs.second);
  const auto &Pair = AAQI.AliasCache.try_emplace(
      Locs, AAQueryInfo::CacheEntry{NoAlias, 0});
  if (!Pair.second) {
    auto &Entry = Pair.first->second;
    if (!Entry.isDefinitive()) {
      // Remember that we used an assumption.
      ++Entry.NumAssumptionUses;
      ++AAQI.NumAssumptionUses;
    }
    return Entry.Result;
  }

  int OrigNumAssumptionUses = AAQI.NumAssumptionUses;
  unsigned OrigNumAssumptionBasedResults = AAQI.AssumptionBasedResults.size();
  AliasResult Result = aliasCheckRecursive(V1, V1Size, V1AAInfo, V2, V2Size,
                                           V2AAInfo, AAQI, O1, O2);

  auto It = AAQI.AliasCache.find(Locs);
  assert(It != AAQI.AliasCache.end() && "Must be in cache");
  auto &Entry = It->second;

  // Check whether a NoAlias assumption has been used, but disproven.
  bool AssumptionDisproven = Entry.NumAssumptionUses > 0 && Result != NoAlias;
  if (AssumptionDisproven)
    Result = MayAlias;

  // This is a definitive result now, when considered as a root query.
  AAQI.NumAssumptionUses -= Entry.NumAssumptionUses;
  Entry.Result = Result;
  Entry.NumAssumptionUses = -1;

  // If the assumption has been disproven, remove any results that may have
  // been based on this assumption. Do this after the Entry updates above to
  // avoid iterator invalidation.
  if (AssumptionDisproven)
    while (AAQI.AssumptionBasedResults.size() > OrigNumAssumptionBasedResults)
      AAQI.AliasCache.erase(AAQI.AssumptionBasedResults.pop_back_val());

  // The result may still be based on assumptions higher up in the chain.
  // Remember it, so it can be purged from the cache later.
  if (OrigNumAssumptionUses != AAQI.NumAssumptionUses && Result != MayAlias)
    AAQI.AssumptionBasedResults.push_back(Locs);
  return Result;
}

AliasResult BasicAAResult::aliasCheckRecursive(
    const Value *V1, LocationSize V1Size, const AAMDNodes &V1AAInfo,
    const Value *V2, LocationSize V2Size, const AAMDNodes &V2AAInfo,
    AAQueryInfo &AAQI, const Value *O1, const Value *O2) {
  if (const GEPOperator *GV1 = dyn_cast<GEPOperator>(V1)) {
    AliasResult Result =
        aliasGEP(GV1, V1Size, V1AAInfo, V2, V2Size, V2AAInfo, O1, O2, AAQI);
    if (Result != MayAlias)
      return Result;
  } else if (const GEPOperator *GV2 = dyn_cast<GEPOperator>(V2)) {
    AliasResult Result =
        aliasGEP(GV2, V2Size, V2AAInfo, V1, V1Size, V1AAInfo, O2, O1, AAQI);
    if (Result != MayAlias)
      return Result;
  }

  if (const PHINode *PN = dyn_cast<PHINode>(V1)) {
    AliasResult Result =
        aliasPHI(PN, V1Size, V1AAInfo, V2, V2Size, V2AAInfo, AAQI);
    if (Result != MayAlias)
      return Result;
  } else if (const PHINode *PN = dyn_cast<PHINode>(V2)) {
    AliasResult Result =
        aliasPHI(PN, V2Size, V2AAInfo, V1, V1Size, V1AAInfo, AAQI);
    if (Result != MayAlias)
      return Result;
  }

  if (const SelectInst *S1 = dyn_cast<SelectInst>(V1)) {
    AliasResult Result =
        aliasSelect(S1, V1Size, V1AAInfo, V2, V2Size, V2AAInfo, AAQI);
    if (Result != MayAlias)
      return Result;
  } else if (const SelectInst *S2 = dyn_cast<SelectInst>(V2)) {
    AliasResult Result =
        aliasSelect(S2, V2Size, V2AAInfo, V1, V1Size, V1AAInfo, AAQI);
    if (Result != MayAlias)
      return Result;
  }

  // If both pointers are pointing into the same object and one of them
  // accesses the entire object, then the accesses must overlap in some way.
  if (O1 == O2) {
    bool NullIsValidLocation = NullPointerIsDefined(&F);
    if (V1Size.isPrecise() && V2Size.isPrecise() &&
        (isObjectSize(O1, V1Size.getValue(), DL, TLI, NullIsValidLocation) ||
         isObjectSize(O2, V2Size.getValue(), DL, TLI, NullIsValidLocation)))
      return PartialAlias;
  }

  return MayAlias;
}

/// Check whether two Values can be considered equivalent.
///
/// In addition to pointer equivalence of \p V1 and \p V2 this checks whether
/// they can not be part of a cycle in the value graph by looking at all
/// visited phi nodes an making sure that the phis cannot reach the value. We
/// have to do this because we are looking through phi nodes (That is we say
/// noalias(V, phi(VA, VB)) if noalias(V, VA) and noalias(V, VB).
bool BasicAAResult::isValueEqualInPotentialCycles(const Value *V,
                                                  const Value *V2) {
  if (V != V2)
    return false;

  const Instruction *Inst = dyn_cast<Instruction>(V);
  if (!Inst)
    return true;

  if (VisitedPhiBBs.empty())
    return true;

  if (VisitedPhiBBs.size() > MaxNumPhiBBsValueReachabilityCheck)
    return false;

  // Make sure that the visited phis cannot reach the Value. This ensures that
  // the Values cannot come from different iterations of a potential cycle the
  // phi nodes could be involved in.
  for (auto *P : VisitedPhiBBs)
    if (isPotentiallyReachable(&P->front(), Inst, nullptr, DT, LI))
      return false;

  return true;
}

/// Computes the symbolic difference between two de-composed GEPs.
///
/// Dest and Src are the variable indices from two decomposed GetElementPtr
/// instructions GEP1 and GEP2 which have common base pointers.
void BasicAAResult::GetIndexDifference(
    SmallVectorImpl<VariableGEPIndex> &Dest,
    const SmallVectorImpl<VariableGEPIndex> &Src) {
  if (Src.empty())
    return;

  for (unsigned i = 0, e = Src.size(); i != e; ++i) {
    const Value *V = Src[i].V;
    unsigned ZExtBits = Src[i].ZExtBits, SExtBits = Src[i].SExtBits;
    APInt Scale = Src[i].Scale;

    // Find V in Dest.  This is N^2, but pointer indices almost never have more
    // than a few variable indexes.
    for (unsigned j = 0, e = Dest.size(); j != e; ++j) {
      if (!isValueEqualInPotentialCycles(Dest[j].V, V) ||
          Dest[j].ZExtBits != ZExtBits || Dest[j].SExtBits != SExtBits)
        continue;

      // If we found it, subtract off Scale V's from the entry in Dest.  If it
      // goes to zero, remove the entry.
      if (Dest[j].Scale != Scale)
        Dest[j].Scale -= Scale;
      else
        Dest.erase(Dest.begin() + j);
      Scale = 0;
      break;
    }

    // If we didn't consume this entry, add it to the end of the Dest list.
    if (!!Scale) {
      VariableGEPIndex Entry = {V, ZExtBits, SExtBits, -Scale, Src[i].CxtI};
      Dest.push_back(Entry);
    }
  }
}

bool BasicAAResult::constantOffsetHeuristic(
    const SmallVectorImpl<VariableGEPIndex> &VarIndices,
    LocationSize MaybeV1Size, LocationSize MaybeV2Size, const APInt &BaseOffset,
    AssumptionCache *AC, DominatorTree *DT) {
  if (VarIndices.size() != 2 || !MaybeV1Size.hasValue() ||
      !MaybeV2Size.hasValue())
    return false;

  const uint64_t V1Size = MaybeV1Size.getValue();
  const uint64_t V2Size = MaybeV2Size.getValue();

  const VariableGEPIndex &Var0 = VarIndices[0], &Var1 = VarIndices[1];

  if (Var0.ZExtBits != Var1.ZExtBits || Var0.SExtBits != Var1.SExtBits ||
      Var0.Scale != -Var1.Scale)
    return false;

  unsigned Width = Var1.V->getType()->getIntegerBitWidth();

  // We'll strip off the Extensions of Var0 and Var1 and do another round
  // of GetLinearExpression decomposition. In the example above, if Var0
  // is zext(%x + 1) we should get V1 == %x and V1Offset == 1.

  APInt V0Scale(Width, 0), V0Offset(Width, 0), V1Scale(Width, 0),
      V1Offset(Width, 0);
  bool NSW = true, NUW = true;
  unsigned V0ZExtBits = 0, V0SExtBits = 0, V1ZExtBits = 0, V1SExtBits = 0;
  const Value *V0 = GetLinearExpression(Var0.V, V0Scale, V0Offset, V0ZExtBits,
                                        V0SExtBits, DL, 0, AC, DT, NSW, NUW);
  NSW = true;
  NUW = true;
  const Value *V1 = GetLinearExpression(Var1.V, V1Scale, V1Offset, V1ZExtBits,
                                        V1SExtBits, DL, 0, AC, DT, NSW, NUW);

  if (V0Scale != V1Scale || V0ZExtBits != V1ZExtBits ||
      V0SExtBits != V1SExtBits || !isValueEqualInPotentialCycles(V0, V1))
    return false;

  // We have a hit - Var0 and Var1 only differ by a constant offset!

  // If we've been sext'ed then zext'd the maximum difference between Var0 and
  // Var1 is possible to calculate, but we're just interested in the absolute
  // minimum difference between the two. The minimum distance may occur due to
  // wrapping; consider "add i3 %i, 5": if %i == 7 then 7 + 5 mod 8 == 4, and so
  // the minimum distance between %i and %i + 5 is 3.
  APInt MinDiff = V0Offset - V1Offset, Wrapped = -MinDiff;
  MinDiff = APIntOps::umin(MinDiff, Wrapped);
  APInt MinDiffBytes =
    MinDiff.zextOrTrunc(Var0.Scale.getBitWidth()) * Var0.Scale.abs();

  // We can't definitely say whether GEP1 is before or after V2 due to wrapping
  // arithmetic (i.e. for some values of GEP1 and V2 GEP1 < V2, and for other
  // values GEP1 > V2). We'll therefore only declare NoAlias if both V1Size and
  // V2Size can fit in the MinDiffBytes gap.
  return MinDiffBytes.uge(V1Size + BaseOffset.abs()) &&
         MinDiffBytes.uge(V2Size + BaseOffset.abs());
}

//===----------------------------------------------------------------------===//
// BasicAliasAnalysis Pass
//===----------------------------------------------------------------------===//

AnalysisKey BasicAA::Key;

BasicAAResult BasicAA::run(Function &F, FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto *DT = &AM.getResult<DominatorTreeAnalysis>(F);
  auto *LI = AM.getCachedResult<LoopAnalysis>(F);
  auto *PV = AM.getCachedResult<PhiValuesAnalysis>(F);
  return BasicAAResult(F.getParent()->getDataLayout(), F, TLI, AC, DT, LI, PV);
}

BasicAAWrapperPass::BasicAAWrapperPass() : FunctionPass(ID) {
  initializeBasicAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

char BasicAAWrapperPass::ID = 0;

void BasicAAWrapperPass::anchor() {}

INITIALIZE_PASS_BEGIN(BasicAAWrapperPass, "basic-aa",
                      "Basic Alias Analysis (stateless AA impl)", true, true)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PhiValuesWrapperPass)
INITIALIZE_PASS_END(BasicAAWrapperPass, "basic-aa",
                    "Basic Alias Analysis (stateless AA impl)", true, true)

FunctionPass *llvm::createBasicAAWrapperPass() {
  return new BasicAAWrapperPass();
}

bool BasicAAWrapperPass::runOnFunction(Function &F) {
  auto &ACT = getAnalysis<AssumptionCacheTracker>();
  auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
  auto &DTWP = getAnalysis<DominatorTreeWrapperPass>();
  auto *LIWP = getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *PVWP = getAnalysisIfAvailable<PhiValuesWrapperPass>();

  Result.reset(new BasicAAResult(F.getParent()->getDataLayout(), F,
                                 TLIWP.getTLI(F), ACT.getAssumptionCache(F),
                                 &DTWP.getDomTree(),
                                 LIWP ? &LIWP->getLoopInfo() : nullptr,
                                 PVWP ? &PVWP->getResult() : nullptr));

  return false;
}

void BasicAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<AssumptionCacheTracker>();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<TargetLibraryInfoWrapperPass>();
  AU.addUsedIfAvailable<PhiValuesWrapperPass>();
}

BasicAAResult llvm::createLegacyPMBasicAAResult(Pass &P, Function &F) {
  return BasicAAResult(
      F.getParent()->getDataLayout(), F,
      P.getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F),
      P.getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F));
}
