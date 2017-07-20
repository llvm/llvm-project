//===------ Support/ScopHelper.h -- Some Helper Functions for Scop. -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with LLVM-IR.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_IRHELPER_H
#define POLLY_SUPPORT_IRHELPER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueHandle.h"
#include <tuple>
#include <vector>

namespace llvm {
class LoopInfo;
class Loop;
class ScalarEvolution;
class SCEV;
class Region;
class Pass;
class DominatorTree;
class RegionInfo;
class GetElementPtrInst;
} // namespace llvm

namespace polly {
class Scop;

/// Type to remap values.
using ValueMapT = llvm::DenseMap<llvm::AssertingVH<llvm::Value>,
                                 llvm::AssertingVH<llvm::Value>>;

/// Type for a set of invariant loads.
using InvariantLoadsSetTy = llvm::SetVector<llvm::AssertingVH<llvm::LoadInst>>;

/// Set type for parameters.
using ParameterSetTy = llvm::SetVector<const llvm::SCEV *>;

/// Set of loops (used to remember loops in non-affine subregions).
using BoxedLoopsSetTy = llvm::SetVector<const llvm::Loop *>;

/// Utility proxy to wrap the common members of LoadInst and StoreInst.
///
/// This works like the LLVM utility class CallSite, ie. it forwards all calls
/// to either a LoadInst, StoreInst, MemIntrinsic or MemTransferInst.
/// It is similar to LLVM's utility classes IntrinsicInst, MemIntrinsic,
/// MemTransferInst, etc. in that it offers a common interface, but does not act
/// as a fake base class.
/// It is similar to StringRef and ArrayRef in that it holds a pointer to the
/// referenced object and should be passed by-value as it is small enough.
///
/// This proxy can either represent a LoadInst instance, a StoreInst instance,
/// a MemIntrinsic instance (memset, memmove, memcpy), a CallInst instance or a
/// nullptr (only creatable using the default constructor); never an Instruction
/// that is neither of the above mentioned. When representing a nullptr, only
/// the following methods are defined:
/// isNull(), isInstruction(), isLoad(), isStore(), ..., isMemTransferInst(),
/// operator bool(), operator!()
///
/// The functions isa, cast, cast_or_null, dyn_cast are modeled te resemble
/// those from llvm/Support/Casting.h. Partial template function specialization
/// is currently not supported in C++ such that those cannot be used directly.
/// (llvm::isa could, but then llvm:cast etc. would not have the expected
/// behavior)
class MemAccInst {
private:
  llvm::Instruction *I;

public:
  MemAccInst() : I(nullptr) {}
  MemAccInst(const MemAccInst &Inst) : I(Inst.I) {}
  /* implicit */ MemAccInst(llvm::LoadInst &LI) : I(&LI) {}
  /* implicit */ MemAccInst(llvm::LoadInst *LI) : I(LI) {}
  /* implicit */ MemAccInst(llvm::StoreInst &SI) : I(&SI) {}
  /* implicit */ MemAccInst(llvm::StoreInst *SI) : I(SI) {}
  /* implicit */ MemAccInst(llvm::MemIntrinsic *MI) : I(MI) {}
  /* implicit */ MemAccInst(llvm::CallInst *CI) : I(CI) {}
  explicit MemAccInst(llvm::Instruction &I) : I(&I) { assert(isa(I)); }
  explicit MemAccInst(llvm::Instruction *I) : I(I) { assert(isa(I)); }

  static bool isa(const llvm::Value &V) {
    return llvm::isa<llvm::LoadInst>(V) || llvm::isa<llvm::StoreInst>(V) ||
           llvm::isa<llvm::CallInst>(V) || llvm::isa<llvm::MemIntrinsic>(V);
  }
  static bool isa(const llvm::Value *V) {
    return llvm::isa<llvm::LoadInst>(V) || llvm::isa<llvm::StoreInst>(V) ||
           llvm::isa<llvm::CallInst>(V) || llvm::isa<llvm::MemIntrinsic>(V);
  }
  static MemAccInst cast(llvm::Value &V) {
    return MemAccInst(llvm::cast<llvm::Instruction>(V));
  }
  static MemAccInst cast(llvm::Value *V) {
    return MemAccInst(llvm::cast<llvm::Instruction>(V));
  }
  static MemAccInst cast_or_null(llvm::Value &V) {
    return MemAccInst(llvm::cast<llvm::Instruction>(V));
  }
  static MemAccInst cast_or_null(llvm::Value *V) {
    if (!V)
      return MemAccInst();
    return MemAccInst(llvm::cast<llvm::Instruction>(V));
  }
  static MemAccInst dyn_cast(llvm::Value &V) {
    if (isa(V))
      return MemAccInst(llvm::cast<llvm::Instruction>(V));
    return MemAccInst();
  }
  static MemAccInst dyn_cast(llvm::Value *V) {
    assert(V);
    if (isa(V))
      return MemAccInst(llvm::cast<llvm::Instruction>(V));
    return MemAccInst();
  }

  MemAccInst &operator=(const MemAccInst &Inst) {
    I = Inst.I;
    return *this;
  }
  MemAccInst &operator=(llvm::LoadInst &LI) {
    I = &LI;
    return *this;
  }
  MemAccInst &operator=(llvm::LoadInst *LI) {
    I = LI;
    return *this;
  }
  MemAccInst &operator=(llvm::StoreInst &SI) {
    I = &SI;
    return *this;
  }
  MemAccInst &operator=(llvm::StoreInst *SI) {
    I = SI;
    return *this;
  }
  MemAccInst &operator=(llvm::MemIntrinsic &MI) {
    I = &MI;
    return *this;
  }
  MemAccInst &operator=(llvm::MemIntrinsic *MI) {
    I = MI;
    return *this;
  }
  MemAccInst &operator=(llvm::CallInst &CI) {
    I = &CI;
    return *this;
  }
  MemAccInst &operator=(llvm::CallInst *CI) {
    I = CI;
    return *this;
  }

  llvm::Instruction *get() const {
    assert(I && "Unexpected nullptr!");
    return I;
  }
  operator llvm::Instruction *() const { return asInstruction(); }
  llvm::Instruction *operator->() const { return get(); }

  explicit operator bool() const { return isInstruction(); }
  bool operator!() const { return isNull(); }

  llvm::Value *getValueOperand() const {
    if (isLoad())
      return asLoad();
    if (isStore())
      return asStore()->getValueOperand();
    if (isMemIntrinsic())
      return nullptr;
    if (isCallInst())
      return nullptr;
    llvm_unreachable("Operation not supported on nullptr");
  }
  llvm::Value *getPointerOperand() const {
    if (isLoad())
      return asLoad()->getPointerOperand();
    if (isStore())
      return asStore()->getPointerOperand();
    if (isMemIntrinsic())
      return asMemIntrinsic()->getRawDest();
    if (isCallInst())
      return nullptr;
    llvm_unreachable("Operation not supported on nullptr");
  }

  unsigned getAlignment() const {
    if (isLoad())
      return asLoad()->getAlignment();
    if (isStore())
      return asStore()->getAlignment();
    if (isMemIntrinsic())
      return asMemIntrinsic()->getAlignment();
    if (isCallInst())
      return 0;
    llvm_unreachable("Operation not supported on nullptr");
  }
  bool isVolatile() const {
    if (isLoad())
      return asLoad()->isVolatile();
    if (isStore())
      return asStore()->isVolatile();
    if (isMemIntrinsic())
      return asMemIntrinsic()->isVolatile();
    if (isCallInst())
      return false;
    llvm_unreachable("Operation not supported on nullptr");
  }
  bool isSimple() const {
    if (isLoad())
      return asLoad()->isSimple();
    if (isStore())
      return asStore()->isSimple();
    if (isMemIntrinsic())
      return !asMemIntrinsic()->isVolatile();
    if (isCallInst())
      return true;
    llvm_unreachable("Operation not supported on nullptr");
  }
  llvm::AtomicOrdering getOrdering() const {
    if (isLoad())
      return asLoad()->getOrdering();
    if (isStore())
      return asStore()->getOrdering();
    if (isMemIntrinsic())
      return llvm::AtomicOrdering::NotAtomic;
    if (isCallInst())
      return llvm::AtomicOrdering::NotAtomic;
    llvm_unreachable("Operation not supported on nullptr");
  }
  bool isUnordered() const {
    if (isLoad())
      return asLoad()->isUnordered();
    if (isStore())
      return asStore()->isUnordered();
    // Copied from the Load/Store implementation of isUnordered:
    if (isMemIntrinsic())
      return !asMemIntrinsic()->isVolatile();
    if (isCallInst())
      return true;
    llvm_unreachable("Operation not supported on nullptr");
  }

  bool isNull() const { return !I; }
  bool isInstruction() const { return I; }

  llvm::Instruction *asInstruction() const { return I; }

private:
  bool isLoad() const { return I && llvm::isa<llvm::LoadInst>(I); }
  bool isStore() const { return I && llvm::isa<llvm::StoreInst>(I); }
  bool isCallInst() const { return I && llvm::isa<llvm::CallInst>(I); }
  bool isMemIntrinsic() const { return I && llvm::isa<llvm::MemIntrinsic>(I); }
  bool isMemSetInst() const { return I && llvm::isa<llvm::MemSetInst>(I); }
  bool isMemTransferInst() const {
    return I && llvm::isa<llvm::MemTransferInst>(I);
  }

  llvm::LoadInst *asLoad() const { return llvm::cast<llvm::LoadInst>(I); }
  llvm::StoreInst *asStore() const { return llvm::cast<llvm::StoreInst>(I); }
  llvm::CallInst *asCallInst() const { return llvm::cast<llvm::CallInst>(I); }
  llvm::MemIntrinsic *asMemIntrinsic() const {
    return llvm::cast<llvm::MemIntrinsic>(I);
  }
  llvm::MemSetInst *asMemSetInst() const {
    return llvm::cast<llvm::MemSetInst>(I);
  }
  llvm::MemTransferInst *asMemTransferInst() const {
    return llvm::cast<llvm::MemTransferInst>(I);
  }
};
} // namespace polly

namespace llvm {
/// Specialize simplify_type for MemAccInst to enable dyn_cast and cast
///        from a MemAccInst object.
template <> struct simplify_type<polly::MemAccInst> {
  typedef Instruction *SimpleType;
  static SimpleType getSimplifiedValue(polly::MemAccInst &I) {
    return I.asInstruction();
  }
};
} // namespace llvm

namespace polly {

/// Simplify the region to have a single unconditional entry edge and a
/// single exit edge.
///
/// Although this function allows DT and RI to be null, regions only work
/// properly if the DominatorTree (for Region::contains) and RegionInfo are kept
/// up-to-date.
///
/// @param R  The region to be simplified
/// @param DT DominatorTree to be updated.
/// @param LI LoopInfo to be updated.
/// @param RI RegionInfo to be updated.
void simplifyRegion(llvm::Region *R, llvm::DominatorTree *DT,
                    llvm::LoopInfo *LI, llvm::RegionInfo *RI);

/// Split the entry block of a function to store the newly inserted
///        allocations outside of all Scops.
///
/// @param EntryBlock The entry block of the current function.
/// @param P          The pass that currently running.
///
void splitEntryBlockForAlloca(llvm::BasicBlock *EntryBlock, llvm::Pass *P);

/// Wrapper for SCEVExpander extended to all Polly features.
///
/// This wrapper will internally call the SCEVExpander but also makes sure that
/// all additional features not represented in SCEV (e.g., SDiv/SRem are not
/// black boxes but can be part of the function) will be expanded correctly.
///
/// The parameters are the same as for the creation of a SCEVExpander as well
/// as the call to SCEVExpander::expandCodeFor:
///
/// @param S     The current Scop.
/// @param SE    The Scalar Evolution pass.
/// @param DL    The module data layout.
/// @param Name  The suffix added to the new instruction names.
/// @param E     The expression for which code is actually generated.
/// @param Ty    The type of the resulting code.
/// @param IP    The insertion point for the new code.
/// @param VMap  A remapping of values used in @p E.
/// @param RTCBB The last block of the RTC. Used to insert loop-invariant
///              instructions in rare cases.
llvm::Value *expandCodeFor(Scop &S, llvm::ScalarEvolution &SE,
                           const llvm::DataLayout &DL, const char *Name,
                           const llvm::SCEV *E, llvm::Type *Ty,
                           llvm::Instruction *IP, ValueMapT *VMap,
                           llvm::BasicBlock *RTCBB);

/// Check if the block is a error block.
///
/// A error block is currently any block that fulfills at least one of
/// the following conditions:
///
///  - It is terminated by an unreachable instruction
///  - It contains a call to a non-pure function that is not immediately
///    dominated by a loop header and that does not dominate the region exit.
///    This is a heuristic to pick only error blocks that are conditionally
///    executed and can be assumed to be not executed at all without the domains
///    being available.
///
/// @param BB The block to check.
/// @param R  The analyzed region.
/// @param LI The loop info analysis.
/// @param DT The dominator tree of the function.
///
/// @return True if the block is a error block, false otherwise.
bool isErrorBlock(llvm::BasicBlock &BB, const llvm::Region &R,
                  llvm::LoopInfo &LI, const llvm::DominatorTree &DT);

/// Return the condition for the terminator @p TI.
///
/// For unconditional branches the "i1 true" condition will be returned.
///
/// @param TI The terminator to get the condition from.
///
/// @return The condition of @p TI and nullptr if none could be extracted.
llvm::Value *getConditionFromTerminator(llvm::TerminatorInst *TI);

/// Check if @p LInst can be hoisted in @p R.
///
/// @param LInst The load to check.
/// @param R     The analyzed region.
/// @param LI    The loop info.
/// @param SE    The scalar evolution analysis.
/// @param DT    The dominator tree of the function.
///
/// @return True if @p LInst can be hoisted in @p R.
bool isHoistableLoad(llvm::LoadInst *LInst, llvm::Region &R, llvm::LoopInfo &LI,
                     llvm::ScalarEvolution &SE, const llvm::DominatorTree &DT);

/// Return true iff @p V is an intrinsic that we ignore during code
///        generation.
bool isIgnoredIntrinsic(const llvm::Value *V);

/// Check whether a value an be synthesized by the code generator.
///
/// Some value will be recalculated only from information that is code generated
/// from the polyhedral representation. For such instructions we do not need to
/// ensure that their operands are available during code generation.
///
/// @param V The value to check.
/// @param S The current SCoP.
/// @param SE The scalar evolution database.
/// @param Scope Location where the value would by synthesized.
/// @return If the instruction I can be regenerated from its
///         scalar evolution representation, return true,
///         otherwise return false.
bool canSynthesize(const llvm::Value *V, const Scop &S,
                   llvm::ScalarEvolution *SE, llvm::Loop *Scope);

/// Return the block in which a value is used.
///
/// For normal instructions, this is the instruction's parent block. For PHI
/// nodes, this is the incoming block of that use, because this is where the
/// operand must be defined (i.e. its definition dominates this block).
/// Non-instructions do not use operands at a specific point such that in this
/// case this function returns nullptr.
llvm::BasicBlock *getUseBlock(llvm::Use &U);

/// Derive the individual index expressions from a GEP instruction.
///
/// This function optimistically assumes the GEP references into a fixed size
/// array. If this is actually true, this function returns a list of array
/// subscript expressions as SCEV as well as a list of integers describing
/// the size of the individual array dimensions. Both lists have either equal
/// length or the size list is one element shorter in case there is no known
/// size available for the outermost array dimension.
///
/// @param GEP The GetElementPtr instruction to analyze.
///
/// @return A tuple with the subscript expressions and the dimension sizes.
std::tuple<std::vector<const llvm::SCEV *>, std::vector<int>>
getIndexExpressionsFromGEP(llvm::GetElementPtrInst *GEP,
                           llvm::ScalarEvolution &SE);

// If the loop is nonaffine/boxed, return the first non-boxed surrounding loop
// for Polly. If the loop is affine, return the loop itself.
//
// @param L             Pointer to the Loop object to analyze.
// @param LI            Reference to the LoopInfo.
// @param BoxedLoops    Set of Boxed Loops we get from the SCoP.
llvm::Loop *getFirstNonBoxedLoopFor(llvm::Loop *L, llvm::LoopInfo &LI,
                                    const BoxedLoopsSetTy &BoxedLoops);

// If the Basic Block belongs to a loop that is nonaffine/boxed, return the
// first non-boxed surrounding loop for Polly. If the loop is affine, return
// the loop itself.
//
// @param BB            Pointer to the Basic Block to analyze.
// @param LI            Reference to the LoopInfo.
// @param BoxedLoops    Set of Boxed Loops we get from the SCoP.
llvm::Loop *getFirstNonBoxedLoopFor(llvm::BasicBlock *BB, llvm::LoopInfo &LI,
                                    const BoxedLoopsSetTy &BoxedLoops);
} // namespace polly
#endif
