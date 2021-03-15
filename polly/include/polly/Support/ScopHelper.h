//===------ Support/ScopHelper.h -- Some Helper Functions for Scop. -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Small functions that help with LLVM-IR.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SUPPORT_IRHELPER_H
#define POLLY_SUPPORT_IRHELPER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueHandle.h"
#include "isl/isl-noexceptions.h"

namespace llvm {
class LoopInfo;
class Loop;
class ScalarEvolution;
class SCEV;
class Region;
class Pass;
class DominatorTree;
class RegionInfo;
class RegionNode;
} // namespace llvm

namespace polly {
class Scop;
class ScopStmt;

/// Enumeration of assumptions Polly can take.
enum AssumptionKind {
  ALIASING,
  INBOUNDS,
  WRAPPING,
  UNSIGNED,
  PROFITABLE,
  ERRORBLOCK,
  COMPLEXITY,
  INFINITELOOP,
  INVARIANTLOAD,
  DELINEARIZATION,
};

/// Enum to distinguish between assumptions and restrictions.
enum AssumptionSign { AS_ASSUMPTION, AS_RESTRICTION };

/// Helper struct to remember assumptions.
struct Assumption {
  /// The kind of the assumption (e.g., WRAPPING).
  AssumptionKind Kind;

  /// Flag to distinguish assumptions and restrictions.
  AssumptionSign Sign;

  /// The valid/invalid context if this is an assumption/restriction.
  isl::set Set;

  /// The location that caused this assumption.
  llvm::DebugLoc Loc;

  /// An optional block whose domain can simplify the assumption.
  llvm::BasicBlock *BB;

  // Whether the assumption must be checked at runtime.
  bool RequiresRTC;
};

using RecordedAssumptionsTy = llvm::SmallVector<Assumption, 8>;

/// Record an assumption for later addition to the assumed context.
///
/// This function will add the assumption to the RecordedAssumptions. This
/// collection will be added (@see addAssumption) to the assumed context once
/// all paramaters are known and the context is fully built.
///
/// @param RecordedAssumption container which keeps all recorded assumptions.
/// @param Kind The assumption kind describing the underlying cause.
/// @param Set  The relations between parameters that are assumed to hold.
/// @param Loc  The location in the source that caused this assumption.
/// @param Sign Enum to indicate if the assumptions in @p Set are positive
///             (needed/assumptions) or negative (invalid/restrictions).
/// @param BB   The block in which this assumption was taken. If it is
///             set, the domain of that block will be used to simplify the
///             actual assumption in @p Set once it is added. This is useful
///             if the assumption was created prior to the domain.
/// @param RTC  Does the assumption require a runtime check?
void recordAssumption(RecordedAssumptionsTy *RecordedAssumptions,
                      AssumptionKind Kind, isl::set Set, llvm::DebugLoc Loc,
                      AssumptionSign Sign, llvm::BasicBlock *BB = nullptr,
                      bool RTC = true);

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
    if (isMemTransferInst())
      return std::min(asMemTransferInst()->getDestAlignment(),
                      asMemTransferInst()->getSourceAlignment());
    if (isMemIntrinsic())
      return asMemIntrinsic()->getDestAlignment();
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

/// Split the entry block of a function to store the newly inserted
///        allocations outside of all Scops.
///
/// @param DT DominatorTree to be updated.
/// @param LI LoopInfo to be updated.
/// @param RI RegionInfo to be updated.
void splitEntryBlockForAlloca(llvm::BasicBlock *EntryBlock,
                              llvm::DominatorTree *DT, llvm::LoopInfo *LI,
                              llvm::RegionInfo *RI);

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
llvm::Value *getConditionFromTerminator(llvm::Instruction *TI);

/// Get the smallest loop that contains @p S but is not in @p S.
llvm::Loop *getLoopSurroundingScop(Scop &S, llvm::LoopInfo &LI);

/// Get the number of blocks in @p L.
///
/// The number of blocks in a loop are the number of basic blocks actually
/// belonging to the loop, as well as all single basic blocks that the loop
/// exits to and which terminate in an unreachable instruction. We do not
/// allow such basic blocks in the exit of a scop, hence they belong to the
/// scop and represent run-time conditions which we want to model and
/// subsequently speculate away.
///
/// @see getRegionNodeLoop for additional details.
unsigned getNumBlocksInLoop(llvm::Loop *L);

/// Get the number of blocks in @p RN.
unsigned getNumBlocksInRegionNode(llvm::RegionNode *RN);

/// Return the smallest loop surrounding @p RN.
llvm::Loop *getRegionNodeLoop(llvm::RegionNode *RN, llvm::LoopInfo &LI);

/// Check if @p LInst can be hoisted in @p R.
///
/// @param LInst The load to check.
/// @param R     The analyzed region.
/// @param LI    The loop info.
/// @param SE    The scalar evolution analysis.
/// @param DT    The dominator tree of the function.
/// @param KnownInvariantLoads The invariant load set.
///
/// @return True if @p LInst can be hoisted in @p R.
bool isHoistableLoad(llvm::LoadInst *LInst, llvm::Region &R, llvm::LoopInfo &LI,
                     llvm::ScalarEvolution &SE, const llvm::DominatorTree &DT,
                     const InvariantLoadsSetTy &KnownInvariantLoads);

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
llvm::BasicBlock *getUseBlock(const llvm::Use &U);

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

/// Is the given instruction a call to a debug function?
///
/// A debug function can be used to insert output in Polly-optimized code which
/// normally does not allow function calls with side-effects. For instance, a
/// printf can be inserted to check whether a value still has the expected value
/// after Polly generated code:
///
///     int sum = 0;
///     for (int i = 0; i < 16; i+=1) {
///       sum += i;
///       printf("The value of sum at i=%d is %d\n", sum, i);
///     }
bool isDebugCall(llvm::Instruction *Inst);

/// Does the statement contain a call to a debug function?
///
/// Such a statement must not be removed, even if has no side-effects.
bool hasDebugCall(ScopStmt *Stmt);

/// Find a property value in a LoopID.
///
/// Generally, a property MDNode has the format
///
///   !{ !"Name", value }
///
/// In which case the value is returned.
///
/// If the property is just
///
///   !{ !"Name" }
///
/// Then `nullptr` is set to mark the property is existing, but does not carry
/// any value. If the property does not exist, `None` is returned.
llvm::Optional<llvm::Metadata *> findMetadataOperand(llvm::MDNode *LoopMD,
                                                     llvm::StringRef Name);

/// Does the loop's LoopID contain a 'llvm.loop.disable_heuristics' property?
///
/// This is equivalent to llvm::hasDisableAllTransformsHint(Loop*), but
/// including the LoopUtils.h header indirectly also declares llvm::MemoryAccess
/// which clashes with polly::MemoryAccess. Declaring this alias here avoid
/// having to include LoopUtils.h in other files.
bool hasDisableAllTransformsHint(llvm::Loop *L);

/// Represent the attributes of a loop.
struct BandAttr {
  /// LoopID which stores the properties of the loop, such as transformations to
  /// apply and the metadata of followup-loops.
  ///
  /// Cannot be used to identify a loop. Two different loops can have the same
  /// metadata.
  llvm::MDNode *Metadata = nullptr;

  /// The LoopInfo reference for this loop.
  ///
  /// Only loops from the original IR are represented by LoopInfo. Loops that
  /// were generated by Polly are not tracked by LoopInfo.
  llvm::Loop *OriginalLoop = nullptr;
};

/// Get an isl::id representing a loop.
///
/// This takes the ownership of the BandAttr and will be free'd when the
/// returned isl::Id is free'd.
isl::id getIslLoopAttr(isl::ctx Ctx, BandAttr *Attr);

/// Create an isl::id that identifies an original loop.
///
/// Return nullptr if the loop does not need a BandAttr (i.e. has no
/// properties);
///
/// This creates a BandAttr which must be unique per loop and therefore this
/// must not be called multiple times on the same loop as their id would be
/// different.
isl::id createIslLoopAttr(isl::ctx Ctx, llvm::Loop *L);

/// Is @p Id representing a loop?
///
/// Such ids contain a polly::BandAttr as its user pointer.
bool isLoopAttr(const isl::id &Id);

/// Return the BandAttr of a loop's isl::id.
BandAttr *getLoopAttr(const isl::id &Id);

} // namespace polly
#endif
