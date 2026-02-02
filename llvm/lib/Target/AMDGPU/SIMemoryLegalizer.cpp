//===- SIMemoryLegalizer.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Memory legalizer - implements memory model. More information can be
/// found here:
///   http://llvm.org/docs/AMDGPUUsage.html#memory-model
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUMachineModuleInfo.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;
using namespace llvm::AMDGPU;

#define DEBUG_TYPE "si-memory-legalizer"
#define PASS_NAME "SI Memory Legalizer"

static cl::opt<bool> AmdgcnSkipCacheInvalidations(
    "amdgcn-skip-cache-invalidations", cl::init(false), cl::Hidden,
    cl::desc("Use this to skip inserting cache invalidating instructions."));

namespace {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

/// Memory operation flags. Can be ORed together.
enum class SIMemOp {
  NONE = 0u,
  LOAD = 1u << 0,
  STORE = 1u << 1,
  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ STORE)
};

/// Position to insert a new instruction relative to an existing
/// instruction.
enum class Position {
  BEFORE,
  AFTER
};

/// The atomic synchronization scopes supported by the AMDGPU target.
enum class SIAtomicScope {
  NONE,
  SINGLETHREAD,
  WAVEFRONT,
  WORKGROUP,
  CLUSTER, // Promoted to AGENT on targets without workgroup clusters.
  AGENT,
  SYSTEM
};

/// The distinct address spaces supported by the AMDGPU target for
/// atomic memory operation. Can be ORed together.
enum class SIAtomicAddrSpace {
  NONE = 0u,
  GLOBAL = 1u << 0,
  LDS = 1u << 1,
  SCRATCH = 1u << 2,
  GDS = 1u << 3,
  OTHER = 1u << 4,

  /// The address spaces that can be accessed by a FLAT instruction.
  FLAT = GLOBAL | LDS | SCRATCH,

  /// The address spaces that support atomic instructions.
  ATOMIC = GLOBAL | LDS | SCRATCH | GDS,

  /// All address spaces.
  ALL = GLOBAL | LDS | SCRATCH | GDS | OTHER,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ ALL)
};

class SIMemOpInfo final {
private:

  friend class SIMemOpAccess;

  AtomicOrdering Ordering = AtomicOrdering::NotAtomic;
  AtomicOrdering FailureOrdering = AtomicOrdering::NotAtomic;
  SIAtomicScope Scope = SIAtomicScope::SYSTEM;
  SIAtomicAddrSpace OrderingAddrSpace = SIAtomicAddrSpace::NONE;
  SIAtomicAddrSpace InstrAddrSpace = SIAtomicAddrSpace::NONE;
  bool IsCrossAddressSpaceOrdering = false;
  bool IsVolatile = false;
  bool IsNonTemporal = false;
  bool IsLastUse = false;
  bool IsCooperative = false;

  // TODO: Should we assume Cooperative=true if no MMO is present?
  SIMemOpInfo(
      const GCNSubtarget &ST,
      AtomicOrdering Ordering = AtomicOrdering::SequentiallyConsistent,
      SIAtomicScope Scope = SIAtomicScope::SYSTEM,
      SIAtomicAddrSpace OrderingAddrSpace = SIAtomicAddrSpace::ATOMIC,
      SIAtomicAddrSpace InstrAddrSpace = SIAtomicAddrSpace::ALL,
      bool IsCrossAddressSpaceOrdering = true,
      AtomicOrdering FailureOrdering = AtomicOrdering::SequentiallyConsistent,
      bool IsVolatile = false, bool IsNonTemporal = false,
      bool IsLastUse = false, bool IsCooperative = false)
      : Ordering(Ordering), FailureOrdering(FailureOrdering), Scope(Scope),
        OrderingAddrSpace(OrderingAddrSpace), InstrAddrSpace(InstrAddrSpace),
        IsCrossAddressSpaceOrdering(IsCrossAddressSpaceOrdering),
        IsVolatile(IsVolatile), IsNonTemporal(IsNonTemporal),
        IsLastUse(IsLastUse), IsCooperative(IsCooperative) {

    if (Ordering == AtomicOrdering::NotAtomic) {
      assert(!IsCooperative && "Cannot be cooperative & non-atomic!");
      assert(Scope == SIAtomicScope::NONE &&
             OrderingAddrSpace == SIAtomicAddrSpace::NONE &&
             !IsCrossAddressSpaceOrdering &&
             FailureOrdering == AtomicOrdering::NotAtomic);
      return;
    }

    assert(Scope != SIAtomicScope::NONE &&
           (OrderingAddrSpace & SIAtomicAddrSpace::ATOMIC) !=
               SIAtomicAddrSpace::NONE &&
           (InstrAddrSpace & SIAtomicAddrSpace::ATOMIC) !=
               SIAtomicAddrSpace::NONE);

    // There is also no cross address space ordering if the ordering
    // address space is the same as the instruction address space and
    // only contains a single address space.
    if ((OrderingAddrSpace == InstrAddrSpace) &&
        isPowerOf2_32(uint32_t(InstrAddrSpace)))
      this->IsCrossAddressSpaceOrdering = false;

    // Limit the scope to the maximum supported by the instruction's address
    // spaces.
    if ((InstrAddrSpace & ~SIAtomicAddrSpace::SCRATCH) ==
        SIAtomicAddrSpace::NONE) {
      this->Scope = std::min(Scope, SIAtomicScope::SINGLETHREAD);
    } else if ((InstrAddrSpace &
                ~(SIAtomicAddrSpace::SCRATCH | SIAtomicAddrSpace::LDS)) ==
               SIAtomicAddrSpace::NONE) {
      this->Scope = std::min(Scope, SIAtomicScope::WORKGROUP);
    } else if ((InstrAddrSpace &
                ~(SIAtomicAddrSpace::SCRATCH | SIAtomicAddrSpace::LDS |
                  SIAtomicAddrSpace::GDS)) == SIAtomicAddrSpace::NONE) {
      this->Scope = std::min(Scope, SIAtomicScope::AGENT);
    }

    // On targets that have no concept of a workgroup cluster, use
    // AGENT scope as a conservatively correct alternative.
    if (this->Scope == SIAtomicScope::CLUSTER && !ST.hasClusters())
      this->Scope = SIAtomicScope::AGENT;
  }

public:
  /// \returns Atomic synchronization scope of the machine instruction used to
  /// create this SIMemOpInfo.
  SIAtomicScope getScope() const {
    return Scope;
  }

  /// \returns Ordering constraint of the machine instruction used to
  /// create this SIMemOpInfo.
  AtomicOrdering getOrdering() const {
    return Ordering;
  }

  /// \returns Failure ordering constraint of the machine instruction used to
  /// create this SIMemOpInfo.
  AtomicOrdering getFailureOrdering() const {
    return FailureOrdering;
  }

  /// \returns The address spaces be accessed by the machine
  /// instruction used to create this SIMemOpInfo.
  SIAtomicAddrSpace getInstrAddrSpace() const {
    return InstrAddrSpace;
  }

  /// \returns The address spaces that must be ordered by the machine
  /// instruction used to create this SIMemOpInfo.
  SIAtomicAddrSpace getOrderingAddrSpace() const {
    return OrderingAddrSpace;
  }

  /// \returns Return true iff memory ordering of operations on
  /// different address spaces is required.
  bool getIsCrossAddressSpaceOrdering() const {
    return IsCrossAddressSpaceOrdering;
  }

  /// \returns True if memory access of the machine instruction used to
  /// create this SIMemOpInfo is volatile, false otherwise.
  bool isVolatile() const {
    return IsVolatile;
  }

  /// \returns True if memory access of the machine instruction used to
  /// create this SIMemOpInfo is nontemporal, false otherwise.
  bool isNonTemporal() const {
    return IsNonTemporal;
  }

  /// \returns True if memory access of the machine instruction used to
  /// create this SIMemOpInfo is last use, false otherwise.
  bool isLastUse() const { return IsLastUse; }

  /// \returns True if this is a cooperative load or store atomic.
  bool isCooperative() const { return IsCooperative; }

  /// \returns True if ordering constraint of the machine instruction used to
  /// create this SIMemOpInfo is unordered or higher, false otherwise.
  bool isAtomic() const {
    return Ordering != AtomicOrdering::NotAtomic;
  }

};

class SIMemOpAccess final {
private:
  const AMDGPUMachineModuleInfo *MMI = nullptr;
  const GCNSubtarget &ST;

  /// Reports unsupported message \p Msg for \p MI to LLVM context.
  void reportUnsupported(const MachineBasicBlock::iterator &MI,
                         const char *Msg) const;

  /// Inspects the target synchronization scope \p SSID and determines
  /// the SI atomic scope it corresponds to, the address spaces it
  /// covers, and whether the memory ordering applies between address
  /// spaces.
  std::optional<std::tuple<SIAtomicScope, SIAtomicAddrSpace, bool>>
  toSIAtomicScope(SyncScope::ID SSID, SIAtomicAddrSpace InstrAddrSpace) const;

  /// \return Return a bit set of the address spaces accessed by \p AS.
  SIAtomicAddrSpace toSIAtomicAddrSpace(unsigned AS) const;

  /// \returns Info constructed from \p MI, which has at least machine memory
  /// operand.
  std::optional<SIMemOpInfo>
  constructFromMIWithMMO(const MachineBasicBlock::iterator &MI) const;

public:
  /// Construct class to support accessing the machine memory operands
  /// of instructions in the machine function \p MF.
  SIMemOpAccess(const AMDGPUMachineModuleInfo &MMI, const GCNSubtarget &ST);

  /// \returns Load info if \p MI is a load operation, "std::nullopt" otherwise.
  std::optional<SIMemOpInfo>
  getLoadInfo(const MachineBasicBlock::iterator &MI) const;

  /// \returns Store info if \p MI is a store operation, "std::nullopt"
  /// otherwise.
  std::optional<SIMemOpInfo>
  getStoreInfo(const MachineBasicBlock::iterator &MI) const;

  /// \returns Atomic fence info if \p MI is an atomic fence operation,
  /// "std::nullopt" otherwise.
  std::optional<SIMemOpInfo>
  getAtomicFenceInfo(const MachineBasicBlock::iterator &MI) const;

  /// \returns Atomic cmpxchg/rmw info if \p MI is an atomic cmpxchg or
  /// rmw operation, "std::nullopt" otherwise.
  std::optional<SIMemOpInfo>
  getAtomicCmpxchgOrRmwInfo(const MachineBasicBlock::iterator &MI) const;

  /// \returns DMA to LDS info if \p MI is as a direct-to/from-LDS load/store,
  /// along with an indication of whether this is a load or store. If it is not
  /// a direct-to-LDS operation, returns std::nullopt.
  std::optional<SIMemOpInfo>
  getLDSDMAInfo(const MachineBasicBlock::iterator &MI) const;
};

class SICacheControl {
protected:

  /// AMDGPU subtarget info.
  const GCNSubtarget &ST;

  /// Instruction info.
  const SIInstrInfo *TII = nullptr;

  IsaVersion IV;

  /// Whether to insert cache invalidating instructions.
  bool InsertCacheInv;

  SICacheControl(const GCNSubtarget &ST);

  /// Sets CPol \p Bits to "true" if present in instruction \p MI.
  /// \returns Returns true if \p MI is modified, false otherwise.
  bool enableCPolBits(const MachineBasicBlock::iterator MI,
                      unsigned Bits) const;

  /// Check if any atomic operation on AS can affect memory accessible via the
  /// global address space.
  bool canAffectGlobalAddrSpace(SIAtomicAddrSpace AS) const;

public:
  using CPol = AMDGPU::CPol::CPol;

  /// Create a cache control for the subtarget \p ST.
  static std::unique_ptr<SICacheControl> create(const GCNSubtarget &ST);

  /// Update \p MI memory load instruction to bypass any caches up to
  /// the \p Scope memory scope for address spaces \p
  /// AddrSpace. Return true iff the instruction was modified.
  virtual bool enableLoadCacheBypass(const MachineBasicBlock::iterator &MI,
                                     SIAtomicScope Scope,
                                     SIAtomicAddrSpace AddrSpace) const = 0;

  /// Update \p MI memory store instruction to bypass any caches up to
  /// the \p Scope memory scope for address spaces \p
  /// AddrSpace. Return true iff the instruction was modified.
  virtual bool enableStoreCacheBypass(const MachineBasicBlock::iterator &MI,
                                      SIAtomicScope Scope,
                                      SIAtomicAddrSpace AddrSpace) const = 0;

  /// Update \p MI memory read-modify-write instruction to bypass any caches up
  /// to the \p Scope memory scope for address spaces \p AddrSpace. Return true
  /// iff the instruction was modified.
  virtual bool enableRMWCacheBypass(const MachineBasicBlock::iterator &MI,
                                    SIAtomicScope Scope,
                                    SIAtomicAddrSpace AddrSpace) const = 0;

  /// Update \p MI memory instruction of kind \p Op associated with address
  /// spaces \p AddrSpace to indicate it is volatile and/or
  /// nontemporal/last-use. Return true iff the instruction was modified.
  virtual bool enableVolatileAndOrNonTemporal(MachineBasicBlock::iterator &MI,
                                              SIAtomicAddrSpace AddrSpace,
                                              SIMemOp Op, bool IsVolatile,
                                              bool IsNonTemporal,
                                              bool IsLastUse = false) const = 0;

  /// Add final touches to a `mayStore` instruction \p MI, which may be a
  /// Store or RMW instruction.
  /// FIXME: This takes a MI because iterators aren't handled properly. When
  /// this is called, they often point to entirely different insts. Thus we back
  /// up the inst early and pass it here instead.
  virtual bool finalizeStore(MachineInstr &MI, bool Atomic) const {
    return false;
  };

  /// Handle cooperative load/store atomics.
  virtual bool handleCooperativeAtomic(MachineInstr &MI) const {
    llvm_unreachable(
        "cooperative atomics are not available on this architecture");
  }

  /// Inserts any necessary instructions at position \p Pos relative
  /// to instruction \p MI to ensure memory instructions before \p Pos of kind
  /// \p Op associated with address spaces \p AddrSpace have completed. Used
  /// between memory instructions to enforce the order they become visible as
  /// observed by other memory instructions executing in memory scope \p Scope.
  /// \p IsCrossAddrSpaceOrdering indicates if the memory ordering is between
  /// address spaces. If \p AtomicsOnly is true, only insert waits for counters
  /// that are used by atomic instructions.
  /// Returns true iff any instructions inserted.
  virtual bool insertWait(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                          SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                          bool IsCrossAddrSpaceOrdering, Position Pos,
                          AtomicOrdering Order, bool AtomicsOnly) const = 0;

  /// Inserts any necessary instructions at position \p Pos relative to
  /// instruction \p MI to ensure any subsequent memory instructions of this
  /// thread with address spaces \p AddrSpace will observe the previous memory
  /// operations by any thread for memory scopes up to memory scope \p Scope .
  /// Returns true iff any instructions inserted.
  virtual bool insertAcquire(MachineBasicBlock::iterator &MI,
                             SIAtomicScope Scope,
                             SIAtomicAddrSpace AddrSpace,
                             Position Pos) const = 0;

  /// Inserts any necessary instructions at position \p Pos relative to
  /// instruction \p MI to ensure previous memory instructions by this thread
  /// with address spaces \p AddrSpace have completed and can be observed by
  /// subsequent memory instructions by any thread executing in memory scope \p
  /// Scope. \p IsCrossAddrSpaceOrdering indicates if the memory ordering is
  /// between address spaces. Returns true iff any instructions inserted.
  virtual bool insertRelease(MachineBasicBlock::iterator &MI,
                             SIAtomicScope Scope,
                             SIAtomicAddrSpace AddrSpace,
                             bool IsCrossAddrSpaceOrdering,
                             Position Pos) const = 0;

  /// Virtual destructor to allow derivations to be deleted.
  virtual ~SICacheControl() = default;
};

/// Generates code sequences for the memory model of all GFX targets below
/// GFX10.
class SIGfx6CacheControl final : public SICacheControl {
public:

  SIGfx6CacheControl(const GCNSubtarget &ST) : SICacheControl(ST) {}

  bool enableLoadCacheBypass(const MachineBasicBlock::iterator &MI,
                             SIAtomicScope Scope,
                             SIAtomicAddrSpace AddrSpace) const override;

  bool enableStoreCacheBypass(const MachineBasicBlock::iterator &MI,
                              SIAtomicScope Scope,
                              SIAtomicAddrSpace AddrSpace) const override;

  bool enableRMWCacheBypass(const MachineBasicBlock::iterator &MI,
                            SIAtomicScope Scope,
                            SIAtomicAddrSpace AddrSpace) const override;

  bool enableVolatileAndOrNonTemporal(MachineBasicBlock::iterator &MI,
                                      SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                                      bool IsVolatile, bool IsNonTemporal,
                                      bool IsLastUse) const override;

  bool insertWait(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                  SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                  bool IsCrossAddrSpaceOrdering, Position Pos,
                  AtomicOrdering Order, bool AtomicsOnly) const override;

  bool insertAcquire(MachineBasicBlock::iterator &MI,
                     SIAtomicScope Scope,
                     SIAtomicAddrSpace AddrSpace,
                     Position Pos) const override;

  bool insertRelease(MachineBasicBlock::iterator &MI,
                     SIAtomicScope Scope,
                     SIAtomicAddrSpace AddrSpace,
                     bool IsCrossAddrSpaceOrdering,
                     Position Pos) const override;
};

/// Generates code sequences for the memory model of GFX10/11.
class SIGfx10CacheControl final : public SICacheControl {
public:
  SIGfx10CacheControl(const GCNSubtarget &ST) : SICacheControl(ST) {}

  bool enableLoadCacheBypass(const MachineBasicBlock::iterator &MI,
                             SIAtomicScope Scope,
                             SIAtomicAddrSpace AddrSpace) const override;

  bool enableStoreCacheBypass(const MachineBasicBlock::iterator &MI,
                              SIAtomicScope Scope,
                              SIAtomicAddrSpace AddrSpace) const override {
    return false;
  }

  bool enableRMWCacheBypass(const MachineBasicBlock::iterator &MI,
                            SIAtomicScope Scope,
                            SIAtomicAddrSpace AddrSpace) const override {
    return false;
  }

  bool enableVolatileAndOrNonTemporal(MachineBasicBlock::iterator &MI,
                                      SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                                      bool IsVolatile, bool IsNonTemporal,
                                      bool IsLastUse) const override;

  bool insertWait(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                  SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                  bool IsCrossAddrSpaceOrdering, Position Pos,
                  AtomicOrdering Order, bool AtomicsOnly) const override;

  bool insertAcquire(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                     SIAtomicAddrSpace AddrSpace, Position Pos) const override;

  bool insertRelease(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                     SIAtomicAddrSpace AddrSpace, bool IsCrossAddrSpaceOrdering,
                     Position Pos) const override {
    return insertWait(MI, Scope, AddrSpace, SIMemOp::LOAD | SIMemOp::STORE,
                      IsCrossAddrSpaceOrdering, Pos, AtomicOrdering::Release,
                      /*AtomicsOnly=*/false);
  }
};

class SIGfx12CacheControl final : public SICacheControl {
protected:
  // Sets TH policy to \p Value if CPol operand is present in instruction \p MI.
  // \returns Returns true if \p MI is modified, false otherwise.
  bool setTH(const MachineBasicBlock::iterator MI,
             AMDGPU::CPol::CPol Value) const;

  // Sets Scope policy to \p Value if CPol operand is present in instruction \p
  // MI. \returns Returns true if \p MI is modified, false otherwise.
  bool setScope(const MachineBasicBlock::iterator MI,
                AMDGPU::CPol::CPol Value) const;

  // Stores with system scope (SCOPE_SYS) need to wait for:
  // - loads or atomics(returning) - wait for {LOAD|SAMPLE|BVH|KM}CNT==0
  // - non-returning-atomics       - wait for STORECNT==0
  //   TODO: SIInsertWaitcnts will not always be able to remove STORECNT waits
  //   since it does not distinguish atomics-with-return from regular stores.
  // There is no need to wait if memory is cached (mtype != UC).
  bool
  insertWaitsBeforeSystemScopeStore(const MachineBasicBlock::iterator MI) const;

  bool setAtomicScope(const MachineBasicBlock::iterator &MI,
                      SIAtomicScope Scope, SIAtomicAddrSpace AddrSpace) const;

public:
  SIGfx12CacheControl(const GCNSubtarget &ST) : SICacheControl(ST) {
    // GFX120x and GFX125x memory models greatly overlap, and in some cases
    // the behavior is the same if assuming GFX120x in CU mode.
    assert(!ST.hasGFX1250Insts() || ST.hasGFX13Insts() || ST.isCuModeEnabled());
  }

  bool insertWait(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                  SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                  bool IsCrossAddrSpaceOrdering, Position Pos,
                  AtomicOrdering Order, bool AtomicsOnly) const override;

  bool insertAcquire(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                     SIAtomicAddrSpace AddrSpace, Position Pos) const override;

  bool enableVolatileAndOrNonTemporal(MachineBasicBlock::iterator &MI,
                                      SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                                      bool IsVolatile, bool IsNonTemporal,
                                      bool IsLastUse) const override;

  bool finalizeStore(MachineInstr &MI, bool Atomic) const override;

  bool handleCooperativeAtomic(MachineInstr &MI) const override;

  bool insertRelease(MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
                     SIAtomicAddrSpace AddrSpace, bool IsCrossAddrSpaceOrdering,
                     Position Pos) const override;

  bool enableLoadCacheBypass(const MachineBasicBlock::iterator &MI,
                             SIAtomicScope Scope,
                             SIAtomicAddrSpace AddrSpace) const override {
    return setAtomicScope(MI, Scope, AddrSpace);
  }

  bool enableStoreCacheBypass(const MachineBasicBlock::iterator &MI,
                              SIAtomicScope Scope,
                              SIAtomicAddrSpace AddrSpace) const override {
    return setAtomicScope(MI, Scope, AddrSpace);
  }

  bool enableRMWCacheBypass(const MachineBasicBlock::iterator &MI,
                            SIAtomicScope Scope,
                            SIAtomicAddrSpace AddrSpace) const override {
    return setAtomicScope(MI, Scope, AddrSpace);
  }
};

class SIMemoryLegalizer final {
private:
  const MachineModuleInfo &MMI;
  /// Cache Control.
  std::unique_ptr<SICacheControl> CC = nullptr;

  /// List of atomic pseudo instructions.
  std::list<MachineBasicBlock::iterator> AtomicPseudoMIs;

  /// Return true iff instruction \p MI is a atomic instruction that
  /// returns a result.
  bool isAtomicRet(const MachineInstr &MI) const {
    return SIInstrInfo::isAtomicRet(MI);
  }

  /// Removes all processed atomic pseudo instructions from the current
  /// function. Returns true if current function is modified, false otherwise.
  bool removeAtomicPseudoMIs();

  /// Expands load operation \p MI. Returns true if instructions are
  /// added/deleted or \p MI is modified, false otherwise.
  bool expandLoad(const SIMemOpInfo &MOI,
                  MachineBasicBlock::iterator &MI);
  /// Expands store operation \p MI. Returns true if instructions are
  /// added/deleted or \p MI is modified, false otherwise.
  bool expandStore(const SIMemOpInfo &MOI,
                   MachineBasicBlock::iterator &MI);
  /// Expands atomic fence operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicFence(const SIMemOpInfo &MOI,
                         MachineBasicBlock::iterator &MI);
  /// Expands atomic cmpxchg or rmw operation \p MI. Returns true if
  /// instructions are added/deleted or \p MI is modified, false otherwise.
  bool expandAtomicCmpxchgOrRmw(const SIMemOpInfo &MOI,
                                MachineBasicBlock::iterator &MI);
  /// Expands LDS DMA operation \p MI. Returns true if instructions are
  /// added/deleted or \p MI is modified, false otherwise.
  bool expandLDSDMA(const SIMemOpInfo &MOI, MachineBasicBlock::iterator &MI);

public:
  SIMemoryLegalizer(const MachineModuleInfo &MMI) : MMI(MMI) {};
  bool run(MachineFunction &MF);
};

class SIMemoryLegalizerLegacy final : public MachineFunctionPass {
public:
  static char ID;

  SIMemoryLegalizerLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return PASS_NAME;
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

static const StringMap<SIAtomicAddrSpace> ASNames = {{
    {"global", SIAtomicAddrSpace::GLOBAL},
    {"local", SIAtomicAddrSpace::LDS},
}};

void diagnoseUnknownMMRAASName(const MachineInstr &MI, StringRef AS) {
  const MachineFunction *MF = MI.getMF();
  const Function &Fn = MF->getFunction();
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  OS << "unknown address space '" << AS << "'; expected one of ";
  ListSeparator LS;
  for (const auto &[Name, Val] : ASNames)
    OS << LS << '\'' << Name << '\'';
  Fn.getContext().diagnose(
      DiagnosticInfoUnsupported(Fn, Str.str(), MI.getDebugLoc(), DS_Warning));
}

/// Reads \p MI's MMRAs to parse the "amdgpu-synchronize-as" MMRA.
/// If this tag isn't present, or if it has no meaningful values, returns
/// \p none, otherwise returns the address spaces specified by the MD.
static std::optional<SIAtomicAddrSpace>
getSynchronizeAddrSpaceMD(const MachineInstr &MI) {
  static constexpr StringLiteral FenceASPrefix = "amdgpu-synchronize-as";

  auto MMRA = MMRAMetadata(MI.getMMRAMetadata());
  if (!MMRA)
    return std::nullopt;

  SIAtomicAddrSpace Result = SIAtomicAddrSpace::NONE;
  for (const auto &[Prefix, Suffix] : MMRA) {
    if (Prefix != FenceASPrefix)
      continue;

    if (auto It = ASNames.find(Suffix); It != ASNames.end())
      Result |= It->second;
    else
      diagnoseUnknownMMRAASName(MI, Suffix);
  }

  if (Result == SIAtomicAddrSpace::NONE)
    return std::nullopt;

  return Result;
}

} // end anonymous namespace

void SIMemOpAccess::reportUnsupported(const MachineBasicBlock::iterator &MI,
                                      const char *Msg) const {
  const Function &Func = MI->getMF()->getFunction();
  Func.getContext().diagnose(
      DiagnosticInfoUnsupported(Func, Msg, MI->getDebugLoc()));
}

std::optional<std::tuple<SIAtomicScope, SIAtomicAddrSpace, bool>>
SIMemOpAccess::toSIAtomicScope(SyncScope::ID SSID,
                               SIAtomicAddrSpace InstrAddrSpace) const {
  if (SSID == SyncScope::System)
    return std::tuple(SIAtomicScope::SYSTEM, SIAtomicAddrSpace::ATOMIC, true);
  if (SSID == MMI->getAgentSSID())
    return std::tuple(SIAtomicScope::AGENT, SIAtomicAddrSpace::ATOMIC, true);
  if (SSID == MMI->getClusterSSID())
    return std::tuple(SIAtomicScope::CLUSTER, SIAtomicAddrSpace::ATOMIC, true);
  if (SSID == MMI->getWorkgroupSSID())
    return std::tuple(SIAtomicScope::WORKGROUP, SIAtomicAddrSpace::ATOMIC,
                      true);
  if (SSID == MMI->getWavefrontSSID())
    return std::tuple(SIAtomicScope::WAVEFRONT, SIAtomicAddrSpace::ATOMIC,
                      true);
  if (SSID == SyncScope::SingleThread)
    return std::tuple(SIAtomicScope::SINGLETHREAD, SIAtomicAddrSpace::ATOMIC,
                      true);
  if (SSID == MMI->getSystemOneAddressSpaceSSID())
    return std::tuple(SIAtomicScope::SYSTEM,
                      SIAtomicAddrSpace::ATOMIC & InstrAddrSpace, false);
  if (SSID == MMI->getAgentOneAddressSpaceSSID())
    return std::tuple(SIAtomicScope::AGENT,
                      SIAtomicAddrSpace::ATOMIC & InstrAddrSpace, false);
  if (SSID == MMI->getClusterOneAddressSpaceSSID())
    return std::tuple(SIAtomicScope::CLUSTER,
                      SIAtomicAddrSpace::ATOMIC & InstrAddrSpace, false);
  if (SSID == MMI->getWorkgroupOneAddressSpaceSSID())
    return std::tuple(SIAtomicScope::WORKGROUP,
                      SIAtomicAddrSpace::ATOMIC & InstrAddrSpace, false);
  if (SSID == MMI->getWavefrontOneAddressSpaceSSID())
    return std::tuple(SIAtomicScope::WAVEFRONT,
                      SIAtomicAddrSpace::ATOMIC & InstrAddrSpace, false);
  if (SSID == MMI->getSingleThreadOneAddressSpaceSSID())
    return std::tuple(SIAtomicScope::SINGLETHREAD,
                      SIAtomicAddrSpace::ATOMIC & InstrAddrSpace, false);
  return std::nullopt;
}

SIAtomicAddrSpace SIMemOpAccess::toSIAtomicAddrSpace(unsigned AS) const {
  if (AS == AMDGPUAS::FLAT_ADDRESS)
    return SIAtomicAddrSpace::FLAT;
  if (AS == AMDGPUAS::GLOBAL_ADDRESS)
    return SIAtomicAddrSpace::GLOBAL;
  if (AS == AMDGPUAS::LOCAL_ADDRESS)
    return SIAtomicAddrSpace::LDS;
  if (AS == AMDGPUAS::PRIVATE_ADDRESS)
    return SIAtomicAddrSpace::SCRATCH;
  if (AS == AMDGPUAS::REGION_ADDRESS)
    return SIAtomicAddrSpace::GDS;
  if (AS == AMDGPUAS::BUFFER_FAT_POINTER || AS == AMDGPUAS::BUFFER_RESOURCE ||
      AS == AMDGPUAS::BUFFER_STRIDED_POINTER)
    return SIAtomicAddrSpace::GLOBAL;

  return SIAtomicAddrSpace::OTHER;
}

SIMemOpAccess::SIMemOpAccess(const AMDGPUMachineModuleInfo &MMI_,
                             const GCNSubtarget &ST)
    : MMI(&MMI_), ST(ST) {}

std::optional<SIMemOpInfo> SIMemOpAccess::constructFromMIWithMMO(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getNumMemOperands() > 0);

  SyncScope::ID SSID = SyncScope::SingleThread;
  AtomicOrdering Ordering = AtomicOrdering::NotAtomic;
  AtomicOrdering FailureOrdering = AtomicOrdering::NotAtomic;
  SIAtomicAddrSpace InstrAddrSpace = SIAtomicAddrSpace::NONE;
  bool IsNonTemporal = true;
  bool IsVolatile = false;
  bool IsLastUse = false;
  bool IsCooperative = false;

  // Validator should check whether or not MMOs cover the entire set of
  // locations accessed by the memory instruction.
  for (const auto &MMO : MI->memoperands()) {
    IsNonTemporal &= MMO->isNonTemporal();
    IsVolatile |= MMO->isVolatile();
    IsLastUse |= MMO->getFlags() & MOLastUse;
    IsCooperative |= MMO->getFlags() & MOCooperative;
    InstrAddrSpace |=
      toSIAtomicAddrSpace(MMO->getPointerInfo().getAddrSpace());
    AtomicOrdering OpOrdering = MMO->getSuccessOrdering();
    if (OpOrdering != AtomicOrdering::NotAtomic) {
      const auto &IsSyncScopeInclusion =
          MMI->isSyncScopeInclusion(SSID, MMO->getSyncScopeID());
      if (!IsSyncScopeInclusion) {
        reportUnsupported(MI,
          "Unsupported non-inclusive atomic synchronization scope");
        return std::nullopt;
      }

      SSID = *IsSyncScopeInclusion ? SSID : MMO->getSyncScopeID();
      Ordering = getMergedAtomicOrdering(Ordering, OpOrdering);
      assert(MMO->getFailureOrdering() != AtomicOrdering::Release &&
             MMO->getFailureOrdering() != AtomicOrdering::AcquireRelease);
      FailureOrdering =
          getMergedAtomicOrdering(FailureOrdering, MMO->getFailureOrdering());
    }
  }

  // FIXME: The MMO of buffer atomic instructions does not always have an atomic
  // ordering. We only need to handle VBUFFER atomics on GFX12+ so we can fix it
  // here, but the lowering should really be cleaned up at some point.
  if ((ST.getGeneration() >= GCNSubtarget::GFX12) && SIInstrInfo::isBUF(*MI) &&
      SIInstrInfo::isAtomic(*MI) && Ordering == AtomicOrdering::NotAtomic)
    Ordering = AtomicOrdering::Monotonic;

  SIAtomicScope Scope = SIAtomicScope::NONE;
  SIAtomicAddrSpace OrderingAddrSpace = SIAtomicAddrSpace::NONE;
  bool IsCrossAddressSpaceOrdering = false;
  if (Ordering != AtomicOrdering::NotAtomic) {
    auto ScopeOrNone = toSIAtomicScope(SSID, InstrAddrSpace);
    if (!ScopeOrNone) {
      reportUnsupported(MI, "Unsupported atomic synchronization scope");
      return std::nullopt;
    }
    std::tie(Scope, OrderingAddrSpace, IsCrossAddressSpaceOrdering) =
        *ScopeOrNone;
    if ((OrderingAddrSpace == SIAtomicAddrSpace::NONE) ||
        ((OrderingAddrSpace & SIAtomicAddrSpace::ATOMIC) != OrderingAddrSpace) ||
        ((InstrAddrSpace & SIAtomicAddrSpace::ATOMIC) == SIAtomicAddrSpace::NONE)) {
      reportUnsupported(MI, "Unsupported atomic address space");
      return std::nullopt;
    }
  }
  return SIMemOpInfo(ST, Ordering, Scope, OrderingAddrSpace, InstrAddrSpace,
                     IsCrossAddressSpaceOrdering, FailureOrdering, IsVolatile,
                     IsNonTemporal, IsLastUse, IsCooperative);
}

std::optional<SIMemOpInfo>
SIMemOpAccess::getLoadInfo(const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && !MI->mayStore()))
    return std::nullopt;

  // Be conservative if there are no memory operands.
  if (MI->getNumMemOperands() == 0)
    return SIMemOpInfo(ST);

  return constructFromMIWithMMO(MI);
}

std::optional<SIMemOpInfo>
SIMemOpAccess::getStoreInfo(const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(!MI->mayLoad() && MI->mayStore()))
    return std::nullopt;

  // Be conservative if there are no memory operands.
  if (MI->getNumMemOperands() == 0)
    return SIMemOpInfo(ST);

  return constructFromMIWithMMO(MI);
}

std::optional<SIMemOpInfo>
SIMemOpAccess::getAtomicFenceInfo(const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (MI->getOpcode() != AMDGPU::ATOMIC_FENCE)
    return std::nullopt;

  AtomicOrdering Ordering =
    static_cast<AtomicOrdering>(MI->getOperand(0).getImm());

  SyncScope::ID SSID = static_cast<SyncScope::ID>(MI->getOperand(1).getImm());
  auto ScopeOrNone = toSIAtomicScope(SSID, SIAtomicAddrSpace::ATOMIC);
  if (!ScopeOrNone) {
    reportUnsupported(MI, "Unsupported atomic synchronization scope");
    return std::nullopt;
  }

  SIAtomicScope Scope = SIAtomicScope::NONE;
  SIAtomicAddrSpace OrderingAddrSpace = SIAtomicAddrSpace::NONE;
  bool IsCrossAddressSpaceOrdering = false;
  std::tie(Scope, OrderingAddrSpace, IsCrossAddressSpaceOrdering) =
      *ScopeOrNone;

  if (OrderingAddrSpace != SIAtomicAddrSpace::ATOMIC) {
    // We currently expect refineOrderingAS to be the only place that
    // can refine the AS ordered by the fence.
    // If that changes, we need to review the semantics of that function
    // in case it needs to preserve certain address spaces.
    reportUnsupported(MI, "Unsupported atomic address space");
    return std::nullopt;
  }

  auto SynchronizeAS = getSynchronizeAddrSpaceMD(*MI);
  if (SynchronizeAS)
    OrderingAddrSpace = *SynchronizeAS;

  return SIMemOpInfo(ST, Ordering, Scope, OrderingAddrSpace,
                     SIAtomicAddrSpace::ATOMIC, IsCrossAddressSpaceOrdering,
                     AtomicOrdering::NotAtomic);
}

std::optional<SIMemOpInfo> SIMemOpAccess::getAtomicCmpxchgOrRmwInfo(
    const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!(MI->mayLoad() && MI->mayStore()))
    return std::nullopt;

  // Be conservative if there are no memory operands.
  if (MI->getNumMemOperands() == 0)
    return SIMemOpInfo(ST);

  return constructFromMIWithMMO(MI);
}

std::optional<SIMemOpInfo>
SIMemOpAccess::getLDSDMAInfo(const MachineBasicBlock::iterator &MI) const {
  assert(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic);

  if (!SIInstrInfo::isLDSDMA(*MI))
    return std::nullopt;

  return constructFromMIWithMMO(MI);
}

SICacheControl::SICacheControl(const GCNSubtarget &ST) : ST(ST) {
  TII = ST.getInstrInfo();
  IV = getIsaVersion(ST.getCPU());
  InsertCacheInv = !AmdgcnSkipCacheInvalidations;
}

bool SICacheControl::enableCPolBits(const MachineBasicBlock::iterator MI,
                                    unsigned Bits) const {
  MachineOperand *CPol = TII->getNamedOperand(*MI, AMDGPU::OpName::cpol);
  if (!CPol)
    return false;

  CPol->setImm(CPol->getImm() | Bits);
  return true;
}

bool SICacheControl::canAffectGlobalAddrSpace(SIAtomicAddrSpace AS) const {
  assert((!ST.hasGloballyAddressableScratch() ||
          (AS & SIAtomicAddrSpace::GLOBAL) != SIAtomicAddrSpace::NONE ||
          (AS & SIAtomicAddrSpace::SCRATCH) == SIAtomicAddrSpace::NONE) &&
         "scratch instructions should already be replaced by flat "
         "instructions if GloballyAddressableScratch is enabled");
  return (AS & SIAtomicAddrSpace::GLOBAL) != SIAtomicAddrSpace::NONE;
}

/* static */
std::unique_ptr<SICacheControl> SICacheControl::create(const GCNSubtarget &ST) {
  GCNSubtarget::Generation Generation = ST.getGeneration();
  if (Generation < AMDGPUSubtarget::GFX10)
    return std::make_unique<SIGfx6CacheControl>(ST);
  if (Generation < AMDGPUSubtarget::GFX12)
    return std::make_unique<SIGfx10CacheControl>(ST);
  return std::make_unique<SIGfx12CacheControl>(ST);
}

bool SIGfx6CacheControl::enableLoadCacheBypass(
    const MachineBasicBlock::iterator &MI,
    SIAtomicScope Scope,
    SIAtomicAddrSpace AddrSpace) const {
  assert(MI->mayLoad() && !MI->mayStore());

  if (!canAffectGlobalAddrSpace(AddrSpace)) {
    /// The scratch address space does not need the global memory caches
    /// to be bypassed as all memory operations by the same thread are
    /// sequentially consistent, and no other thread can access scratch
    /// memory.

    /// Other address spaces do not have a cache.
    return false;
  }

  bool Changed = false;
  switch (Scope) {
  case SIAtomicScope::SYSTEM:
    if (ST.hasGFX940Insts()) {
      // Set SC bits to indicate system scope.
      Changed |= enableCPolBits(MI, CPol::SC0 | CPol::SC1);
      break;
    }
    [[fallthrough]];
  case SIAtomicScope::AGENT:
    if (ST.hasGFX940Insts()) {
      // Set SC bits to indicate agent scope.
      Changed |= enableCPolBits(MI, CPol::SC1);
    } else {
      // Set L1 cache policy to MISS_EVICT.
      // Note: there is no L2 cache bypass policy at the ISA level.
      Changed |= enableCPolBits(MI, CPol::GLC);
    }
    break;
  case SIAtomicScope::WORKGROUP:
    if (ST.hasGFX940Insts()) {
      // In threadgroup split mode the waves of a work-group can be executing
      // on different CUs. Therefore need to bypass the L1 which is per CU.
      // Otherwise in non-threadgroup split mode all waves of a work-group are
      // on the same CU, and so the L1 does not need to be bypassed. Setting
      // SC bits to indicate work-group scope will do this automatically.
      Changed |= enableCPolBits(MI, CPol::SC0);
    } else if (ST.hasGFX90AInsts()) {
      // In threadgroup split mode the waves of a work-group can be executing
      // on different CUs. Therefore need to bypass the L1 which is per CU.
      // Otherwise in non-threadgroup split mode all waves of a work-group are
      // on the same CU, and so the L1 does not need to be bypassed.
      if (ST.isTgSplitEnabled())
        Changed |= enableCPolBits(MI, CPol::GLC);
    }
    break;
  case SIAtomicScope::WAVEFRONT:
  case SIAtomicScope::SINGLETHREAD:
    // No cache to bypass.
    break;
  default:
    llvm_unreachable("Unsupported synchronization scope");
  }

  return Changed;
}

bool SIGfx6CacheControl::enableStoreCacheBypass(
    const MachineBasicBlock::iterator &MI,
    SIAtomicScope Scope,
    SIAtomicAddrSpace AddrSpace) const {
  assert(!MI->mayLoad() && MI->mayStore());
  bool Changed = false;

  /// For targets other than GFX940, the L1 cache is write through so does not
  /// need to be bypassed. There is no bypass control for the L2 cache at the
  /// isa level.

  if (ST.hasGFX940Insts() && canAffectGlobalAddrSpace(AddrSpace)) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
      // Set SC bits to indicate system scope.
      Changed |= enableCPolBits(MI, CPol::SC0 | CPol::SC1);
      break;
    case SIAtomicScope::AGENT:
      // Set SC bits to indicate agent scope.
      Changed |= enableCPolBits(MI, CPol::SC1);
      break;
    case SIAtomicScope::WORKGROUP:
      // Set SC bits to indicate workgroup scope.
      Changed |= enableCPolBits(MI, CPol::SC0);
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // Leave SC bits unset to indicate wavefront scope.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }

    /// The scratch address space does not need the global memory caches
    /// to be bypassed as all memory operations by the same thread are
    /// sequentially consistent, and no other thread can access scratch
    /// memory.

    /// Other address spaces do not have a cache.
  }

  return Changed;
}

bool SIGfx6CacheControl::enableRMWCacheBypass(
    const MachineBasicBlock::iterator &MI,
    SIAtomicScope Scope,
    SIAtomicAddrSpace AddrSpace) const {
  assert(MI->mayLoad() && MI->mayStore());
  bool Changed = false;

  /// For targets other than GFX940, do not set GLC for RMW atomic operations as
  /// L0/L1 cache is automatically bypassed, and the GLC bit is instead used to
  /// indicate if they are return or no-return. Note: there is no L2 cache
  /// coherent bypass control at the ISA level.
  ///       For GFX90A+, RMW atomics implicitly bypass the L1 cache.

  if (ST.hasGFX940Insts() && canAffectGlobalAddrSpace(AddrSpace)) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
      // Set SC1 bit to indicate system scope.
      Changed |= enableCPolBits(MI, CPol::SC1);
      break;
    case SIAtomicScope::AGENT:
    case SIAtomicScope::WORKGROUP:
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // RMW atomic operations implicitly bypass the L1 cache and only use SC1
      // to indicate system or agent scope. The SC0 bit is used to indicate if
      // they are return or no-return. Leave SC1 bit unset to indicate agent
      // scope.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  return Changed;
}

bool SIGfx6CacheControl::enableVolatileAndOrNonTemporal(
    MachineBasicBlock::iterator &MI, SIAtomicAddrSpace AddrSpace, SIMemOp Op,
    bool IsVolatile, bool IsNonTemporal, bool IsLastUse = false) const {
  // Only handle load and store, not atomic read-modify-write insructions. The
  // latter use glc to indicate if the atomic returns a result and so must not
  // be used for cache control.
  assert((MI->mayLoad() ^ MI->mayStore()) || SIInstrInfo::isLDSDMA(*MI));

  // Only update load and store, not LLVM IR atomic read-modify-write
  // instructions. The latter are always marked as volatile so cannot sensibly
  // handle it as do not want to pessimize all atomics. Also they do not support
  // the nontemporal attribute.
  assert(Op == SIMemOp::LOAD || Op == SIMemOp::STORE);

  bool Changed = false;

  if (IsVolatile) {
    if (ST.hasGFX940Insts()) {
      // Set SC bits to indicate system scope.
      Changed |= enableCPolBits(MI, CPol::SC0 | CPol::SC1);
    } else if (Op == SIMemOp::LOAD) {
      // Set L1 cache policy to be MISS_EVICT for load instructions
      // and MISS_LRU for store instructions.
      // Note: there is no L2 cache bypass policy at the ISA level.
      Changed |= enableCPolBits(MI, CPol::GLC);
    }

    // Ensure operation has completed at system scope to cause all volatile
    // operations to be visible outside the program in a global order. Do not
    // request cross address space as only the global address space can be
    // observable outside the program, so no need to cause a waitcnt for LDS
    // address space operations.
    Changed |= insertWait(MI, SIAtomicScope::SYSTEM, AddrSpace, Op, false,
                          Position::AFTER, AtomicOrdering::Unordered,
                          /*AtomicsOnly=*/false);

    return Changed;
  }

  if (IsNonTemporal) {
    if (ST.hasGFX940Insts()) {
      Changed |= enableCPolBits(MI, CPol::NT);
    } else {
      // Setting both GLC and SLC configures L1 cache policy to MISS_EVICT
      // for both loads and stores, and the L2 cache policy to STREAM.
      Changed |= enableCPolBits(MI, CPol::SLC | CPol::GLC);
    }
    return Changed;
  }

  return Changed;
}

bool SIGfx6CacheControl::insertWait(MachineBasicBlock::iterator &MI,
                                    SIAtomicScope Scope,
                                    SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                                    bool IsCrossAddrSpaceOrdering, Position Pos,
                                    AtomicOrdering Order,
                                    bool AtomicsOnly) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  // GFX90A+
  if (ST.hasGFX90AInsts() && ST.isTgSplitEnabled()) {
    // In threadgroup split mode the waves of a work-group can be executing on
    // different CUs. Therefore need to wait for global or GDS memory operations
    // to complete to ensure they are visible to waves in the other CUs.
    // Otherwise in non-threadgroup split mode all waves of a work-group are on
    // the same CU, so no need to wait for global memory as all waves in the
    // work-group access the same the L1, nor wait for GDS as access are ordered
    // on a CU.
    if (((AddrSpace & (SIAtomicAddrSpace::GLOBAL | SIAtomicAddrSpace::SCRATCH |
                       SIAtomicAddrSpace::GDS)) != SIAtomicAddrSpace::NONE) &&
        (Scope == SIAtomicScope::WORKGROUP)) {
      // Same as <GFX90A at AGENT scope;
      Scope = SIAtomicScope::AGENT;
    }
    // In threadgroup split mode LDS cannot be allocated so no need to wait for
    // LDS memory operations.
    AddrSpace &= ~SIAtomicAddrSpace::LDS;
  }

  bool VMCnt = false;
  bool LGKMCnt = false;

  if ((AddrSpace & (SIAtomicAddrSpace::GLOBAL | SIAtomicAddrSpace::SCRATCH)) !=
      SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
      VMCnt |= true;
      break;
    case SIAtomicScope::WORKGROUP:
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The L1 cache keeps all memory operations in order for
      // wavefronts in the same work-group.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if ((AddrSpace & SIAtomicAddrSpace::LDS) != SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
    case SIAtomicScope::WORKGROUP:
      // If no cross address space ordering then an "S_WAITCNT lgkmcnt(0)" is
      // not needed as LDS operations for all waves are executed in a total
      // global ordering as observed by all waves. Required if also
      // synchronizing with global/GDS memory as LDS operations could be
      // reordered with respect to later global/GDS memory operations of the
      // same wave.
      LGKMCnt |= IsCrossAddrSpaceOrdering;
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The LDS keeps all memory operations in order for
      // the same wavefront.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if ((AddrSpace & SIAtomicAddrSpace::GDS) != SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
      // If no cross address space ordering then an GDS "S_WAITCNT lgkmcnt(0)"
      // is not needed as GDS operations for all waves are executed in a total
      // global ordering as observed by all waves. Required if also
      // synchronizing with global/LDS memory as GDS operations could be
      // reordered with respect to later global/LDS memory operations of the
      // same wave.
      LGKMCnt |= IsCrossAddrSpaceOrdering;
      break;
    case SIAtomicScope::WORKGROUP:
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The GDS keeps all memory operations in order for
      // the same work-group.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if (VMCnt || LGKMCnt) {
    unsigned WaitCntImmediate =
      AMDGPU::encodeWaitcnt(IV,
                            VMCnt ? 0 : getVmcntBitMask(IV),
                            getExpcntBitMask(IV),
                            LGKMCnt ? 0 : getLgkmcntBitMask(IV));
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAITCNT_soft))
        .addImm(WaitCntImmediate);
    Changed = true;
  }

  // On architectures that support direct loads to LDS, emit an unknown waitcnt
  // at workgroup-scoped release operations that specify the LDS address space.
  // SIInsertWaitcnts will later replace this with a vmcnt().
  if (ST.hasVMemToLDSLoad() && isReleaseOrStronger(Order) &&
      Scope == SIAtomicScope::WORKGROUP &&
      (AddrSpace & SIAtomicAddrSpace::LDS) != SIAtomicAddrSpace::NONE) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAITCNT_lds_direct));
    Changed = true;
  }

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

static bool canUseBUFFER_WBINVL1_VOL(const GCNSubtarget &ST) {
  if (ST.getGeneration() <= AMDGPUSubtarget::SOUTHERN_ISLANDS)
    return false;
  return !ST.isAmdPalOS() && !ST.isMesa3DOS();
}

bool SIGfx6CacheControl::insertAcquire(MachineBasicBlock::iterator &MI,
                                       SIAtomicScope Scope,
                                       SIAtomicAddrSpace AddrSpace,
                                       Position Pos) const {
  if (!InsertCacheInv)
    return false;

  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  const unsigned InvalidateL1 = canUseBUFFER_WBINVL1_VOL(ST)
                                    ? AMDGPU::BUFFER_WBINVL1_VOL
                                    : AMDGPU::BUFFER_WBINVL1;

  if (canAffectGlobalAddrSpace(AddrSpace)) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
      if (ST.hasGFX940Insts()) {
        // Ensures that following loads will not see stale remote VMEM data or
        // stale local VMEM data with MTYPE NC. Local VMEM data with MTYPE RW
        // and CC will never be stale due to the local memory probes.
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_INV))
            // Set SC bits to indicate system scope.
            .addImm(AMDGPU::CPol::SC0 | AMDGPU::CPol::SC1);
        // Inserting a "S_WAITCNT vmcnt(0)" after is not required because the
        // hardware does not reorder memory operations by the same wave with
        // respect to a preceding "BUFFER_INV". The invalidate is guaranteed to
        // remove any cache lines of earlier writes by the same wave and ensures
        // later reads by the same wave will refetch the cache lines.
        Changed = true;
        break;
      }

      if (ST.hasGFX90AInsts()) {
        // Ensures that following loads will not see stale remote VMEM data or
        // stale local VMEM data with MTYPE NC. Local VMEM data with MTYPE RW
        // and CC will never be stale due to the local memory probes.
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_INVL2));
        BuildMI(MBB, MI, DL, TII->get(InvalidateL1));
        // Inserting a "S_WAITCNT vmcnt(0)" after is not required because the
        // hardware does not reorder memory operations by the same wave with
        // respect to a preceding "BUFFER_INVL2". The invalidate is guaranteed
        // to remove any cache lines of earlier writes by the same wave and
        // ensures later reads by the same wave will refetch the cache lines.
        Changed = true;
        break;
      }
      [[fallthrough]];
    case SIAtomicScope::AGENT:
      if (ST.hasGFX940Insts()) {
        // Ensures that following loads will not see stale remote date or local
        // MTYPE NC global data. Local MTYPE RW and CC memory will never be
        // stale due to the memory probes.
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_INV))
            // Set SC bits to indicate agent scope.
            .addImm(AMDGPU::CPol::SC1);
        // Inserting "S_WAITCNT vmcnt(0)" is not required because the hardware
        // does not reorder memory operations with respect to preceeding buffer
        // invalidate. The invalidate is guaranteed to remove any cache lines of
        // earlier writes and ensures later writes will refetch the cache lines.
      } else
        BuildMI(MBB, MI, DL, TII->get(InvalidateL1));
      Changed = true;
      break;
    case SIAtomicScope::WORKGROUP:
      if (ST.isTgSplitEnabled()) {
        if (ST.hasGFX940Insts()) {
          // In threadgroup split mode the waves of a work-group can be
          // executing on different CUs. Therefore need to invalidate the L1
          // which is per CU. Otherwise in non-threadgroup split mode all waves
          // of a work-group are on the same CU, and so the L1 does not need to
          // be invalidated.

          // Ensures L1 is invalidated if in threadgroup split mode. In
          // non-threadgroup split mode it is a NOP, but no point generating it
          // in that case if know not in that mode.
          BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_INV))
              // Set SC bits to indicate work-group scope.
              .addImm(AMDGPU::CPol::SC0);
          // Inserting "S_WAITCNT vmcnt(0)" is not required because the hardware
          // does not reorder memory operations with respect to preceeding
          // buffer invalidate. The invalidate is guaranteed to remove any cache
          // lines of earlier writes and ensures later writes will refetch the
          // cache lines.
          Changed = true;
        } else if (ST.hasGFX90AInsts()) {
          BuildMI(MBB, MI, DL, TII->get(InvalidateL1));
          Changed = true;
        }
      }
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // For GFX940, we could generate "BUFFER_INV" but it would do nothing as
      // there are no caches to invalidate. All other targets have no cache to
      // invalidate.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  /// The scratch address space does not need the global memory cache
  /// to be flushed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not have a cache.

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool SIGfx6CacheControl::insertRelease(MachineBasicBlock::iterator &MI,
                                       SIAtomicScope Scope,
                                       SIAtomicAddrSpace AddrSpace,
                                       bool IsCrossAddrSpaceOrdering,
                                       Position Pos) const {
  bool Changed = false;

  if (ST.hasGFX90AInsts()) {
    MachineBasicBlock &MBB = *MI->getParent();
    const DebugLoc &DL = MI->getDebugLoc();

    if (Pos == Position::AFTER)
      ++MI;

    if (canAffectGlobalAddrSpace(AddrSpace)) {
      switch (Scope) {
      case SIAtomicScope::SYSTEM:
        // Inserting a "S_WAITCNT vmcnt(0)" before is not required because the
        // hardware does not reorder memory operations by the same wave with
        // respect to a following "BUFFER_WBL2". The "BUFFER_WBL2" is guaranteed
        // to initiate writeback of any dirty cache lines of earlier writes by
        // the same wave. A "S_WAITCNT vmcnt(0)" is needed after to ensure the
        // writeback has completed.
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_WBL2))
            // Set SC bits to indicate system scope.
            .addImm(AMDGPU::CPol::SC0 | AMDGPU::CPol::SC1);
        Changed = true;
        break;
      case SIAtomicScope::AGENT:
        if (ST.hasGFX940Insts()) {
          BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_WBL2))
              // Set SC bits to indicate agent scope.
              .addImm(AMDGPU::CPol::SC1);

          // Since AddrSpace contains SIAtomicAddrSpace::GLOBAL and Scope is
          // SIAtomicScope::AGENT, the following insertWait will generate the
          // required "S_WAITCNT vmcnt(0)".
          Changed = true;
        }
        break;
      case SIAtomicScope::WORKGROUP:
      case SIAtomicScope::WAVEFRONT:
      case SIAtomicScope::SINGLETHREAD:
        // For GFX940, do not generate "BUFFER_WBL2" as there are no caches it
        // would writeback, and would require an otherwise unnecessary
        // "S_WAITCNT vmcnt(0)".
        break;
      default:
        llvm_unreachable("Unsupported synchronization scope");
      }
    }

    if (Pos == Position::AFTER)
      --MI;
  }

  // Ensure the necessary S_WAITCNT needed by any "BUFFER_WBL2" as well as other
  // S_WAITCNT needed.
  Changed |= insertWait(MI, Scope, AddrSpace, SIMemOp::LOAD | SIMemOp::STORE,
                        IsCrossAddrSpaceOrdering, Pos, AtomicOrdering::Release,
                        /*AtomicsOnly=*/false);

  return Changed;
}

bool SIGfx10CacheControl::enableLoadCacheBypass(
    const MachineBasicBlock::iterator &MI, SIAtomicScope Scope,
    SIAtomicAddrSpace AddrSpace) const {
  assert(MI->mayLoad() && !MI->mayStore());
  bool Changed = false;

  if (canAffectGlobalAddrSpace(AddrSpace)) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
      // Set the L0 and L1 cache policies to MISS_EVICT.
      // Note: there is no L2 cache coherent bypass control at the ISA level.
      // For GFX10, set GLC+DLC, for GFX11, only set GLC.
      Changed |=
          enableCPolBits(MI, CPol::GLC | (AMDGPU::isGFX10(ST) ? CPol::DLC : 0));
      break;
    case SIAtomicScope::WORKGROUP:
      // In WGP mode the waves of a work-group can be executing on either CU of
      // the WGP. Therefore need to bypass the L0 which is per CU. Otherwise in
      // CU mode all waves of a work-group are on the same CU, and so the L0
      // does not need to be bypassed.
      if (!ST.isCuModeEnabled())
        Changed |= enableCPolBits(MI, CPol::GLC);
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // No cache to bypass.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  /// The scratch address space does not need the global memory caches
  /// to be bypassed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not have a cache.

  return Changed;
}

bool SIGfx10CacheControl::enableVolatileAndOrNonTemporal(
    MachineBasicBlock::iterator &MI, SIAtomicAddrSpace AddrSpace, SIMemOp Op,
    bool IsVolatile, bool IsNonTemporal, bool IsLastUse = false) const {

  // Only handle load and store, not atomic read-modify-write insructions. The
  // latter use glc to indicate if the atomic returns a result and so must not
  // be used for cache control.
  assert((MI->mayLoad() ^ MI->mayStore()) || SIInstrInfo::isLDSDMA(*MI));

  // Only update load and store, not LLVM IR atomic read-modify-write
  // instructions. The latter are always marked as volatile so cannot sensibly
  // handle it as do not want to pessimize all atomics. Also they do not support
  // the nontemporal attribute.
  assert(Op == SIMemOp::LOAD || Op == SIMemOp::STORE);

  bool Changed = false;

  if (IsVolatile) {
    // Set L0 and L1 cache policy to be MISS_EVICT for load instructions
    // and MISS_LRU for store instructions.
    // Note: there is no L2 cache coherent bypass control at the ISA level.
    if (Op == SIMemOp::LOAD) {
      Changed |= enableCPolBits(MI, CPol::GLC | CPol::DLC);
    }

    // GFX11: Set MALL NOALLOC for both load and store instructions.
    if (AMDGPU::isGFX11(ST))
      Changed |= enableCPolBits(MI, CPol::DLC);

    // Ensure operation has completed at system scope to cause all volatile
    // operations to be visible outside the program in a global order. Do not
    // request cross address space as only the global address space can be
    // observable outside the program, so no need to cause a waitcnt for LDS
    // address space operations.
    Changed |= insertWait(MI, SIAtomicScope::SYSTEM, AddrSpace, Op, false,
                          Position::AFTER, AtomicOrdering::Unordered,
                          /*AtomicsOnly=*/false);
    return Changed;
  }

  if (IsNonTemporal) {
    // For loads setting SLC configures L0 and L1 cache policy to HIT_EVICT
    // and L2 cache policy to STREAM.
    // For stores setting both GLC and SLC configures L0 and L1 cache policy
    // to MISS_EVICT and the L2 cache policy to STREAM.
    if (Op == SIMemOp::STORE)
      Changed |= enableCPolBits(MI, CPol::GLC);
    Changed |= enableCPolBits(MI, CPol::SLC);

    // GFX11: Set MALL NOALLOC for both load and store instructions.
    if (AMDGPU::isGFX11(ST))
      Changed |= enableCPolBits(MI, CPol::DLC);

    return Changed;
  }

  return Changed;
}

bool SIGfx10CacheControl::insertWait(MachineBasicBlock::iterator &MI,
                                     SIAtomicScope Scope,
                                     SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                                     bool IsCrossAddrSpaceOrdering,
                                     Position Pos, AtomicOrdering Order,
                                     bool AtomicsOnly) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  bool VMCnt = false;
  bool VSCnt = false;
  bool LGKMCnt = false;

  if ((AddrSpace & (SIAtomicAddrSpace::GLOBAL | SIAtomicAddrSpace::SCRATCH)) !=
      SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
      if ((Op & SIMemOp::LOAD) != SIMemOp::NONE)
        VMCnt |= true;
      if ((Op & SIMemOp::STORE) != SIMemOp::NONE)
        VSCnt |= true;
      break;
    case SIAtomicScope::WORKGROUP:
      // In WGP mode the waves of a work-group can be executing on either CU of
      // the WGP. Therefore need to wait for operations to complete to ensure
      // they are visible to waves in the other CU as the L0 is per CU.
      // Otherwise in CU mode and all waves of a work-group are on the same CU
      // which shares the same L0. Note that we still need to wait when
      // performing a release in this mode to respect the transitivity of
      // happens-before, e.g. other waves of the workgroup must be able to
      // release the memory from another wave at a wider scope.
      if (!ST.isCuModeEnabled() || isReleaseOrStronger(Order)) {
        if ((Op & SIMemOp::LOAD) != SIMemOp::NONE)
          VMCnt |= true;
        if ((Op & SIMemOp::STORE) != SIMemOp::NONE)
          VSCnt |= true;
      }
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The L0 cache keeps all memory operations in order for
      // work-items in the same wavefront.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if ((AddrSpace & SIAtomicAddrSpace::LDS) != SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
    case SIAtomicScope::WORKGROUP:
      // If no cross address space ordering then an "S_WAITCNT lgkmcnt(0)" is
      // not needed as LDS operations for all waves are executed in a total
      // global ordering as observed by all waves. Required if also
      // synchronizing with global/GDS memory as LDS operations could be
      // reordered with respect to later global/GDS memory operations of the
      // same wave.
      LGKMCnt |= IsCrossAddrSpaceOrdering;
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The LDS keeps all memory operations in order for
      // the same wavefront.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if ((AddrSpace & SIAtomicAddrSpace::GDS) != SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
      // If no cross address space ordering then an GDS "S_WAITCNT lgkmcnt(0)"
      // is not needed as GDS operations for all waves are executed in a total
      // global ordering as observed by all waves. Required if also
      // synchronizing with global/LDS memory as GDS operations could be
      // reordered with respect to later global/LDS memory operations of the
      // same wave.
      LGKMCnt |= IsCrossAddrSpaceOrdering;
      break;
    case SIAtomicScope::WORKGROUP:
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The GDS keeps all memory operations in order for
      // the same work-group.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if (VMCnt || LGKMCnt) {
    unsigned WaitCntImmediate =
      AMDGPU::encodeWaitcnt(IV,
                            VMCnt ? 0 : getVmcntBitMask(IV),
                            getExpcntBitMask(IV),
                            LGKMCnt ? 0 : getLgkmcntBitMask(IV));
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAITCNT_soft))
        .addImm(WaitCntImmediate);
    Changed = true;
  }

  // On architectures that support direct loads to LDS, emit an unknown waitcnt
  // at workgroup-scoped release operations that specify the LDS address space.
  // SIInsertWaitcnts will later replace this with a vmcnt().
  if (ST.hasVMemToLDSLoad() && isReleaseOrStronger(Order) &&
      Scope == SIAtomicScope::WORKGROUP &&
      (AddrSpace & SIAtomicAddrSpace::LDS) != SIAtomicAddrSpace::NONE) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAITCNT_lds_direct));
    Changed = true;
  }

  if (VSCnt) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAITCNT_VSCNT_soft))
        .addReg(AMDGPU::SGPR_NULL, RegState::Undef)
        .addImm(0);
    Changed = true;
  }

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool SIGfx10CacheControl::insertAcquire(MachineBasicBlock::iterator &MI,
                                        SIAtomicScope Scope,
                                        SIAtomicAddrSpace AddrSpace,
                                        Position Pos) const {
  if (!InsertCacheInv)
    return false;

  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  if (Pos == Position::AFTER)
    ++MI;

  if (canAffectGlobalAddrSpace(AddrSpace)) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
      // The order of invalidates matter here. We must invalidate "outer in"
      // so L1 -> L0 to avoid L0 pulling in stale data from L1 when it is
      // invalidated.
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_GL1_INV));
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_GL0_INV));
      Changed = true;
      break;
    case SIAtomicScope::WORKGROUP:
      // In WGP mode the waves of a work-group can be executing on either CU of
      // the WGP. Therefore need to invalidate the L0 which is per CU. Otherwise
      // in CU mode and all waves of a work-group are on the same CU, and so the
      // L0 does not need to be invalidated.
      if (!ST.isCuModeEnabled()) {
        BuildMI(MBB, MI, DL, TII->get(AMDGPU::BUFFER_GL0_INV));
        Changed = true;
      }
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // No cache to invalidate.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  /// The scratch address space does not need the global memory cache
  /// to be flushed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not have a cache.

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool SIGfx12CacheControl::setTH(const MachineBasicBlock::iterator MI,
                                AMDGPU::CPol::CPol Value) const {
  MachineOperand *CPol = TII->getNamedOperand(*MI, OpName::cpol);
  if (!CPol)
    return false;

  uint64_t NewTH = Value & AMDGPU::CPol::TH;
  if ((CPol->getImm() & AMDGPU::CPol::TH) != NewTH) {
    CPol->setImm((CPol->getImm() & ~AMDGPU::CPol::TH) | NewTH);
    return true;
  }

  return false;
}

bool SIGfx12CacheControl::setScope(const MachineBasicBlock::iterator MI,
                                   AMDGPU::CPol::CPol Value) const {
  MachineOperand *CPol = TII->getNamedOperand(*MI, OpName::cpol);
  if (!CPol)
    return false;

  uint64_t NewScope = Value & AMDGPU::CPol::SCOPE;
  if ((CPol->getImm() & AMDGPU::CPol::SCOPE) != NewScope) {
    CPol->setImm((CPol->getImm() & ~AMDGPU::CPol::SCOPE) | NewScope);
    return true;
  }

  return false;
}

bool SIGfx12CacheControl::insertWaitsBeforeSystemScopeStore(
    const MachineBasicBlock::iterator MI) const {
  // TODO: implement flag for frontend to give us a hint not to insert waits.

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  BuildMI(MBB, MI, DL, TII->get(S_WAIT_LOADCNT_soft)).addImm(0);
  if (ST.hasImageInsts()) {
    BuildMI(MBB, MI, DL, TII->get(S_WAIT_SAMPLECNT_soft)).addImm(0);
    BuildMI(MBB, MI, DL, TII->get(S_WAIT_BVHCNT_soft)).addImm(0);
  }
  BuildMI(MBB, MI, DL, TII->get(S_WAIT_KMCNT_soft)).addImm(0);
  BuildMI(MBB, MI, DL, TII->get(S_WAIT_STORECNT_soft)).addImm(0);

  return true;
}

bool SIGfx12CacheControl::insertWait(MachineBasicBlock::iterator &MI,
                                     SIAtomicScope Scope,
                                     SIAtomicAddrSpace AddrSpace, SIMemOp Op,
                                     bool IsCrossAddrSpaceOrdering,
                                     Position Pos, AtomicOrdering Order,
                                     bool AtomicsOnly) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  bool LOADCnt = false;
  bool DSCnt = false;
  bool STORECnt = false;

  if (Pos == Position::AFTER)
    ++MI;

  if ((AddrSpace & (SIAtomicAddrSpace::GLOBAL | SIAtomicAddrSpace::SCRATCH)) !=
      SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
    case SIAtomicScope::CLUSTER:
      if ((Op & SIMemOp::LOAD) != SIMemOp::NONE)
        LOADCnt |= true;
      if ((Op & SIMemOp::STORE) != SIMemOp::NONE)
        STORECnt |= true;
      break;
    case SIAtomicScope::WORKGROUP:
      // GFX12.0:
      //   In WGP mode the waves of a work-group can be executing on either CU
      //   of the WGP. Therefore need to wait for operations to complete to
      //   ensure they are visible to waves in the other CU as the L0 is per CU.
      //
      //   Otherwise in CU mode and all waves of a work-group are on the same CU
      //   which shares the same L0. Note that we still need to wait when
      //   performing a release in this mode to respect the transitivity of
      //   happens-before, e.g. other waves of the workgroup must be able to
      //   release the memory from another wave at a wider scope.
      //
      // GFX12.5:
      //   CU$ has two ports. To ensure operations are visible at the workgroup
      //   level, we need to ensure all operations in this port have completed
      //   so the other SIMDs in the WG can see them. There is no ordering
      //   guarantee between the ports.
      if (!ST.isCuModeEnabled() || ST.hasGFX1250Insts() ||
          isReleaseOrStronger(Order)) {
        if ((Op & SIMemOp::LOAD) != SIMemOp::NONE)
          LOADCnt |= true;
        if ((Op & SIMemOp::STORE) != SIMemOp::NONE)
          STORECnt |= true;
      }
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The L0 cache keeps all memory operations in order for
      // work-items in the same wavefront.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if ((AddrSpace & SIAtomicAddrSpace::LDS) != SIAtomicAddrSpace::NONE) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
    case SIAtomicScope::AGENT:
    case SIAtomicScope::CLUSTER:
    case SIAtomicScope::WORKGROUP:
      // If no cross address space ordering then an "S_WAITCNT lgkmcnt(0)" is
      // not needed as LDS operations for all waves are executed in a total
      // global ordering as observed by all waves. Required if also
      // synchronizing with global/GDS memory as LDS operations could be
      // reordered with respect to later global/GDS memory operations of the
      // same wave.
      DSCnt |= IsCrossAddrSpaceOrdering;
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // The LDS keeps all memory operations in order for
      // the same wavefront.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  if (LOADCnt) {
    // Acquire sequences only need to wait on the previous atomic operation.
    // e.g. a typical sequence looks like
    //    atomic load
    //    (wait)
    //    global_inv
    //
    // We do not have BVH or SAMPLE atomics, so the atomic load is always going
    // to be tracked using loadcnt.
    //
    // This also applies to fences. Fences cannot pair with an instruction
    // tracked with bvh/samplecnt as we don't have any atomics that do that.
    if (!AtomicsOnly && ST.hasImageInsts()) {
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAIT_BVHCNT_soft)).addImm(0);
      BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAIT_SAMPLECNT_soft)).addImm(0);
    }
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAIT_LOADCNT_soft)).addImm(0);
    Changed = true;
  }

  if (STORECnt) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAIT_STORECNT_soft)).addImm(0);
    Changed = true;
  }

  if (DSCnt) {
    BuildMI(MBB, MI, DL, TII->get(AMDGPU::S_WAIT_DSCNT_soft)).addImm(0);
    Changed = true;
  }

  if (Pos == Position::AFTER)
    --MI;

  return Changed;
}

bool SIGfx12CacheControl::insertAcquire(MachineBasicBlock::iterator &MI,
                                        SIAtomicScope Scope,
                                        SIAtomicAddrSpace AddrSpace,
                                        Position Pos) const {
  if (!InsertCacheInv)
    return false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  /// The scratch address space does not need the global memory cache
  /// to be flushed as all memory operations by the same thread are
  /// sequentially consistent, and no other thread can access scratch
  /// memory.

  /// Other address spaces do not have a cache.
  if (!canAffectGlobalAddrSpace(AddrSpace))
    return false;

  AMDGPU::CPol::CPol ScopeImm = AMDGPU::CPol::SCOPE_DEV;
  switch (Scope) {
  case SIAtomicScope::SYSTEM:
    ScopeImm = AMDGPU::CPol::SCOPE_SYS;
    break;
  case SIAtomicScope::AGENT:
    ScopeImm = AMDGPU::CPol::SCOPE_DEV;
    break;
  case SIAtomicScope::CLUSTER:
    ScopeImm = AMDGPU::CPol::SCOPE_SE;
    break;
  case SIAtomicScope::WORKGROUP:
    // GFX12.0:
    //  In WGP mode the waves of a work-group can be executing on either CU of
    //  the WGP. Therefore we need to invalidate the L0 which is per CU.
    //  Otherwise in CU mode all waves of a work-group are on the same CU, and
    //  so the L0 does not need to be invalidated.
    //
    // GFX12.5 has a shared WGP$, so no invalidates are required.
    if (ST.isCuModeEnabled())
      return false;

    ScopeImm = AMDGPU::CPol::SCOPE_SE;
    break;
  case SIAtomicScope::WAVEFRONT:
  case SIAtomicScope::SINGLETHREAD:
    // No cache to invalidate.
    return false;
  default:
    llvm_unreachable("Unsupported synchronization scope");
  }

  if (Pos == Position::AFTER)
    ++MI;

  BuildMI(MBB, MI, DL, TII->get(AMDGPU::GLOBAL_INV)).addImm(ScopeImm);

  if (Pos == Position::AFTER)
    --MI;

  // Target requires a waitcnt to ensure that the proceeding INV has completed
  // as it may get reorded with following load instructions.
  if (ST.hasINVWBL2WaitCntRequirement() && Scope > SIAtomicScope::CLUSTER) {
    insertWait(MI, Scope, AddrSpace, SIMemOp::LOAD,
               /*IsCrossAddrSpaceOrdering=*/false, Pos, AtomicOrdering::Acquire,
               /*AtomicsOnly=*/false);

    if (Pos == Position::AFTER)
      --MI;
  }

  return true;
}

bool SIGfx12CacheControl::insertRelease(MachineBasicBlock::iterator &MI,
                                        SIAtomicScope Scope,
                                        SIAtomicAddrSpace AddrSpace,
                                        bool IsCrossAddrSpaceOrdering,
                                        Position Pos) const {
  bool Changed = false;

  MachineBasicBlock &MBB = *MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();

  // The scratch address space does not need the global memory cache
  // writeback as all memory operations by the same thread are
  // sequentially consistent, and no other thread can access scratch
  // memory.
  if (canAffectGlobalAddrSpace(AddrSpace)) {
    if (Pos == Position::AFTER)
      ++MI;

    // global_wb is only necessary at system scope for GFX12.0,
    // they're also necessary at device scope for GFX12.5 as stores
    // cannot report completion earlier than L2.
    //
    // Emitting it for lower scopes is a slow no-op, so we omit it
    // for performance.
    std::optional<AMDGPU::CPol::CPol> NeedsWB;
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
      NeedsWB = AMDGPU::CPol::SCOPE_SYS;
      break;
    case SIAtomicScope::AGENT:
      // GFX12.5 may have >1 L2 per device so we must emit a device scope WB.
      if (ST.hasGFX1250Insts())
        NeedsWB = AMDGPU::CPol::SCOPE_DEV;
      break;
    case SIAtomicScope::CLUSTER:
    case SIAtomicScope::WORKGROUP:
      // No WB necessary, but we still have to wait.
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // No WB or wait necessary here, but insertWait takes care of that.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }

    if (NeedsWB) {
      // Target requires a waitcnt to ensure that the proceeding store
      // proceeding store/rmw operations have completed in L2 so their data will
      // be written back by the WB instruction.
      if (ST.hasINVWBL2WaitCntRequirement())
        insertWait(MI, Scope, AddrSpace, SIMemOp::LOAD | SIMemOp::STORE,
                   /*IsCrossAddrSpaceOrdering=*/false, Pos,
                   AtomicOrdering::Release,
                   /*AtomicsOnly=*/false);

      BuildMI(MBB, MI, DL, TII->get(AMDGPU::GLOBAL_WB)).addImm(*NeedsWB);
      Changed = true;
    }

    if (Pos == Position::AFTER)
      --MI;
  }

  // We always have to wait for previous memory operations (load/store) to
  // complete, whether we inserted a WB or not. If we inserted a WB (storecnt),
  // we of course need to wait for that as well.
  Changed |= insertWait(MI, Scope, AddrSpace, SIMemOp::LOAD | SIMemOp::STORE,
                        IsCrossAddrSpaceOrdering, Pos, AtomicOrdering::Release,
                        /*AtomicsOnly=*/false);

  return Changed;
}

bool SIGfx12CacheControl::enableVolatileAndOrNonTemporal(
    MachineBasicBlock::iterator &MI, SIAtomicAddrSpace AddrSpace, SIMemOp Op,
    bool IsVolatile, bool IsNonTemporal, bool IsLastUse = false) const {

  // Only handle load and store, not atomic read-modify-write instructions.
  assert((MI->mayLoad() ^ MI->mayStore()) || SIInstrInfo::isLDSDMA(*MI));

  // Only update load and store, not LLVM IR atomic read-modify-write
  // instructions. The latter are always marked as volatile so cannot sensibly
  // handle it as do not want to pessimize all atomics. Also they do not support
  // the nontemporal attribute.
  assert(Op == SIMemOp::LOAD || Op == SIMemOp::STORE);

  bool Changed = false;

  if (IsLastUse) {
    // Set last-use hint.
    Changed |= setTH(MI, AMDGPU::CPol::TH_LU);
  } else if (IsNonTemporal) {
    // Set non-temporal hint for all cache levels.
    Changed |= setTH(MI, AMDGPU::CPol::TH_NT);
  }

  if (IsVolatile) {
    Changed |= setScope(MI, AMDGPU::CPol::SCOPE_SYS);

    if (ST.requiresWaitXCntForSingleAccessInstructions() &&
        SIInstrInfo::isVMEM(*MI)) {
      MachineBasicBlock &MBB = *MI->getParent();
      BuildMI(MBB, MI, MI->getDebugLoc(), TII->get(S_WAIT_XCNT_soft)).addImm(0);
      Changed = true;
    }

    // Ensure operation has completed at system scope to cause all volatile
    // operations to be visible outside the program in a global order. Do not
    // request cross address space as only the global address space can be
    // observable outside the program, so no need to cause a waitcnt for LDS
    // address space operations.
    Changed |= insertWait(MI, SIAtomicScope::SYSTEM, AddrSpace, Op, false,
                          Position::AFTER, AtomicOrdering::Unordered,
                          /*AtomicsOnly=*/false);
  }

  return Changed;
}

bool SIGfx12CacheControl::finalizeStore(MachineInstr &MI, bool Atomic) const {
  assert(MI.mayStore() && "Not a Store inst");
  const bool IsRMW = (MI.mayLoad() && MI.mayStore());
  bool Changed = false;

  if (Atomic && ST.requiresWaitXCntForSingleAccessInstructions() &&
      SIInstrInfo::isVMEM(MI)) {
    MachineBasicBlock &MBB = *MI.getParent();
    BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(S_WAIT_XCNT_soft)).addImm(0);
    Changed = true;
  }

  // Remaining fixes do not apply to RMWs.
  if (IsRMW)
    return Changed;

  MachineOperand *CPol = TII->getNamedOperand(MI, OpName::cpol);
  if (!CPol) // Some vmem operations do not have a scope and are not concerned.
    return Changed;
  const unsigned Scope = CPol->getImm() & CPol::SCOPE;

  // GFX12.0 only: Extra waits needed before system scope stores.
  if (ST.requiresWaitsBeforeSystemScopeStores() && !Atomic &&
      Scope == CPol::SCOPE_SYS)
    Changed |= insertWaitsBeforeSystemScopeStore(MI.getIterator());

  return Changed;
}

bool SIGfx12CacheControl::handleCooperativeAtomic(MachineInstr &MI) const {
  if (!ST.hasGFX1250Insts())
    return false;

  // Cooperative atomics need to be SCOPE_DEV or higher.
  MachineOperand *CPol = TII->getNamedOperand(MI, OpName::cpol);
  assert(CPol && "No CPol operand?");
  const unsigned Scope = CPol->getImm() & CPol::SCOPE;
  if (Scope < CPol::SCOPE_DEV)
    return setScope(MI, CPol::SCOPE_DEV);
  return false;
}

bool SIGfx12CacheControl::setAtomicScope(const MachineBasicBlock::iterator &MI,
                                         SIAtomicScope Scope,
                                         SIAtomicAddrSpace AddrSpace) const {
  bool Changed = false;

  if (canAffectGlobalAddrSpace(AddrSpace)) {
    switch (Scope) {
    case SIAtomicScope::SYSTEM:
      Changed |= setScope(MI, AMDGPU::CPol::SCOPE_SYS);
      break;
    case SIAtomicScope::AGENT:
      Changed |= setScope(MI, AMDGPU::CPol::SCOPE_DEV);
      break;
    case SIAtomicScope::CLUSTER:
      Changed |= setScope(MI, AMDGPU::CPol::SCOPE_SE);
      break;
    case SIAtomicScope::WORKGROUP:
      // In workgroup mode, SCOPE_SE is needed as waves can executes on
      // different CUs that access different L0s.
      if (!ST.isCuModeEnabled())
        Changed |= setScope(MI, AMDGPU::CPol::SCOPE_SE);
      break;
    case SIAtomicScope::WAVEFRONT:
    case SIAtomicScope::SINGLETHREAD:
      // No cache to bypass.
      break;
    default:
      llvm_unreachable("Unsupported synchronization scope");
    }
  }

  // The scratch address space does not need the global memory caches
  // to be bypassed as all memory operations by the same thread are
  // sequentially consistent, and no other thread can access scratch
  // memory.

  // Other address spaces do not have a cache.

  return Changed;
}

bool SIMemoryLegalizer::removeAtomicPseudoMIs() {
  if (AtomicPseudoMIs.empty())
    return false;

  for (auto &MI : AtomicPseudoMIs)
    MI->eraseFromParent();

  AtomicPseudoMIs.clear();
  return true;
}

bool SIMemoryLegalizer::expandLoad(const SIMemOpInfo &MOI,
                                   MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && !MI->mayStore());

  bool Changed = false;

  if (MOI.isAtomic()) {
    const AtomicOrdering Order = MOI.getOrdering();
    if (Order == AtomicOrdering::Monotonic ||
        Order == AtomicOrdering::Acquire ||
        Order == AtomicOrdering::SequentiallyConsistent) {
      Changed |= CC->enableLoadCacheBypass(MI, MOI.getScope(),
                                           MOI.getOrderingAddrSpace());
    }

    // Handle cooperative atomics after cache bypass step, as it may override
    // the scope of the instruction to a greater scope.
    if (MOI.isCooperative())
      Changed |= CC->handleCooperativeAtomic(*MI);

    if (Order == AtomicOrdering::SequentiallyConsistent)
      Changed |= CC->insertWait(MI, MOI.getScope(), MOI.getOrderingAddrSpace(),
                                SIMemOp::LOAD | SIMemOp::STORE,
                                MOI.getIsCrossAddressSpaceOrdering(),
                                Position::BEFORE, Order, /*AtomicsOnly=*/false);

    if (Order == AtomicOrdering::Acquire ||
        Order == AtomicOrdering::SequentiallyConsistent) {
      // The wait below only needs to wait on the prior atomic.
      Changed |=
          CC->insertWait(MI, MOI.getScope(), MOI.getInstrAddrSpace(),
                         SIMemOp::LOAD, MOI.getIsCrossAddressSpaceOrdering(),
                         Position::AFTER, Order, /*AtomicsOnly=*/true);
      Changed |= CC->insertAcquire(MI, MOI.getScope(),
                                   MOI.getOrderingAddrSpace(),
                                   Position::AFTER);
    }

    return Changed;
  }

  // Atomic instructions already bypass caches to the scope specified by the
  // SyncScope operand. Only non-atomic volatile and nontemporal/last-use
  // instructions need additional treatment.
  Changed |= CC->enableVolatileAndOrNonTemporal(
      MI, MOI.getInstrAddrSpace(), SIMemOp::LOAD, MOI.isVolatile(),
      MOI.isNonTemporal(), MOI.isLastUse());

  return Changed;
}

bool SIMemoryLegalizer::expandStore(const SIMemOpInfo &MOI,
                                    MachineBasicBlock::iterator &MI) {
  assert(!MI->mayLoad() && MI->mayStore());

  bool Changed = false;
  // FIXME: Necessary hack because iterator can lose track of the store.
  MachineInstr &StoreMI = *MI;

  if (MOI.isAtomic()) {
    if (MOI.getOrdering() == AtomicOrdering::Monotonic ||
        MOI.getOrdering() == AtomicOrdering::Release ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent) {
      Changed |= CC->enableStoreCacheBypass(MI, MOI.getScope(),
                                            MOI.getOrderingAddrSpace());
    }

    // Handle cooperative atomics after cache bypass step, as it may override
    // the scope of the instruction to a greater scope.
    if (MOI.isCooperative())
      Changed |= CC->handleCooperativeAtomic(*MI);

    if (MOI.getOrdering() == AtomicOrdering::Release ||
        MOI.getOrdering() == AtomicOrdering::SequentiallyConsistent)
      Changed |= CC->insertRelease(MI, MOI.getScope(),
                                   MOI.getOrderingAddrSpace(),
                                   MOI.getIsCrossAddressSpaceOrdering(),
                                   Position::BEFORE);

    Changed |= CC->finalizeStore(StoreMI, /*Atomic=*/true);
    return Changed;
  }

  // Atomic instructions already bypass caches to the scope specified by the
  // SyncScope operand. Only non-atomic volatile and nontemporal instructions
  // need additional treatment.
  Changed |= CC->enableVolatileAndOrNonTemporal(
      MI, MOI.getInstrAddrSpace(), SIMemOp::STORE, MOI.isVolatile(),
      MOI.isNonTemporal());

  // GFX12 specific, scope(desired coherence domain in cache hierarchy) is
  // instruction field, do not confuse it with atomic scope.
  Changed |= CC->finalizeStore(StoreMI, /*Atomic=*/false);
  return Changed;
}

bool SIMemoryLegalizer::expandAtomicFence(const SIMemOpInfo &MOI,
                                          MachineBasicBlock::iterator &MI) {
  assert(MI->getOpcode() == AMDGPU::ATOMIC_FENCE);

  AtomicPseudoMIs.push_back(MI);
  bool Changed = false;

  const SIAtomicAddrSpace OrderingAddrSpace = MOI.getOrderingAddrSpace();

  if (MOI.isAtomic()) {
    const AtomicOrdering Order = MOI.getOrdering();
    if (Order == AtomicOrdering::Acquire) {
      // Acquire fences only need to wait on the previous atomic they pair with.
      Changed |= CC->insertWait(MI, MOI.getScope(), OrderingAddrSpace,
                                SIMemOp::LOAD | SIMemOp::STORE,
                                MOI.getIsCrossAddressSpaceOrdering(),
                                Position::BEFORE, Order, /*AtomicsOnly=*/true);
    }

    if (Order == AtomicOrdering::Release ||
        Order == AtomicOrdering::AcquireRelease ||
        Order == AtomicOrdering::SequentiallyConsistent)
      /// TODO: This relies on a barrier always generating a waitcnt
      /// for LDS to ensure it is not reordered with the completion of
      /// the proceeding LDS operations. If barrier had a memory
      /// ordering and memory scope, then library does not need to
      /// generate a fence. Could add support in this file for
      /// barrier. SIInsertWaitcnt.cpp could then stop unconditionally
      /// adding S_WAITCNT before a S_BARRIER.
      Changed |= CC->insertRelease(MI, MOI.getScope(), OrderingAddrSpace,
                                   MOI.getIsCrossAddressSpaceOrdering(),
                                   Position::BEFORE);

    // TODO: If both release and invalidate are happening they could be combined
    // to use the single "BUFFER_WBINV*" instruction. This could be done by
    // reorganizing this code or as part of optimizing SIInsertWaitcnt pass to
    // track cache invalidate and write back instructions.

    if (Order == AtomicOrdering::Acquire ||
        Order == AtomicOrdering::AcquireRelease ||
        Order == AtomicOrdering::SequentiallyConsistent)
      Changed |= CC->insertAcquire(MI, MOI.getScope(), OrderingAddrSpace,
                                   Position::BEFORE);

    return Changed;
  }

  return Changed;
}

bool SIMemoryLegalizer::expandAtomicCmpxchgOrRmw(const SIMemOpInfo &MOI,
  MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && MI->mayStore());

  bool Changed = false;
  MachineInstr &RMWMI = *MI;

  if (MOI.isAtomic()) {
    const AtomicOrdering Order = MOI.getOrdering();
    if (Order == AtomicOrdering::Monotonic ||
        Order == AtomicOrdering::Acquire || Order == AtomicOrdering::Release ||
        Order == AtomicOrdering::AcquireRelease ||
        Order == AtomicOrdering::SequentiallyConsistent) {
      Changed |= CC->enableRMWCacheBypass(MI, MOI.getScope(),
                                          MOI.getInstrAddrSpace());
    }

    if (Order == AtomicOrdering::Release ||
        Order == AtomicOrdering::AcquireRelease ||
        Order == AtomicOrdering::SequentiallyConsistent ||
        MOI.getFailureOrdering() == AtomicOrdering::SequentiallyConsistent)
      Changed |= CC->insertRelease(MI, MOI.getScope(),
                                   MOI.getOrderingAddrSpace(),
                                   MOI.getIsCrossAddressSpaceOrdering(),
                                   Position::BEFORE);

    if (Order == AtomicOrdering::Acquire ||
        Order == AtomicOrdering::AcquireRelease ||
        Order == AtomicOrdering::SequentiallyConsistent ||
        MOI.getFailureOrdering() == AtomicOrdering::Acquire ||
        MOI.getFailureOrdering() == AtomicOrdering::SequentiallyConsistent) {
      // Only wait on the previous atomic.
      Changed |=
          CC->insertWait(MI, MOI.getScope(), MOI.getInstrAddrSpace(),
                         isAtomicRet(*MI) ? SIMemOp::LOAD : SIMemOp::STORE,
                         MOI.getIsCrossAddressSpaceOrdering(), Position::AFTER,
                         Order, /*AtomicsOnly=*/true);
      Changed |= CC->insertAcquire(MI, MOI.getScope(),
                                   MOI.getOrderingAddrSpace(),
                                   Position::AFTER);
    }

    Changed |= CC->finalizeStore(RMWMI, /*Atomic=*/true);
    return Changed;
  }

  return Changed;
}

bool SIMemoryLegalizer::expandLDSDMA(const SIMemOpInfo &MOI,
                                     MachineBasicBlock::iterator &MI) {
  assert(MI->mayLoad() && MI->mayStore());

  // The volatility or nontemporal-ness of the operation is a
  // function of the global memory, not the LDS.
  SIMemOp OpKind =
      SIInstrInfo::mayWriteLDSThroughDMA(*MI) ? SIMemOp::LOAD : SIMemOp::STORE;

  // Handle volatile and/or nontemporal markers on direct-to-LDS loads and
  // stores. The operation is treated as a volatile/nontemporal store
  // to its second argument.
  return CC->enableVolatileAndOrNonTemporal(
      MI, MOI.getInstrAddrSpace(), OpKind, MOI.isVolatile(),
      MOI.isNonTemporal(), MOI.isLastUse());
}

bool SIMemoryLegalizerLegacy::runOnMachineFunction(MachineFunction &MF) {
  const MachineModuleInfo &MMI =
      getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  return SIMemoryLegalizer(MMI).run(MF);
}

PreservedAnalyses
SIMemoryLegalizerPass::run(MachineFunction &MF,
                           MachineFunctionAnalysisManager &MFAM) {
  auto *MMI = MFAM.getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
                  .getCachedResult<MachineModuleAnalysis>(
                      *MF.getFunction().getParent());
  assert(MMI && "MachineModuleAnalysis must be available");
  if (!SIMemoryLegalizer(MMI->getMMI()).run(MF))
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses().preserveSet<CFGAnalyses>();
}

bool SIMemoryLegalizer::run(MachineFunction &MF) {
  bool Changed = false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  SIMemOpAccess MOA(MMI.getObjFileInfo<AMDGPUMachineModuleInfo>(), ST);
  CC = SICacheControl::create(ST);

  for (auto &MBB : MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {

      // Unbundle instructions after the post-RA scheduler.
      if (MI->isBundle() && MI->mayLoadOrStore()) {
        MachineBasicBlock::instr_iterator II(MI->getIterator());
        for (MachineBasicBlock::instr_iterator I = ++II, E = MBB.instr_end();
             I != E && I->isBundledWithPred(); ++I) {
          I->unbundleFromPred();
          for (MachineOperand &MO : I->operands())
            if (MO.isReg())
              MO.setIsInternalRead(false);
        }

        MI->eraseFromParent();
        MI = II->getIterator();
      }

      if (!(MI->getDesc().TSFlags & SIInstrFlags::maybeAtomic))
        continue;

      if (const auto &MOI = MOA.getLoadInfo(MI)) {
        Changed |= expandLoad(*MOI, MI);
      } else if (const auto &MOI = MOA.getStoreInfo(MI)) {
        Changed |= expandStore(*MOI, MI);
      } else if (const auto &MOI = MOA.getLDSDMAInfo(MI)) {
        Changed |= expandLDSDMA(*MOI, MI);
      } else if (const auto &MOI = MOA.getAtomicFenceInfo(MI)) {
        Changed |= expandAtomicFence(*MOI, MI);
      } else if (const auto &MOI = MOA.getAtomicCmpxchgOrRmwInfo(MI)) {
        Changed |= expandAtomicCmpxchgOrRmw(*MOI, MI);
      }
    }
  }

  Changed |= removeAtomicPseudoMIs();
  return Changed;
}

INITIALIZE_PASS(SIMemoryLegalizerLegacy, DEBUG_TYPE, PASS_NAME, false, false)

char SIMemoryLegalizerLegacy::ID = 0;
char &llvm::SIMemoryLegalizerID = SIMemoryLegalizerLegacy::ID;

FunctionPass *llvm::createSIMemoryLegalizerPass() {
  return new SIMemoryLegalizerLegacy();
}
