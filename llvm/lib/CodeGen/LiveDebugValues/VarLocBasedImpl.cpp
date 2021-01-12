//===- VarLocBasedImpl.cpp - Tracking Debug Value MIs with VarLoc class----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file VarLocBasedImpl.cpp
///
/// LiveDebugValues is an optimistic "available expressions" dataflow
/// algorithm. The set of expressions is the set of machine locations
/// (registers, spill slots, constants) that a variable fragment might be
/// located, qualified by a DIExpression and indirect-ness flag, while each
/// variable is identified by a DebugVariable object. The availability of an
/// expression begins when a DBG_VALUE instruction specifies the location of a
/// DebugVariable, and continues until that location is clobbered or
/// re-specified by a different DBG_VALUE for the same DebugVariable.
///
/// The output of LiveDebugValues is additional DBG_VALUE instructions,
/// placed to extend variable locations as far they're available. This file
/// and the VarLocBasedLDV class is an implementation that explicitly tracks
/// locations, using the VarLoc class.
///
/// The canonical "available expressions" problem doesn't have expression
/// clobbering, instead when a variable is re-assigned, any expressions using
/// that variable get invalidated. LiveDebugValues can map onto "available
/// expressions" by having every register represented by a variable, which is
/// used in an expression that becomes available at a DBG_VALUE instruction.
/// When the register is clobbered, its variable is effectively reassigned, and
/// expressions computed from it become unavailable. A similar construct is
/// needed when a DebugVariable has its location re-specified, to invalidate
/// all other locations for that DebugVariable.
///
/// Using the dataflow analysis to compute the available expressions, we create
/// a DBG_VALUE at the beginning of each block where the expression is
/// live-in. This propagates variable locations into every basic block where
/// the location can be determined, rather than only having DBG_VALUEs in blocks
/// where locations are specified due to an assignment or some optimization.
/// Movements of values between registers and spill slots are annotated with
/// DBG_VALUEs too to track variable values bewteen locations. All this allows
/// DbgEntityHistoryCalculator to focus on only the locations within individual
/// blocks, facilitating testing and improving modularity.
///
/// We follow an optimisic dataflow approach, with this lattice:
///
/// \verbatim
///                    ┬ "Unknown"
///                          |
///                          v
///                         True
///                          |
///                          v
///                      ⊥ False
/// \endverbatim With "True" signifying that the expression is available (and
/// thus a DebugVariable's location is the corresponding register), while
/// "False" signifies that the expression is unavailable. "Unknown"s never
/// survive to the end of the analysis (see below).
///
/// Formally, all DebugVariable locations that are live-out of a block are
/// initialized to \top.  A blocks live-in values take the meet of the lattice
/// value for every predecessors live-outs, except for the entry block, where
/// all live-ins are \bot. The usual dataflow propagation occurs: the transfer
/// function for a block assigns an expression for a DebugVariable to be "True"
/// if a DBG_VALUE in the block specifies it; "False" if the location is
/// clobbered; or the live-in value if it is unaffected by the block. We
/// visit each block in reverse post order until a fixedpoint is reached. The
/// solution produced is maximal.
///
/// Intuitively, we start by assuming that every expression / variable location
/// is at least "True", and then propagate "False" from the entry block and any
/// clobbers until there are no more changes to make. This gives us an accurate
/// solution because all incorrect locations will have a "False" propagated into
/// them. It also gives us a solution that copes well with loops by assuming
/// that variable locations are live-through every loop, and then removing those
/// that are not through dataflow.
///
/// Within LiveDebugValues: each variable location is represented by a
/// VarLoc object that identifies the source variable, its current
/// machine-location, and the DBG_VALUE inst that specifies the location. Each
/// VarLoc is indexed in the (function-scope) \p VarLocMap, giving each VarLoc a
/// unique index. Rather than operate directly on machine locations, the
/// dataflow analysis in this pass identifies locations by their index in the
/// VarLocMap, meaning all the variable locations in a block can be described
/// by a sparse vector of VarLocMap indicies.
///
/// All the storage for the dataflow analysis is local to the ExtendRanges
/// method and passed down to helper methods. "OutLocs" and "InLocs" record the
/// in and out lattice values for each block. "OpenRanges" maintains a list of
/// variable locations and, with the "process" method, evaluates the transfer
/// function of each block. "flushPendingLocs" installs DBG_VALUEs for each
/// live-in location at the start of blocks, while "Transfers" records
/// transfers of values between machine-locations.
///
/// We avoid explicitly representing the "Unknown" (\top) lattice value in the
/// implementation. Instead, unvisited blocks implicitly have all lattice
/// values set as "Unknown". After being visited, there will be path back to
/// the entry block where the lattice value is "False", and as the transfer
/// function cannot make new "Unknown" locations, there are no scenarios where
/// a block can have an "Unknown" location after being visited. Similarly, we
/// don't enumerate all possible variable locations before exploring the
/// function: when a new location is discovered, all blocks previously explored
/// were implicitly "False" but unrecorded, and become explicitly "False" when
/// a new VarLoc is created with its bit not set in predecessor InLocs or
/// OutLocs.
///
//===----------------------------------------------------------------------===//

#include "LiveDebugValues.h"

#include "llvm/ADT/CoalescingBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");

// Options to prevent pathological compile-time behavior. If InputBBLimit and
// InputDbgValueLimit are both exceeded, range extension is disabled.
static cl::opt<unsigned> InputBBLimit(
    "livedebugvalues-input-bb-limit",
    cl::desc("Maximum input basic blocks before DBG_VALUE limit applies"),
    cl::init(10000), cl::Hidden);
static cl::opt<unsigned> InputDbgValueLimit(
    "livedebugvalues-input-dbg-value-limit",
    cl::desc(
        "Maximum input DBG_VALUE insts supported by debug range extension"),
    cl::init(50000), cl::Hidden);

// If @MI is a DBG_VALUE with debug value described by a defined
// register, returns the number of this register. In the other case, returns 0.
static Register isDbgValueDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue() && "expected a DBG_VALUE");
  assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
  // If location of variable is described using a register (directly
  // or indirectly), this register is always a first operand.
  return MI.getDebugOperand(0).isReg() ? MI.getDebugOperand(0).getReg()
                                       : Register();
}

/// If \p Op is a stack or frame register return true, otherwise return false.
/// This is used to avoid basing the debug entry values on the registers, since
/// we do not support it at the moment.
static bool isRegOtherThanSPAndFP(const MachineOperand &Op,
                                  const MachineInstr &MI,
                                  const TargetRegisterInfo *TRI) {
  if (!Op.isReg())
    return false;

  const MachineFunction *MF = MI.getParent()->getParent();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  Register SP = TLI->getStackPointerRegisterToSaveRestore();
  Register FP = TRI->getFrameRegister(*MF);
  Register Reg = Op.getReg();

  return Reg && Reg != SP && Reg != FP;
}

namespace {

// Max out the number of statically allocated elements in DefinedRegsSet, as
// this prevents fallback to std::set::count() operations.
using DefinedRegsSet = SmallSet<Register, 32>;

using VarLocSet = CoalescingBitVector<uint64_t>;

/// A type-checked pair of {Register Location (or 0), Index}, used to index
/// into a \ref VarLocMap. This can be efficiently converted to a 64-bit int
/// for insertion into a \ref VarLocSet, and efficiently converted back. The
/// type-checker helps ensure that the conversions aren't lossy.
///
/// Why encode a location /into/ the VarLocMap index? This makes it possible
/// to find the open VarLocs killed by a register def very quickly. This is a
/// performance-critical operation for LiveDebugValues.
struct LocIndex {
  using u32_location_t = uint32_t;
  using u32_index_t = uint32_t;

  u32_location_t Location; // Physical registers live in the range [1;2^30) (see
                           // \ref MCRegister), so we have plenty of range left
                           // here to encode non-register locations.
  u32_index_t Index;

  /// The first location greater than 0 that is not reserved for VarLocs of
  /// kind RegisterKind.
  static constexpr u32_location_t kFirstInvalidRegLocation = 1 << 30;

  /// A special location reserved for VarLocs of kind SpillLocKind.
  static constexpr u32_location_t kSpillLocation = kFirstInvalidRegLocation;

  /// A special location reserved for VarLocs of kind EntryValueBackupKind and
  /// EntryValueCopyBackupKind.
  static constexpr u32_location_t kEntryValueBackupLocation =
      kFirstInvalidRegLocation + 1;

  LocIndex(u32_location_t Location, u32_index_t Index)
      : Location(Location), Index(Index) {}

  uint64_t getAsRawInteger() const {
    return (static_cast<uint64_t>(Location) << 32) | Index;
  }

  template<typename IntT> static LocIndex fromRawInteger(IntT ID) {
    static_assert(std::is_unsigned<IntT>::value &&
                      sizeof(ID) == sizeof(uint64_t),
                  "Cannot convert raw integer to LocIndex");
    return {static_cast<u32_location_t>(ID >> 32),
            static_cast<u32_index_t>(ID)};
  }

  /// Get the start of the interval reserved for VarLocs of kind RegisterKind
  /// which reside in \p Reg. The end is at rawIndexForReg(Reg+1)-1.
  static uint64_t rawIndexForReg(uint32_t Reg) {
    return LocIndex(Reg, 0).getAsRawInteger();
  }

  /// Return a range covering all set indices in the interval reserved for
  /// \p Location in \p Set.
  static auto indexRangeForLocation(const VarLocSet &Set,
                                    u32_location_t Location) {
    uint64_t Start = LocIndex(Location, 0).getAsRawInteger();
    uint64_t End = LocIndex(Location + 1, 0).getAsRawInteger();
    return Set.half_open_range(Start, End);
  }
};

class VarLocBasedLDV : public LDVImpl {
private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  TargetPassConfig *TPC;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;
  VarLocSet::Allocator Alloc;

  enum struct TransferKind { TransferCopy, TransferSpill, TransferRestore };

  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  /// A pair of debug variable and value location.
  struct VarLoc {
    // The location at which a spilled variable resides. It consists of a
    // register and an offset.
    struct SpillLoc {
      unsigned SpillBase;
      StackOffset SpillOffset;
      bool operator==(const SpillLoc &Other) const {
        return SpillBase == Other.SpillBase && SpillOffset == Other.SpillOffset;
      }
      bool operator!=(const SpillLoc &Other) const {
        return !(*this == Other);
      }
    };

    /// Identity of the variable at this location.
    const DebugVariable Var;

    /// The expression applied to this location.
    const DIExpression *Expr;

    /// DBG_VALUE to clone var/expr information from if this location
    /// is moved.
    const MachineInstr &MI;

    enum VarLocKind {
      InvalidKind = 0,
      RegisterKind,
      SpillLocKind,
      ImmediateKind,
      EntryValueKind,
      EntryValueBackupKind,
      EntryValueCopyBackupKind
    } Kind = InvalidKind;

    /// The value location. Stored separately to avoid repeatedly
    /// extracting it from MI.
    union LocUnion {
      uint64_t RegNo;
      SpillLoc SpillLocation;
      uint64_t Hash;
      int64_t Immediate;
      const ConstantFP *FPImm;
      const ConstantInt *CImm;
      LocUnion() : Hash(0) {}
    } Loc;

    VarLoc(const MachineInstr &MI, LexicalScopes &LS)
        : Var(MI.getDebugVariable(), MI.getDebugExpression(),
              MI.getDebugLoc()->getInlinedAt()),
          Expr(MI.getDebugExpression()), MI(MI) {
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      if (int RegNo = isDbgValueDescribedByReg(MI)) {
        Kind = RegisterKind;
        Loc.RegNo = RegNo;
      } else if (MI.getDebugOperand(0).isImm()) {
        Kind = ImmediateKind;
        Loc.Immediate = MI.getDebugOperand(0).getImm();
      } else if (MI.getDebugOperand(0).isFPImm()) {
        Kind = ImmediateKind;
        Loc.FPImm = MI.getDebugOperand(0).getFPImm();
      } else if (MI.getDebugOperand(0).isCImm()) {
        Kind = ImmediateKind;
        Loc.CImm = MI.getDebugOperand(0).getCImm();
      }

      // We create the debug entry values from the factory functions rather than
      // from this ctor.
      assert(Kind != EntryValueKind && !isEntryBackupLoc());
    }

    /// Take the variable and machine-location in DBG_VALUE MI, and build an
    /// entry location using the given expression.
    static VarLoc CreateEntryLoc(const MachineInstr &MI, LexicalScopes &LS,
                                 const DIExpression *EntryExpr, Register Reg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = EntryValueKind;
      VL.Expr = EntryExpr;
      VL.Loc.RegNo = Reg;
      return VL;
    }

    /// Take the variable and machine-location from the DBG_VALUE (from the
    /// function entry), and build an entry value backup location. The backup
    /// location will turn into the normal location if the backup is valid at
    /// the time of the primary location clobbering.
    static VarLoc CreateEntryBackupLoc(const MachineInstr &MI,
                                       LexicalScopes &LS,
                                       const DIExpression *EntryExpr) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = EntryValueBackupKind;
      VL.Expr = EntryExpr;
      return VL;
    }

    /// Take the variable and machine-location from the DBG_VALUE (from the
    /// function entry), and build a copy of an entry value backup location by
    /// setting the register location to NewReg.
    static VarLoc CreateEntryCopyBackupLoc(const MachineInstr &MI,
                                           LexicalScopes &LS,
                                           const DIExpression *EntryExpr,
                                           Register NewReg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = EntryValueCopyBackupKind;
      VL.Expr = EntryExpr;
      VL.Loc.RegNo = NewReg;
      return VL;
    }

    /// Copy the register location in DBG_VALUE MI, updating the register to
    /// be NewReg.
    static VarLoc CreateCopyLoc(const MachineInstr &MI, LexicalScopes &LS,
                                Register NewReg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Loc.RegNo = NewReg;
      return VL;
    }

    /// Take the variable described by DBG_VALUE MI, and create a VarLoc
    /// locating it in the specified spill location.
    static VarLoc CreateSpillLoc(const MachineInstr &MI, unsigned SpillBase,
                                 StackOffset SpillOffset, LexicalScopes &LS) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = SpillLocKind;
      VL.Loc.SpillLocation = {SpillBase, SpillOffset};
      return VL;
    }

    /// Create a DBG_VALUE representing this VarLoc in the given function.
    /// Copies variable-specific information such as DILocalVariable and
    /// inlining information from the original DBG_VALUE instruction, which may
    /// have been several transfers ago.
    MachineInstr *BuildDbgValue(MachineFunction &MF) const {
      const DebugLoc &DbgLoc = MI.getDebugLoc();
      bool Indirect = MI.isIndirectDebugValue();
      const auto &IID = MI.getDesc();
      const DILocalVariable *Var = MI.getDebugVariable();
      const DIExpression *DIExpr = MI.getDebugExpression();
      NumInserted++;

      switch (Kind) {
      case EntryValueKind:
        // An entry value is a register location -- but with an updated
        // expression. The register location of such DBG_VALUE is always the one
        // from the entry DBG_VALUE, it does not matter if the entry value was
        // copied in to another register due to some optimizations.
        return BuildMI(MF, DbgLoc, IID, Indirect,
                       MI.getDebugOperand(0).getReg(), Var, Expr);
      case RegisterKind:
        // Register locations are like the source DBG_VALUE, but with the
        // register number from this VarLoc.
        return BuildMI(MF, DbgLoc, IID, Indirect, Loc.RegNo, Var, DIExpr);
      case SpillLocKind: {
        // Spills are indirect DBG_VALUEs, with a base register and offset.
        // Use the original DBG_VALUEs expression to build the spilt location
        // on top of. FIXME: spill locations created before this pass runs
        // are not recognized, and not handled here.
        auto *TRI = MF.getSubtarget().getRegisterInfo();
        auto *SpillExpr = TRI->prependOffsetExpression(
            DIExpr, DIExpression::ApplyOffset, Loc.SpillLocation.SpillOffset);
        unsigned Base = Loc.SpillLocation.SpillBase;
        return BuildMI(MF, DbgLoc, IID, true, Base, Var, SpillExpr);
      }
      case ImmediateKind: {
        MachineOperand MO = MI.getDebugOperand(0);
        return BuildMI(MF, DbgLoc, IID, Indirect, MO, Var, DIExpr);
      }
      case EntryValueBackupKind:
      case EntryValueCopyBackupKind:
      case InvalidKind:
        llvm_unreachable(
            "Tried to produce DBG_VALUE for invalid or backup VarLoc");
      }
      llvm_unreachable("Unrecognized VarLocBasedLDV.VarLoc.Kind enum");
    }

    /// Is the Loc field a constant or constant object?
    bool isConstant() const { return Kind == ImmediateKind; }

    /// Check if the Loc field is an entry backup location.
    bool isEntryBackupLoc() const {
      return Kind == EntryValueBackupKind || Kind == EntryValueCopyBackupKind;
    }

    /// If this variable is described by a register holding the entry value,
    /// return it, otherwise return 0.
    unsigned getEntryValueBackupReg() const {
      if (Kind == EntryValueBackupKind)
        return Loc.RegNo;
      return 0;
    }

    /// If this variable is described by a register holding the copy of the
    /// entry value, return it, otherwise return 0.
    unsigned getEntryValueCopyBackupReg() const {
      if (Kind == EntryValueCopyBackupKind)
        return Loc.RegNo;
      return 0;
    }

    /// If this variable is described by a register, return it,
    /// otherwise return 0.
    unsigned isDescribedByReg() const {
      if (Kind == RegisterKind)
        return Loc.RegNo;
      return 0;
    }

    /// Determine whether the lexical scope of this value's debug location
    /// dominates MBB.
    bool dominates(LexicalScopes &LS, MachineBasicBlock &MBB) const {
      return LS.dominates(MI.getDebugLoc().get(), &MBB);
    }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    // TRI can be null.
    void dump(const TargetRegisterInfo *TRI, raw_ostream &Out = dbgs()) const {
      Out << "VarLoc(";
      switch (Kind) {
      case RegisterKind:
      case EntryValueKind:
      case EntryValueBackupKind:
      case EntryValueCopyBackupKind:
        Out << printReg(Loc.RegNo, TRI);
        break;
      case SpillLocKind:
        Out << printReg(Loc.SpillLocation.SpillBase, TRI);
        Out << "[" << Loc.SpillLocation.SpillOffset.getFixed() << " + "
            << Loc.SpillLocation.SpillOffset.getScalable() << "x vscale"
            << "]";
        break;
      case ImmediateKind:
        Out << Loc.Immediate;
        break;
      case InvalidKind:
        llvm_unreachable("Invalid VarLoc in dump method");
      }

      Out << ", \"" << Var.getVariable()->getName() << "\", " << *Expr << ", ";
      if (Var.getInlinedAt())
        Out << "!" << Var.getInlinedAt()->getMetadataID() << ")\n";
      else
        Out << "(null))";

      if (isEntryBackupLoc())
        Out << " (backup loc)\n";
      else
        Out << "\n";
    }
#endif

    bool operator==(const VarLoc &Other) const {
      if (Kind != Other.Kind || !(Var == Other.Var) || Expr != Other.Expr)
        return false;

      switch (Kind) {
      case SpillLocKind:
        return Loc.SpillLocation == Other.Loc.SpillLocation;
      case RegisterKind:
      case ImmediateKind:
      case EntryValueKind:
      case EntryValueBackupKind:
      case EntryValueCopyBackupKind:
        return Loc.Hash == Other.Loc.Hash;
      default:
        llvm_unreachable("Invalid kind");
      }
    }

    /// This operator guarantees that VarLocs are sorted by Variable first.
    bool operator<(const VarLoc &Other) const {
      switch (Kind) {
      case SpillLocKind:
        return std::make_tuple(Var, Kind, Loc.SpillLocation.SpillBase,
                               Loc.SpillLocation.SpillOffset.getFixed(),
                               Loc.SpillLocation.SpillOffset.getScalable(),
                               Expr) <
               std::make_tuple(
                   Other.Var, Other.Kind, Other.Loc.SpillLocation.SpillBase,
                   Other.Loc.SpillLocation.SpillOffset.getFixed(),
                   Other.Loc.SpillLocation.SpillOffset.getScalable(),
                   Other.Expr);
      case RegisterKind:
      case ImmediateKind:
      case EntryValueKind:
      case EntryValueBackupKind:
      case EntryValueCopyBackupKind:
        return std::tie(Var, Kind, Loc.Hash, Expr) <
               std::tie(Other.Var, Other.Kind, Other.Loc.Hash, Other.Expr);
      default:
        llvm_unreachable("Invalid kind");
      }
    }
  };

  /// VarLocMap is used for two things:
  /// 1) Assigning a unique LocIndex to a VarLoc. This LocIndex can be used to
  ///    virtually insert a VarLoc into a VarLocSet.
  /// 2) Given a LocIndex, look up the unique associated VarLoc.
  class VarLocMap {
    /// Map a VarLoc to an index within the vector reserved for its location
    /// within Loc2Vars.
    std::map<VarLoc, LocIndex::u32_index_t> Var2Index;

    /// Map a location to a vector which holds VarLocs which live in that
    /// location.
    SmallDenseMap<LocIndex::u32_location_t, std::vector<VarLoc>> Loc2Vars;

    /// Determine the 32-bit location reserved for \p VL, based on its kind.
    static LocIndex::u32_location_t getLocationForVar(const VarLoc &VL) {
      switch (VL.Kind) {
      case VarLoc::RegisterKind:
        assert((VL.Loc.RegNo < LocIndex::kFirstInvalidRegLocation) &&
               "Physreg out of range?");
        return VL.Loc.RegNo;
      case VarLoc::SpillLocKind:
        return LocIndex::kSpillLocation;
      case VarLoc::EntryValueBackupKind:
      case VarLoc::EntryValueCopyBackupKind:
        return LocIndex::kEntryValueBackupLocation;
      default:
        return 0;
      }
    }

  public:
    /// Retrieve a unique LocIndex for \p VL.
    LocIndex insert(const VarLoc &VL) {
      LocIndex::u32_location_t Location = getLocationForVar(VL);
      LocIndex::u32_index_t &Index = Var2Index[VL];
      if (!Index) {
        auto &Vars = Loc2Vars[Location];
        Vars.push_back(VL);
        Index = Vars.size();
      }
      return {Location, Index - 1};
    }

    /// Retrieve the unique VarLoc associated with \p ID.
    const VarLoc &operator[](LocIndex ID) const {
      auto LocIt = Loc2Vars.find(ID.Location);
      assert(LocIt != Loc2Vars.end() && "Location not tracked");
      return LocIt->second[ID.Index];
    }
  };

  using VarLocInMBB =
      SmallDenseMap<const MachineBasicBlock *, std::unique_ptr<VarLocSet>>;
  struct TransferDebugPair {
    MachineInstr *TransferInst; ///< Instruction where this transfer occurs.
    LocIndex LocationID;        ///< Location number for the transfer dest.
  };
  using TransferMap = SmallVector<TransferDebugPair, 4>;

  // Types for recording sets of variable fragments that overlap. For a given
  // local variable, we record all other fragments of that variable that could
  // overlap it, to reduce search time.
  using FragmentOfVar =
      std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
  using OverlapMap =
      DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// This holds the working set of currently open ranges. For fast
  /// access, this is done both as a set of VarLocIDs, and a map of
  /// DebugVariable to recent VarLocID. Note that a DBG_VALUE ends all
  /// previous open ranges for the same variable. In addition, we keep
  /// two different maps (Vars/EntryValuesBackupVars), so erase/insert
  /// methods act differently depending on whether a VarLoc is primary
  /// location or backup one. In the case the VarLoc is backup location
  /// we will erase/insert from the EntryValuesBackupVars map, otherwise
  /// we perform the operation on the Vars.
  class OpenRangesSet {
    VarLocSet VarLocs;
    // Map the DebugVariable to recent primary location ID.
    SmallDenseMap<DebugVariable, LocIndex, 8> Vars;
    // Map the DebugVariable to recent backup location ID.
    SmallDenseMap<DebugVariable, LocIndex, 8> EntryValuesBackupVars;
    OverlapMap &OverlappingFragments;

  public:
    OpenRangesSet(VarLocSet::Allocator &Alloc, OverlapMap &_OLapMap)
        : VarLocs(Alloc), OverlappingFragments(_OLapMap) {}

    const VarLocSet &getVarLocs() const { return VarLocs; }

    /// Terminate all open ranges for VL.Var by removing it from the set.
    void erase(const VarLoc &VL);

    /// Terminate all open ranges listed in \c KillSet by removing
    /// them from the set.
    void erase(const VarLocSet &KillSet, const VarLocMap &VarLocIDs);

    /// Insert a new range into the set.
    void insert(LocIndex VarLocID, const VarLoc &VL);

    /// Insert a set of ranges.
    void insertFromLocSet(const VarLocSet &ToLoad, const VarLocMap &Map) {
      for (uint64_t ID : ToLoad) {
        LocIndex Idx = LocIndex::fromRawInteger(ID);
        const VarLoc &VarL = Map[Idx];
        insert(Idx, VarL);
      }
    }

    llvm::Optional<LocIndex> getEntryValueBackup(DebugVariable Var);

    /// Empty the set.
    void clear() {
      VarLocs.clear();
      Vars.clear();
      EntryValuesBackupVars.clear();
    }

    /// Return whether the set is empty or not.
    bool empty() const {
      assert(Vars.empty() == EntryValuesBackupVars.empty() &&
             Vars.empty() == VarLocs.empty() &&
             "open ranges are inconsistent");
      return VarLocs.empty();
    }

    /// Get an empty range of VarLoc IDs.
    auto getEmptyVarLocRange() const {
      return iterator_range<VarLocSet::const_iterator>(getVarLocs().end(),
                                                       getVarLocs().end());
    }

    /// Get all set IDs for VarLocs of kind RegisterKind in \p Reg.
    auto getRegisterVarLocs(Register Reg) const {
      return LocIndex::indexRangeForLocation(getVarLocs(), Reg);
    }

    /// Get all set IDs for VarLocs of kind SpillLocKind.
    auto getSpillVarLocs() const {
      return LocIndex::indexRangeForLocation(getVarLocs(),
                                             LocIndex::kSpillLocation);
    }

    /// Get all set IDs for VarLocs of kind EntryValueBackupKind or
    /// EntryValueCopyBackupKind.
    auto getEntryValueBackupVarLocs() const {
      return LocIndex::indexRangeForLocation(
          getVarLocs(), LocIndex::kEntryValueBackupLocation);
    }
  };

  /// Collect all VarLoc IDs from \p CollectFrom for VarLocs of kind
  /// RegisterKind which are located in any reg in \p Regs. Insert collected IDs
  /// into \p Collected.
  void collectIDsForRegs(VarLocSet &Collected, const DefinedRegsSet &Regs,
                         const VarLocSet &CollectFrom) const;

  /// Get the registers which are used by VarLocs of kind RegisterKind tracked
  /// by \p CollectFrom.
  void getUsedRegs(const VarLocSet &CollectFrom,
                   SmallVectorImpl<uint32_t> &UsedRegs) const;

  VarLocSet &getVarLocsInMBB(const MachineBasicBlock *MBB, VarLocInMBB &Locs) {
    std::unique_ptr<VarLocSet> &VLS = Locs[MBB];
    if (!VLS)
      VLS = std::make_unique<VarLocSet>(Alloc);
    return *VLS.get();
  }

  const VarLocSet &getVarLocsInMBB(const MachineBasicBlock *MBB,
                                   const VarLocInMBB &Locs) const {
    auto It = Locs.find(MBB);
    assert(It != Locs.end() && "MBB not in map");
    return *It->second.get();
  }

  /// Tests whether this instruction is a spill to a stack location.
  bool isSpillInstruction(const MachineInstr &MI, MachineFunction *MF);

  /// Decide if @MI is a spill instruction and return true if it is. We use 2
  /// criteria to make this decision:
  /// - Is this instruction a store to a spill slot?
  /// - Is there a register operand that is both used and killed?
  /// TODO: Store optimization can fold spills into other stores (including
  /// other spills). We do not handle this yet (more than one memory operand).
  bool isLocationSpill(const MachineInstr &MI, MachineFunction *MF,
                       Register &Reg);

  /// Returns true if the given machine instruction is a debug value which we
  /// can emit entry values for.
  ///
  /// Currently, we generate debug entry values only for parameters that are
  /// unmodified throughout the function and located in a register.
  bool isEntryValueCandidate(const MachineInstr &MI,
                             const DefinedRegsSet &Regs) const;

  /// If a given instruction is identified as a spill, return the spill location
  /// and set \p Reg to the spilled register.
  Optional<VarLoc::SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                                  MachineFunction *MF,
                                                  Register &Reg);
  /// Given a spill instruction, extract the register and offset used to
  /// address the spill location in a target independent way.
  VarLoc::SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);
  void insertTransferDebugPair(MachineInstr &MI, OpenRangesSet &OpenRanges,
                               TransferMap &Transfers, VarLocMap &VarLocIDs,
                               LocIndex OldVarID, TransferKind Kind,
                               Register NewReg = Register());

  void transferDebugValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                          VarLocMap &VarLocIDs);
  void transferSpillOrRestoreInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                                  VarLocMap &VarLocIDs, TransferMap &Transfers);
  bool removeEntryValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                        VarLocMap &VarLocIDs, const VarLoc &EntryVL);
  void emitEntryValues(MachineInstr &MI, OpenRangesSet &OpenRanges,
                       VarLocMap &VarLocIDs, TransferMap &Transfers,
                       VarLocSet &KillSet);
  void recordEntryValue(const MachineInstr &MI,
                        const DefinedRegsSet &DefinedRegs,
                        OpenRangesSet &OpenRanges, VarLocMap &VarLocIDs);
  void transferRegisterCopy(MachineInstr &MI, OpenRangesSet &OpenRanges,
                            VarLocMap &VarLocIDs, TransferMap &Transfers);
  void transferRegisterDef(MachineInstr &MI, OpenRangesSet &OpenRanges,
                           VarLocMap &VarLocIDs, TransferMap &Transfers);
  bool transferTerminator(MachineBasicBlock *MBB, OpenRangesSet &OpenRanges,
                          VarLocInMBB &OutLocs, const VarLocMap &VarLocIDs);

  void process(MachineInstr &MI, OpenRangesSet &OpenRanges,
               VarLocMap &VarLocIDs, TransferMap &Transfers);

  void accumulateFragmentMap(MachineInstr &MI, VarToFragments &SeenFragments,
                             OverlapMap &OLapMap);

  bool join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
            const VarLocMap &VarLocIDs,
            SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
            SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks);

  /// Create DBG_VALUE insts for inlocs that have been propagated but
  /// had their instruction creation deferred.
  void flushPendingLocs(VarLocInMBB &PendingInLocs, VarLocMap &VarLocIDs);

  bool ExtendRanges(MachineFunction &MF, TargetPassConfig *TPC) override;

public:
  /// Default construct and initialize the pass.
  VarLocBasedLDV();

  ~VarLocBasedLDV();

  /// Print to ostream with a message.
  void printVarLocInMBB(const MachineFunction &MF, const VarLocInMBB &V,
                        const VarLocMap &VarLocIDs, const char *msg,
                        raw_ostream &Out) const;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

VarLocBasedLDV::VarLocBasedLDV() { }

VarLocBasedLDV::~VarLocBasedLDV() { }

/// Erase a variable from the set of open ranges, and additionally erase any
/// fragments that may overlap it. If the VarLoc is a backup location, erase
/// the variable from the EntryValuesBackupVars set, indicating we should stop
/// tracking its backup entry location. Otherwise, if the VarLoc is primary
/// location, erase the variable from the Vars set.
void VarLocBasedLDV::OpenRangesSet::erase(const VarLoc &VL) {
  // Erasure helper.
  auto DoErase = [VL, this](DebugVariable VarToErase) {
    auto *EraseFrom = VL.isEntryBackupLoc() ? &EntryValuesBackupVars : &Vars;
    auto It = EraseFrom->find(VarToErase);
    if (It != EraseFrom->end()) {
      LocIndex ID = It->second;
      VarLocs.reset(ID.getAsRawInteger());
      EraseFrom->erase(It);
    }
  };

  DebugVariable Var = VL.Var;

  // Erase the variable/fragment that ends here.
  DoErase(Var);

  // Extract the fragment. Interpret an empty fragment as one that covers all
  // possible bits.
  FragmentInfo ThisFragment = Var.getFragmentOrDefault();

  // There may be fragments that overlap the designated fragment. Look them up
  // in the pre-computed overlap map, and erase them too.
  auto MapIt = OverlappingFragments.find({Var.getVariable(), ThisFragment});
  if (MapIt != OverlappingFragments.end()) {
    for (auto Fragment : MapIt->second) {
      VarLocBasedLDV::OptFragmentInfo FragmentHolder;
      if (!DebugVariable::isDefaultFragment(Fragment))
        FragmentHolder = VarLocBasedLDV::OptFragmentInfo(Fragment);
      DoErase({Var.getVariable(), FragmentHolder, Var.getInlinedAt()});
    }
  }
}

void VarLocBasedLDV::OpenRangesSet::erase(const VarLocSet &KillSet,
                                           const VarLocMap &VarLocIDs) {
  VarLocs.intersectWithComplement(KillSet);
  for (uint64_t ID : KillSet) {
    const VarLoc *VL = &VarLocIDs[LocIndex::fromRawInteger(ID)];
    auto *EraseFrom = VL->isEntryBackupLoc() ? &EntryValuesBackupVars : &Vars;
    EraseFrom->erase(VL->Var);
  }
}

void VarLocBasedLDV::OpenRangesSet::insert(LocIndex VarLocID,
                                            const VarLoc &VL) {
  auto *InsertInto = VL.isEntryBackupLoc() ? &EntryValuesBackupVars : &Vars;
  VarLocs.set(VarLocID.getAsRawInteger());
  InsertInto->insert({VL.Var, VarLocID});
}

/// Return the Loc ID of an entry value backup location, if it exists for the
/// variable.
llvm::Optional<LocIndex>
VarLocBasedLDV::OpenRangesSet::getEntryValueBackup(DebugVariable Var) {
  auto It = EntryValuesBackupVars.find(Var);
  if (It != EntryValuesBackupVars.end())
    return It->second;

  return llvm::None;
}

void VarLocBasedLDV::collectIDsForRegs(VarLocSet &Collected,
                                        const DefinedRegsSet &Regs,
                                        const VarLocSet &CollectFrom) const {
  assert(!Regs.empty() && "Nothing to collect");
  SmallVector<uint32_t, 32> SortedRegs;
  for (Register Reg : Regs)
    SortedRegs.push_back(Reg);
  array_pod_sort(SortedRegs.begin(), SortedRegs.end());
  auto It = CollectFrom.find(LocIndex::rawIndexForReg(SortedRegs.front()));
  auto End = CollectFrom.end();
  for (uint32_t Reg : SortedRegs) {
    // The half-open interval [FirstIndexForReg, FirstInvalidIndex) contains all
    // possible VarLoc IDs for VarLocs of kind RegisterKind which live in Reg.
    uint64_t FirstIndexForReg = LocIndex::rawIndexForReg(Reg);
    uint64_t FirstInvalidIndex = LocIndex::rawIndexForReg(Reg + 1);
    It.advanceToLowerBound(FirstIndexForReg);

    // Iterate through that half-open interval and collect all the set IDs.
    for (; It != End && *It < FirstInvalidIndex; ++It)
      Collected.set(*It);

    if (It == End)
      return;
  }
}

void VarLocBasedLDV::getUsedRegs(const VarLocSet &CollectFrom,
                                  SmallVectorImpl<uint32_t> &UsedRegs) const {
  // All register-based VarLocs are assigned indices greater than or equal to
  // FirstRegIndex.
  uint64_t FirstRegIndex = LocIndex::rawIndexForReg(1);
  uint64_t FirstInvalidIndex =
      LocIndex::rawIndexForReg(LocIndex::kFirstInvalidRegLocation);
  for (auto It = CollectFrom.find(FirstRegIndex),
            End = CollectFrom.find(FirstInvalidIndex);
       It != End;) {
    // We found a VarLoc ID for a VarLoc that lives in a register. Figure out
    // which register and add it to UsedRegs.
    uint32_t FoundReg = LocIndex::fromRawInteger(*It).Location;
    assert((UsedRegs.empty() || FoundReg != UsedRegs.back()) &&
           "Duplicate used reg");
    UsedRegs.push_back(FoundReg);

    // Skip to the next /set/ register. Note that this finds a lower bound, so
    // even if there aren't any VarLocs living in `FoundReg+1`, we're still
    // guaranteed to move on to the next register (or to end()).
    uint64_t NextRegIndex = LocIndex::rawIndexForReg(FoundReg + 1);
    It.advanceToLowerBound(NextRegIndex);
  }
}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
void VarLocBasedLDV::printVarLocInMBB(const MachineFunction &MF,
                                       const VarLocInMBB &V,
                                       const VarLocMap &VarLocIDs,
                                       const char *msg,
                                       raw_ostream &Out) const {
  Out << '\n' << msg << '\n';
  for (const MachineBasicBlock &BB : MF) {
    if (!V.count(&BB))
      continue;
    const VarLocSet &L = getVarLocsInMBB(&BB, V);
    if (L.empty())
      continue;
    Out << "MBB: " << BB.getNumber() << ":\n";
    for (uint64_t VLL : L) {
      const VarLoc &VL = VarLocIDs[LocIndex::fromRawInteger(VLL)];
      Out << " Var: " << VL.Var.getVariable()->getName();
      Out << " MI: ";
      VL.dump(TRI, Out);
    }
  }
  Out << "\n";
}
#endif

VarLocBasedLDV::VarLoc::SpillLoc
VarLocBasedLDV::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
  assert(MI.hasOneMemOperand() &&
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  Register Reg;
  StackOffset Offset = TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
  return {Reg, Offset};
}

/// Try to salvage the debug entry value if we encounter a new debug value
/// describing the same parameter, otherwise stop tracking the value. Return
/// true if we should stop tracking the entry value, otherwise return false.
bool VarLocBasedLDV::removeEntryValue(const MachineInstr &MI,
                                       OpenRangesSet &OpenRanges,
                                       VarLocMap &VarLocIDs,
                                       const VarLoc &EntryVL) {
  // Skip the DBG_VALUE which is the debug entry value itself.
  if (MI.isIdenticalTo(EntryVL.MI))
    return false;

  // If the parameter's location is not register location, we can not track
  // the entry value any more. In addition, if the debug expression from the
  // DBG_VALUE is not empty, we can assume the parameter's value has changed
  // indicating that we should stop tracking its entry value as well.
  if (!MI.getDebugOperand(0).isReg() ||
      MI.getDebugExpression()->getNumElements() != 0)
    return true;

  // If the DBG_VALUE comes from a copy instruction that copies the entry value,
  // it means the parameter's value has not changed and we should be able to use
  // its entry value.
  bool TrySalvageEntryValue = false;
  Register Reg = MI.getDebugOperand(0).getReg();
  auto I = std::next(MI.getReverseIterator());
  const MachineOperand *SrcRegOp, *DestRegOp;
  if (I != MI.getParent()->rend()) {
    // TODO: Try to keep tracking of an entry value if we encounter a propagated
    // DBG_VALUE describing the copy of the entry value. (Propagated entry value
    // does not indicate the parameter modification.)
    auto DestSrc = TII->isCopyInstr(*I);
    if (!DestSrc)
      return true;

    SrcRegOp = DestSrc->Source;
    DestRegOp = DestSrc->Destination;
    if (Reg != DestRegOp->getReg())
      return true;
    TrySalvageEntryValue = true;
  }

  if (TrySalvageEntryValue) {
    for (uint64_t ID : OpenRanges.getEntryValueBackupVarLocs()) {
      const VarLoc &VL = VarLocIDs[LocIndex::fromRawInteger(ID)];
      if (VL.getEntryValueCopyBackupReg() == Reg &&
          VL.MI.getDebugOperand(0).getReg() == SrcRegOp->getReg())
        return false;
    }
  }

  return true;
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
void VarLocBasedLDV::transferDebugValue(const MachineInstr &MI,
                                         OpenRangesSet &OpenRanges,
                                         VarLocMap &VarLocIDs) {
  if (!MI.isDebugValue())
    return;
  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  DebugVariable V(Var, Expr, InlinedAt);

  // Check if this DBG_VALUE indicates a parameter's value changing.
  // If that is the case, we should stop tracking its entry value.
  auto EntryValBackupID = OpenRanges.getEntryValueBackup(V);
  if (Var->isParameter() && EntryValBackupID) {
    const VarLoc &EntryVL = VarLocIDs[*EntryValBackupID];
    if (removeEntryValue(MI, OpenRanges, VarLocIDs, EntryVL)) {
      LLVM_DEBUG(dbgs() << "Deleting a DBG entry value because of: ";
                 MI.print(dbgs(), /*IsStandalone*/ false,
                          /*SkipOpers*/ false, /*SkipDebugLoc*/ false,
                          /*AddNewLine*/ true, TII));
      OpenRanges.erase(EntryVL);
    }
  }

  if (isDbgValueDescribedByReg(MI) || MI.getDebugOperand(0).isImm() ||
      MI.getDebugOperand(0).isFPImm() || MI.getDebugOperand(0).isCImm()) {
    // Use normal VarLoc constructor for registers and immediates.
    VarLoc VL(MI, LS);
    // End all previous ranges of VL.Var.
    OpenRanges.erase(VL);

    LocIndex ID = VarLocIDs.insert(VL);
    // Add the VarLoc to OpenRanges from this DBG_VALUE.
    OpenRanges.insert(ID, VL);
  } else if (MI.hasOneMemOperand()) {
    llvm_unreachable("DBG_VALUE with mem operand encountered after regalloc?");
  } else {
    // This must be an undefined location. If it has an open range, erase it.
    assert(MI.getDebugOperand(0).isReg() &&
           MI.getDebugOperand(0).getReg() == 0 &&
           "Unexpected non-undef DBG_VALUE encountered");
    VarLoc VL(MI, LS);
    OpenRanges.erase(VL);
  }
}

/// Turn the entry value backup locations into primary locations.
void VarLocBasedLDV::emitEntryValues(MachineInstr &MI,
                                      OpenRangesSet &OpenRanges,
                                      VarLocMap &VarLocIDs,
                                      TransferMap &Transfers,
                                      VarLocSet &KillSet) {
  // Do not insert entry value locations after a terminator.
  if (MI.isTerminator())
    return;

  for (uint64_t ID : KillSet) {
    LocIndex Idx = LocIndex::fromRawInteger(ID);
    const VarLoc &VL = VarLocIDs[Idx];
    if (!VL.Var.getVariable()->isParameter())
      continue;

    auto DebugVar = VL.Var;
    Optional<LocIndex> EntryValBackupID =
        OpenRanges.getEntryValueBackup(DebugVar);

    // If the parameter has the entry value backup, it means we should
    // be able to use its entry value.
    if (!EntryValBackupID)
      continue;

    const VarLoc &EntryVL = VarLocIDs[*EntryValBackupID];
    VarLoc EntryLoc =
        VarLoc::CreateEntryLoc(EntryVL.MI, LS, EntryVL.Expr, EntryVL.Loc.RegNo);
    LocIndex EntryValueID = VarLocIDs.insert(EntryLoc);
    Transfers.push_back({&MI, EntryValueID});
    OpenRanges.insert(EntryValueID, EntryLoc);
  }
}

/// Create new TransferDebugPair and insert it in \p Transfers. The VarLoc
/// with \p OldVarID should be deleted form \p OpenRanges and replaced with
/// new VarLoc. If \p NewReg is different than default zero value then the
/// new location will be register location created by the copy like instruction,
/// otherwise it is variable's location on the stack.
void VarLocBasedLDV::insertTransferDebugPair(
    MachineInstr &MI, OpenRangesSet &OpenRanges, TransferMap &Transfers,
    VarLocMap &VarLocIDs, LocIndex OldVarID, TransferKind Kind,
    Register NewReg) {
  const MachineInstr *DebugInstr = &VarLocIDs[OldVarID].MI;

  auto ProcessVarLoc = [&MI, &OpenRanges, &Transfers, &VarLocIDs](VarLoc &VL) {
    LocIndex LocId = VarLocIDs.insert(VL);

    // Close this variable's previous location range.
    OpenRanges.erase(VL);

    // Record the new location as an open range, and a postponed transfer
    // inserting a DBG_VALUE for this location.
    OpenRanges.insert(LocId, VL);
    assert(!MI.isTerminator() && "Cannot insert DBG_VALUE after terminator");
    TransferDebugPair MIP = {&MI, LocId};
    Transfers.push_back(MIP);
  };

  // End all previous ranges of VL.Var.
  OpenRanges.erase(VarLocIDs[OldVarID]);
  switch (Kind) {
  case TransferKind::TransferCopy: {
    assert(NewReg &&
           "No register supplied when handling a copy of a debug value");
    // Create a DBG_VALUE instruction to describe the Var in its new
    // register location.
    VarLoc VL = VarLoc::CreateCopyLoc(*DebugInstr, LS, NewReg);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for register copy:";
      VL.dump(TRI);
    });
    return;
  }
  case TransferKind::TransferSpill: {
    // Create a DBG_VALUE instruction to describe the Var in its spilled
    // location.
    VarLoc::SpillLoc SpillLocation = extractSpillBaseRegAndOffset(MI);
    VarLoc VL = VarLoc::CreateSpillLoc(*DebugInstr, SpillLocation.SpillBase,
                                       SpillLocation.SpillOffset, LS);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for spill:";
      VL.dump(TRI);
    });
    return;
  }
  case TransferKind::TransferRestore: {
    assert(NewReg &&
           "No register supplied when handling a restore of a debug value");
    // DebugInstr refers to the pre-spill location, therefore we can reuse
    // its expression.
    VarLoc VL = VarLoc::CreateCopyLoc(*DebugInstr, LS, NewReg);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for restore:";
      VL.dump(TRI);
    });
    return;
  }
  }
  llvm_unreachable("Invalid transfer kind");
}

/// A definition of a register may mark the end of a range.
void VarLocBasedLDV::transferRegisterDef(
    MachineInstr &MI, OpenRangesSet &OpenRanges, VarLocMap &VarLocIDs,
    TransferMap &Transfers) {

  // Meta Instructions do not affect the debug liveness of any register they
  // define.
  if (MI.isMetaInstruction())
    return;

  MachineFunction *MF = MI.getMF();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  Register SP = TLI->getStackPointerRegisterToSaveRestore();

  // Find the regs killed by MI, and find regmasks of preserved regs.
  DefinedRegsSet DeadRegs;
  SmallVector<const uint32_t *, 4> RegMasks;
  for (const MachineOperand &MO : MI.operands()) {
    // Determine whether the operand is a register def.
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        Register::isPhysicalRegister(MO.getReg()) &&
        !(MI.isCall() && MO.getReg() == SP)) {
      // Remove ranges of all aliased registers.
      for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
        // FIXME: Can we break out of this loop early if no insertion occurs?
        DeadRegs.insert(*RAI);
    } else if (MO.isRegMask()) {
      RegMasks.push_back(MO.getRegMask());
    }
  }

  // Erase VarLocs which reside in one of the dead registers. For performance
  // reasons, it's critical to not iterate over the full set of open VarLocs.
  // Iterate over the set of dying/used regs instead.
  if (!RegMasks.empty()) {
    SmallVector<uint32_t, 32> UsedRegs;
    getUsedRegs(OpenRanges.getVarLocs(), UsedRegs);
    for (uint32_t Reg : UsedRegs) {
      // Remove ranges of all clobbered registers. Register masks don't usually
      // list SP as preserved. Assume that call instructions never clobber SP,
      // because some backends (e.g., AArch64) never list SP in the regmask.
      // While the debug info may be off for an instruction or two around
      // callee-cleanup calls, transferring the DEBUG_VALUE across the call is
      // still a better user experience.
      if (Reg == SP)
        continue;
      bool AnyRegMaskKillsReg =
          any_of(RegMasks, [Reg](const uint32_t *RegMask) {
            return MachineOperand::clobbersPhysReg(RegMask, Reg);
          });
      if (AnyRegMaskKillsReg)
        DeadRegs.insert(Reg);
    }
  }

  if (DeadRegs.empty())
    return;

  VarLocSet KillSet(Alloc);
  collectIDsForRegs(KillSet, DeadRegs, OpenRanges.getVarLocs());
  OpenRanges.erase(KillSet, VarLocIDs);

  if (TPC) {
    auto &TM = TPC->getTM<TargetMachine>();
    if (TM.Options.ShouldEmitDebugEntryValues())
      emitEntryValues(MI, OpenRanges, VarLocIDs, Transfers, KillSet);
  }
}

bool VarLocBasedLDV::isSpillInstruction(const MachineInstr &MI,
                                         MachineFunction *MF) {
  // TODO: Handle multiple stores folded into one.
  if (!MI.hasOneMemOperand())
    return false;

  if (!MI.getSpillSize(TII) && !MI.getFoldedSpillSize(TII))
    return false; // This is not a spill instruction, since no valid size was
                  // returned from either function.

  return true;
}

bool VarLocBasedLDV::isLocationSpill(const MachineInstr &MI,
                                      MachineFunction *MF, Register &Reg) {
  if (!isSpillInstruction(MI, MF))
    return false;

  auto isKilledReg = [&](const MachineOperand MO, Register &Reg) {
    if (!MO.isReg() || !MO.isUse()) {
      Reg = 0;
      return false;
    }
    Reg = MO.getReg();
    return MO.isKill();
  };

  for (const MachineOperand &MO : MI.operands()) {
    // In a spill instruction generated by the InlineSpiller the spilled
    // register has its kill flag set.
    if (isKilledReg(MO, Reg))
      return true;
    if (Reg != 0) {
      // Check whether next instruction kills the spilled register.
      // FIXME: Current solution does not cover search for killed register in
      // bundles and instructions further down the chain.
      auto NextI = std::next(MI.getIterator());
      // Skip next instruction that points to basic block end iterator.
      if (MI.getParent()->end() == NextI)
        continue;
      Register RegNext;
      for (const MachineOperand &MONext : NextI->operands()) {
        // Return true if we came across the register from the
        // previous spill instruction that is killed in NextI.
        if (isKilledReg(MONext, RegNext) && RegNext == Reg)
          return true;
      }
    }
  }
  // Return false if we didn't find spilled register.
  return false;
}

Optional<VarLocBasedLDV::VarLoc::SpillLoc>
VarLocBasedLDV::isRestoreInstruction(const MachineInstr &MI,
                                      MachineFunction *MF, Register &Reg) {
  if (!MI.hasOneMemOperand())
    return None;

  // FIXME: Handle folded restore instructions with more than one memory
  // operand.
  if (MI.getRestoreSize(TII)) {
    Reg = MI.getOperand(0).getReg();
    return extractSpillBaseRegAndOffset(MI);
  }
  return None;
}

/// A spilled register may indicate that we have to end the current range of
/// a variable and create a new one for the spill location.
/// A restored register may indicate the reverse situation.
/// We don't want to insert any instructions in process(), so we just create
/// the DBG_VALUE without inserting it and keep track of it in \p Transfers.
/// It will be inserted into the BB when we're done iterating over the
/// instructions.
void VarLocBasedLDV::transferSpillOrRestoreInst(MachineInstr &MI,
                                                 OpenRangesSet &OpenRanges,
                                                 VarLocMap &VarLocIDs,
                                                 TransferMap &Transfers) {
  MachineFunction *MF = MI.getMF();
  TransferKind TKind;
  Register Reg;
  Optional<VarLoc::SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, then close the variable location. The value in memory
  // will have changed.
  VarLocSet KillSet(Alloc);
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);
    for (uint64_t ID : OpenRanges.getSpillVarLocs()) {
      LocIndex Idx = LocIndex::fromRawInteger(ID);
      const VarLoc &VL = VarLocIDs[Idx];
      assert(VL.Kind == VarLoc::SpillLocKind && "Broken VarLocSet?");
      if (VL.Loc.SpillLocation == *Loc) {
        // This location is overwritten by the current instruction -- terminate
        // the open range, and insert an explicit DBG_VALUE $noreg.
        //
        // Doing this at a later stage would require re-interpreting all
        // DBG_VALUes and DIExpressions to identify whether they point at
        // memory, and then analysing all memory writes to see if they
        // overwrite that memory, which is expensive.
        //
        // At this stage, we already know which DBG_VALUEs are for spills and
        // where they are located; it's best to fix handle overwrites now.
        KillSet.set(ID);
        VarLoc UndefVL = VarLoc::CreateCopyLoc(VL.MI, LS, 0);
        LocIndex UndefLocID = VarLocIDs.insert(UndefVL);
        Transfers.push_back({&MI, UndefLocID});
      }
    }
    OpenRanges.erase(KillSet, VarLocIDs);
  }

  // Try to recognise spill and restore instructions that may create a new
  // variable location.
  if (isLocationSpill(MI, MF, Reg)) {
    TKind = TransferKind::TransferSpill;
    LLVM_DEBUG(dbgs() << "Recognized as spill: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return;
    TKind = TransferKind::TransferRestore;
    LLVM_DEBUG(dbgs() << "Recognized as restore: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  }
  // Check if the register or spill location is the location of a debug value.
  auto TransferCandidates = OpenRanges.getEmptyVarLocRange();
  if (TKind == TransferKind::TransferSpill)
    TransferCandidates = OpenRanges.getRegisterVarLocs(Reg);
  else if (TKind == TransferKind::TransferRestore)
    TransferCandidates = OpenRanges.getSpillVarLocs();
  for (uint64_t ID : TransferCandidates) {
    LocIndex Idx = LocIndex::fromRawInteger(ID);
    const VarLoc &VL = VarLocIDs[Idx];
    if (TKind == TransferKind::TransferSpill) {
      assert(VL.isDescribedByReg() == Reg && "Broken VarLocSet?");
      LLVM_DEBUG(dbgs() << "Spilling Register " << printReg(Reg, TRI) << '('
                        << VL.Var.getVariable()->getName() << ")\n");
    } else {
      assert(TKind == TransferKind::TransferRestore &&
             VL.Kind == VarLoc::SpillLocKind && "Broken VarLocSet?");
      if (VL.Loc.SpillLocation != *Loc)
        // The spill location is not the location of a debug value.
        continue;
      LLVM_DEBUG(dbgs() << "Restoring Register " << printReg(Reg, TRI) << '('
                        << VL.Var.getVariable()->getName() << ")\n");
    }
    insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, Idx, TKind,
                            Reg);
    // FIXME: A comment should explain why it's correct to return early here,
    // if that is in fact correct.
    return;
  }
}

/// If \p MI is a register copy instruction, that copies a previously tracked
/// value from one register to another register that is callee saved, we
/// create new DBG_VALUE instruction  described with copy destination register.
void VarLocBasedLDV::transferRegisterCopy(MachineInstr &MI,
                                           OpenRangesSet &OpenRanges,
                                           VarLocMap &VarLocIDs,
                                           TransferMap &Transfers) {
  auto DestSrc = TII->isCopyInstr(MI);
  if (!DestSrc)
    return;

  const MachineOperand *DestRegOp = DestSrc->Destination;
  const MachineOperand *SrcRegOp = DestSrc->Source;

  if (!DestRegOp->isDef())
    return;

  auto isCalleeSavedReg = [&](Register Reg) {
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  };

  Register SrcReg = SrcRegOp->getReg();
  Register DestReg = DestRegOp->getReg();

  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register location, which is callee
  // saved, is going to stay unclobbered longer, even if it is killed.
  if (!isCalleeSavedReg(DestReg))
    return;

  // Remember an entry value movement. If we encounter a new debug value of
  // a parameter describing only a moving of the value around, rather then
  // modifying it, we are still able to use the entry value if needed.
  if (isRegOtherThanSPAndFP(*DestRegOp, MI, TRI)) {
    for (uint64_t ID : OpenRanges.getEntryValueBackupVarLocs()) {
      LocIndex Idx = LocIndex::fromRawInteger(ID);
      const VarLoc &VL = VarLocIDs[Idx];
      if (VL.getEntryValueBackupReg() == SrcReg) {
        LLVM_DEBUG(dbgs() << "Copy of the entry value: "; MI.dump(););
        VarLoc EntryValLocCopyBackup =
            VarLoc::CreateEntryCopyBackupLoc(VL.MI, LS, VL.Expr, DestReg);

        // Stop tracking the original entry value.
        OpenRanges.erase(VL);

        // Start tracking the entry value copy.
        LocIndex EntryValCopyLocID = VarLocIDs.insert(EntryValLocCopyBackup);
        OpenRanges.insert(EntryValCopyLocID, EntryValLocCopyBackup);
        break;
      }
    }
  }

  if (!SrcRegOp->isKill())
    return;

  for (uint64_t ID : OpenRanges.getRegisterVarLocs(SrcReg)) {
    LocIndex Idx = LocIndex::fromRawInteger(ID);
    assert(VarLocIDs[Idx].isDescribedByReg() == SrcReg && "Broken VarLocSet?");
    insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, Idx,
                            TransferKind::TransferCopy, DestReg);
    // FIXME: A comment should explain why it's correct to return early here,
    // if that is in fact correct.
    return;
  }
}

/// Terminate all open ranges at the end of the current basic block.
bool VarLocBasedLDV::transferTerminator(MachineBasicBlock *CurMBB,
                                         OpenRangesSet &OpenRanges,
                                         VarLocInMBB &OutLocs,
                                         const VarLocMap &VarLocIDs) {
  bool Changed = false;

  LLVM_DEBUG(for (uint64_t ID
                  : OpenRanges.getVarLocs()) {
    // Copy OpenRanges to OutLocs, if not already present.
    dbgs() << "Add to OutLocs in MBB #" << CurMBB->getNumber() << ":  ";
    VarLocIDs[LocIndex::fromRawInteger(ID)].dump(TRI);
  });
  VarLocSet &VLS = getVarLocsInMBB(CurMBB, OutLocs);
  Changed = VLS != OpenRanges.getVarLocs();
  // New OutLocs set may be different due to spill, restore or register
  // copy instruction processing.
  if (Changed)
    VLS = OpenRanges.getVarLocs();
  OpenRanges.clear();
  return Changed;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
/// \param SeenFragments Map from DILocalVariable to all fragments of that
///           Variable which are known to exist.
/// \param OverlappingFragments The overlap map being constructed, from one
///           Var/Fragment pair to a vector of fragments known to overlap.
void VarLocBasedLDV::accumulateFragmentMap(MachineInstr &MI,
                                            VarToFragments &SeenFragments,
                                            OverlapMap &OverlappingFragments) {
  DebugVariable MIVar(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
  FragmentInfo ThisFragment = MIVar.getFragmentOrDefault();

  // If this is the first sighting of this variable, then we are guaranteed
  // there are currently no overlapping fragments either. Initialize the set
  // of seen fragments, record no overlaps for the current one, and return.
  auto SeenIt = SeenFragments.find(MIVar.getVariable());
  if (SeenIt == SeenFragments.end()) {
    SmallSet<FragmentInfo, 4> OneFragment;
    OneFragment.insert(ThisFragment);
    SeenFragments.insert({MIVar.getVariable(), OneFragment});

    OverlappingFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlappingFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
  if (!IsInOLapMap.second)
    return;

  auto &ThisFragmentsOverlaps = IsInOLapMap.first->second;
  auto &AllSeenFragments = SeenIt->second;

  // Otherwise, examine all other seen fragments for this variable, with "this"
  // fragment being a previously unseen fragment. Record any pair of
  // overlapping fragments.
  for (auto &ASeenFragment : AllSeenFragments) {
    // Does this previously seen fragment overlap?
    if (DIExpression::fragmentsOverlap(ThisFragment, ASeenFragment)) {
      // Yes: Mark the current fragment as being overlapped.
      ThisFragmentsOverlaps.push_back(ASeenFragment);
      // Mark the previously seen fragment as being overlapped by the current
      // one.
      auto ASeenFragmentsOverlaps =
          OverlappingFragments.find({MIVar.getVariable(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlappingFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

/// This routine creates OpenRanges.
void VarLocBasedLDV::process(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocMap &VarLocIDs, TransferMap &Transfers) {
  transferDebugValue(MI, OpenRanges, VarLocIDs);
  transferRegisterDef(MI, OpenRanges, VarLocIDs, Transfers);
  transferRegisterCopy(MI, OpenRanges, VarLocIDs, Transfers);
  transferSpillOrRestoreInst(MI, OpenRanges, VarLocIDs, Transfers);
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool VarLocBasedLDV::join(
    MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
    const VarLocMap &VarLocIDs,
    SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");

  VarLocSet InLocsT(Alloc); // Temporary incoming locations.

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  for (auto p : MBB.predecessors()) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had locations propagated into it, most locations
    // will not yet be valid, so treat them as all being uninitialized and
    // potentially valid. If a location guessed to be correct here is
    // invalidated later, we will remove it when we revisit this block.
    if (!Visited.count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }
    auto OL = OutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == OutLocs.end())
      return false;

    // Just copy over the Out locs to incoming locs for the first visited
    // predecessor, and for all other predecessors join the Out locs.
    VarLocSet &OutLocVLS = *OL->second.get();
    if (!NumVisited)
      InLocsT = OutLocVLS;
    else
      InLocsT &= OutLocVLS;

    LLVM_DEBUG({
      if (!InLocsT.empty()) {
        for (uint64_t ID : InLocsT)
          dbgs() << "  gathered candidate incoming var: "
                 << VarLocIDs[LocIndex::fromRawInteger(ID)]
                        .Var.getVariable()
                        ->getName()
                 << "\n";
      }
    });

    NumVisited++;
  }

  // Filter out DBG_VALUES that are out of scope.
  VarLocSet KillSet(Alloc);
  bool IsArtificial = ArtificialBlocks.count(&MBB);
  if (!IsArtificial) {
    for (uint64_t ID : InLocsT) {
      LocIndex Idx = LocIndex::fromRawInteger(ID);
      if (!VarLocIDs[Idx].dominates(LS, MBB)) {
        KillSet.set(ID);
        LLVM_DEBUG({
          auto Name = VarLocIDs[Idx].Var.getVariable()->getName();
          dbgs() << "  killing " << Name << ", it doesn't dominate MBB\n";
        });
      }
    }
  }
  InLocsT.intersectWithComplement(KillSet);

  // As we are processing blocks in reverse post-order we
  // should have processed at least one predecessor, unless it
  // is the entry block which has no predecessor.
  assert((NumVisited || MBB.pred_empty()) &&
         "Should have processed at least one predecessor");

  VarLocSet &ILS = getVarLocsInMBB(&MBB, InLocs);
  bool Changed = false;
  if (ILS != InLocsT) {
    ILS = InLocsT;
    Changed = true;
  }

  return Changed;
}

void VarLocBasedLDV::flushPendingLocs(VarLocInMBB &PendingInLocs,
                                       VarLocMap &VarLocIDs) {
  // PendingInLocs records all locations propagated into blocks, which have
  // not had DBG_VALUE insts created. Go through and create those insts now.
  for (auto &Iter : PendingInLocs) {
    // Map is keyed on a constant pointer, unwrap it so we can insert insts.
    auto &MBB = const_cast<MachineBasicBlock &>(*Iter.first);
    VarLocSet &Pending = *Iter.second.get();

    for (uint64_t ID : Pending) {
      // The ID location is live-in to MBB -- work out what kind of machine
      // location it is and create a DBG_VALUE.
      const VarLoc &DiffIt = VarLocIDs[LocIndex::fromRawInteger(ID)];
      if (DiffIt.isEntryBackupLoc())
        continue;
      MachineInstr *MI = DiffIt.BuildDbgValue(*MBB.getParent());
      MBB.insert(MBB.instr_begin(), MI);

      (void)MI;
      LLVM_DEBUG(dbgs() << "Inserted: "; MI->dump(););
    }
  }
}

bool VarLocBasedLDV::isEntryValueCandidate(
    const MachineInstr &MI, const DefinedRegsSet &DefinedRegs) const {
  assert(MI.isDebugValue() && "This must be DBG_VALUE.");

  // TODO: Add support for local variables that are expressed in terms of
  // parameters entry values.
  // TODO: Add support for modified arguments that can be expressed
  // by using its entry value.
  auto *DIVar = MI.getDebugVariable();
  if (!DIVar->isParameter())
    return false;

  // Do not consider parameters that belong to an inlined function.
  if (MI.getDebugLoc()->getInlinedAt())
    return false;

  // Only consider parameters that are described using registers. Parameters
  // that are passed on the stack are not yet supported, so ignore debug
  // values that are described by the frame or stack pointer.
  if (!isRegOtherThanSPAndFP(MI.getDebugOperand(0), MI, TRI))
    return false;

  // If a parameter's value has been propagated from the caller, then the
  // parameter's DBG_VALUE may be described using a register defined by some
  // instruction in the entry block, in which case we shouldn't create an
  // entry value.
  if (DefinedRegs.count(MI.getDebugOperand(0).getReg()))
    return false;

  // TODO: Add support for parameters that have a pre-existing debug expressions
  // (e.g. fragments).
  if (MI.getDebugExpression()->getNumElements() > 0)
    return false;

  return true;
}

/// Collect all register defines (including aliases) for the given instruction.
static void collectRegDefs(const MachineInstr &MI, DefinedRegsSet &Regs,
                           const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() && MO.getReg())
      for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
        Regs.insert(*AI);
}

/// This routine records the entry values of function parameters. The values
/// could be used as backup values. If we loose the track of some unmodified
/// parameters, the backup values will be used as a primary locations.
void VarLocBasedLDV::recordEntryValue(const MachineInstr &MI,
                                       const DefinedRegsSet &DefinedRegs,
                                       OpenRangesSet &OpenRanges,
                                       VarLocMap &VarLocIDs) {
  if (TPC) {
    auto &TM = TPC->getTM<TargetMachine>();
    if (!TM.Options.ShouldEmitDebugEntryValues())
      return;
  }

  DebugVariable V(MI.getDebugVariable(), MI.getDebugExpression(),
                  MI.getDebugLoc()->getInlinedAt());

  if (!isEntryValueCandidate(MI, DefinedRegs) ||
      OpenRanges.getEntryValueBackup(V))
    return;

  LLVM_DEBUG(dbgs() << "Creating the backup entry location: "; MI.dump(););

  // Create the entry value and use it as a backup location until it is
  // valid. It is valid until a parameter is not changed.
  DIExpression *NewExpr =
      DIExpression::prepend(MI.getDebugExpression(), DIExpression::EntryValue);
  VarLoc EntryValLocAsBackup = VarLoc::CreateEntryBackupLoc(MI, LS, NewExpr);
  LocIndex EntryValLocID = VarLocIDs.insert(EntryValLocAsBackup);
  OpenRanges.insert(EntryValLocID, EntryValLocAsBackup);
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool VarLocBasedLDV::ExtendRanges(MachineFunction &MF, TargetPassConfig *TPC) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  if (!MF.getFunction().getSubprogram())
    // VarLocBaseLDV will already have removed all DBG_VALUEs.
    return false;

  // Skip functions from NoDebug compilation units.
  if (MF.getFunction().getSubprogram()->getUnit()->getEmissionKind() ==
      DICompileUnit::NoDebug)
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  TFI->getCalleeSaves(MF, CalleeSavedRegs);
  this->TPC = TPC;
  LS.initialize(MF);

  bool Changed = false;
  bool OLChanged = false;
  bool MBBJoined = false;

  VarLocMap VarLocIDs;         // Map VarLoc<>unique ID for use in bitvectors.
  OverlapMap OverlapFragments; // Map of overlapping variable fragments.
  OpenRangesSet OpenRanges(Alloc, OverlapFragments);
                              // Ranges that are open until end of bb.
  VarLocInMBB OutLocs;        // Ranges that exist beyond bb.
  VarLocInMBB InLocs;         // Ranges that are incoming after joining.
  TransferMap Transfers;      // DBG_VALUEs associated with transfers (such as
                              // spills, copies and restores).

  VarToFragments SeenFragments;

  // Blocks which are artificial, i.e. blocks which exclusively contain
  // instructions without locations, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;

  // Set of register defines that are seen when traversing the entry block
  // looking for debug entry value candidates.
  DefinedRegsSet DefinedRegs;

  // Only in the case of entry MBB collect DBG_VALUEs representing
  // function parameters in order to generate debug entry values for them.
  MachineBasicBlock &First_MBB = *(MF.begin());
  for (auto &MI : First_MBB) {
    collectRegDefs(MI, DefinedRegs, TRI);
    if (MI.isDebugValue())
      recordEntryValue(MI, DefinedRegs, OpenRanges, VarLocIDs);
  }

  // Initialize per-block structures and scan for fragment overlaps.
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      if (MI.isDebugValue())
        accumulateFragmentMap(MI, SeenFragments, OverlapFragments);

  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                              "OutLocs after initialization", dbgs()));

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    Worklist.push(RPONumber);
    ++RPONumber;
  }

  if (RPONumber > InputBBLimit) {
    unsigned NumInputDbgValues = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        if (MI.isDebugValue())
          ++NumInputDbgValues;
    if (NumInputDbgValues > InputDbgValueLimit) {
      LLVM_DEBUG(dbgs() << "Disabling VarLocBasedLDV: " << MF.getName()
                        << " has " << RPONumber << " basic blocks and "
                        << NumInputDbgValues
                        << " input DBG_VALUEs, exceeding limits.\n");
      return false;
    }
  }

  // This is a standard "union of predecessor outs" dataflow problem.
  // To solve it, we perform join() and process() using the two worklist method
  // until the ranges converge.
  // Ranges have converged when both worklists are empty.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    LLVM_DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      Worklist.pop();
      MBBJoined = join(*MBB, OutLocs, InLocs, VarLocIDs, Visited,
                       ArtificialBlocks);
      MBBJoined |= Visited.insert(MBB).second;
      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;
        // Now that we have started to extend ranges across BBs we need to
        // examine spill, copy and restore instructions to see whether they
        // operate with registers that correspond to user variables.
        // First load any pending inlocs.
        OpenRanges.insertFromLocSet(getVarLocsInMBB(MBB, InLocs), VarLocIDs);
        for (auto &MI : *MBB)
          process(MI, OpenRanges, VarLocIDs, Transfers);
        OLChanged |= transferTerminator(MBB, OpenRanges, OutLocs, VarLocIDs);

        LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                                    "OutLocs after propagating", dbgs()));
        LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs,
                                    "InLocs after propagating", dbgs()));

        if (OLChanged) {
          OLChanged = false;
          for (auto s : MBB->successors())
            if (OnPending.insert(s).second) {
              Pending.push(BBToOrder[s]);
            }
        }
      }
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  // Add any DBG_VALUE instructions created by location transfers.
  for (auto &TR : Transfers) {
    assert(!TR.TransferInst->isTerminator() &&
           "Cannot insert DBG_VALUE after terminator");
    MachineBasicBlock *MBB = TR.TransferInst->getParent();
    const VarLoc &VL = VarLocIDs[TR.LocationID];
    MachineInstr *MI = VL.BuildDbgValue(MF);
    MBB->insertAfterBundle(TR.TransferInst->getIterator(), MI);
  }
  Transfers.clear();

  // Deferred inlocs will not have had any DBG_VALUE insts created; do
  // that now.
  flushPendingLocs(InLocs, VarLocIDs);

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs, "Final OutLocs", dbgs()));
  LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs, "Final InLocs", dbgs()));
  return Changed;
}

LDVImpl *
llvm::makeVarLocBasedLiveDebugValues()
{
  return new VarLocBasedLDV();
}
