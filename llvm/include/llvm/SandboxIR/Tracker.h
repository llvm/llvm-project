//===- Tracker.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the component of SandboxIR that tracks all changes made to its
// state, such that we can revert the state when needed.
//
// Tracking changes
// ----------------
// The user needs to call `Tracker::save()` to enable tracking changes
// made to SandboxIR. From that point on, any change made to SandboxIR, will
// automatically create a change tracking object and register it with the
// tracker. IR-change objects are subclasses of `IRChangeBase` and get
// registered with the `Tracker::track()` function. The change objects
// are saved in the order they are registered with the tracker and are stored in
// the `Tracker::Changes` vector. All of this is done transparently to
// the user.
//
// Reverting changes
// -----------------
// Calling `Tracker::revert()` will restore the state saved when
// `Tracker::save()` was called. Internally this goes through the
// change objects in `Tracker::Changes` in reverse order, calling their
// `IRChangeBase::revert()` function one by one.
//
// Accepting changes
// -----------------
// The user needs to either revert or accept changes before the tracker object
// is destroyed. This is enforced in the tracker's destructor.
// This is the job of `Tracker::accept()`. Internally this will go
// through the change objects in `Tracker::Changes` in order, calling
// `IRChangeBase::accept()`.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_TRACKER_H
#define LLVM_SANDBOXIR_TRACKER_H

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/SandboxIR/Use.h"
#include "llvm/Support/Debug.h"
#include <memory>

namespace llvm::sandboxir {

class BasicBlock;
class CallBrInst;
class LoadInst;
class StoreInst;
class Instruction;
class Tracker;
class AllocaInst;
class CatchSwitchInst;
class SwitchInst;
class ConstantInt;
class ShuffleVectorInst;
class CmpInst;
class GlobalVariable;

#ifndef NDEBUG

/// A class that saves hashes and textual IR snapshots of functions in a
/// SandboxIR Context, and does hash comparison when `expectNoDiff` is called.
/// If hashes differ, it prints textual IR for both old and new versions to
/// aid debugging.
///
/// This is used as an additional debug check when reverting changes to
/// SandboxIR, to verify the reverted state matches the initial state.
class IRSnapshotChecker {
  Context &Ctx;

  // A snapshot of textual IR for a function, with a hash for quick comparison.
  struct FunctionSnapshot {
    llvm::stable_hash Hash;
    std::string TextualIR;
  };

  // A snapshot for each llvm::Function found in every module in the SandboxIR
  // Context. In practice there will always be one module, but sandbox IR
  // save/restore ops work at the Context level, so we must take the full state
  // into account.
  using ContextSnapshot = DenseMap<const llvm::Function *, FunctionSnapshot>;

  ContextSnapshot OrigContextSnapshot;

  // Dumps to a string the textual IR for a single Function.
  std::string dumpIR(const llvm::Function &F) const;

  // Returns a snapshot of all the modules in the sandbox IR context.
  ContextSnapshot takeSnapshot() const;

  // Compares two snapshots and returns true if they differ.
  bool diff(const ContextSnapshot &Orig, const ContextSnapshot &Curr) const;

public:
  IRSnapshotChecker(Context &Ctx) : Ctx(Ctx) {}

  /// Saves a snapshot of the current state. If there was any previous snapshot,
  /// it will be replaced with the new one.
  void save();

  /// Checks current state against saved state, crashes if different.
  void expectNoDiff();
};

#endif // NDEBUG

/// The base class for IR Change classes.
class IRChangeBase {
protected:
  friend class Tracker; // For Parent.

public:
  /// This runs when changes get reverted.
  virtual void revert(Tracker &Tracker) = 0;
  /// This runs when changes get accepted.
  virtual void accept() = 0;
  virtual ~IRChangeBase() = default;
#ifndef NDEBUG
  virtual void dump(raw_ostream &OS) const = 0;
  LLVM_DUMP_METHOD virtual void dump() const = 0;
  friend raw_ostream &operator<<(raw_ostream &OS, const IRChangeBase &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

/// Tracks the change of the source Value of a sandboxir::Use.
class UseSet : public IRChangeBase {
  Use U;
  Value *OrigV = nullptr;

public:
  UseSet(const Use &U) : U(U), OrigV(U.get()) {}
  void revert(Tracker &Tracker) final { U.set(OrigV); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "UseSet"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

class PHIRemoveIncoming : public IRChangeBase {
  PHINode *PHI;
  unsigned RemovedIdx;
  Value *RemovedV;
  BasicBlock *RemovedBB;

public:
  PHIRemoveIncoming(PHINode *PHI, unsigned RemovedIdx);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "PHISetIncoming"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

class PHIAddIncoming : public IRChangeBase {
  PHINode *PHI;
  unsigned Idx;

public:
  PHIAddIncoming(PHINode *PHI);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "PHISetIncoming"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

class CmpSwapOperands : public IRChangeBase {
  CmpInst *Cmp;

public:
  CmpSwapOperands(CmpInst *Cmp);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "CmpSwapOperands"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

/// Tracks swapping a Use with another Use.
class UseSwap : public IRChangeBase {
  Use ThisUse;
  Use OtherUse;

public:
  UseSwap(const Use &ThisUse, const Use &OtherUse)
      : ThisUse(ThisUse), OtherUse(OtherUse) {
    assert(ThisUse.getUser() == OtherUse.getUser() && "Expected same user!");
  }
  void revert(Tracker &Tracker) final { ThisUse.swap(OtherUse); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "UseSwap"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

class EraseFromParent : public IRChangeBase {
  /// Contains all the data we need to restore an "erased" (i.e., detached)
  /// instruction: the instruction itself and its operands in order.
  struct InstrAndOperands {
    /// The operands that got dropped.
    SmallVector<llvm::Value *> Operands;
    /// The instruction that got "erased".
    llvm::Instruction *LLVMI;
  };
  /// The instruction data is in reverse program order, which helps create the
  /// original program order during revert().
  SmallVector<InstrAndOperands> InstrData;
  /// This is either the next Instruction in the stream, or the parent
  /// BasicBlock if at the end of the BB.
  PointerUnion<llvm::Instruction *, llvm::BasicBlock *> NextLLVMIOrBB;
  /// We take ownership of the "erased" instruction.
  std::unique_ptr<sandboxir::Value> ErasedIPtr;

public:
  EraseFromParent(std::unique_ptr<sandboxir::Value> &&IPtr);
  void revert(Tracker &Tracker) final;
  void accept() final;
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "EraseFromParent"; }
  LLVM_DUMP_METHOD void dump() const final;
  friend raw_ostream &operator<<(raw_ostream &OS, const EraseFromParent &C) {
    C.dump(OS);
    return OS;
  }
#endif
};

class RemoveFromParent : public IRChangeBase {
  /// The instruction that is about to get removed.
  Instruction *RemovedI = nullptr;
  /// This is either the next instr, or the parent BB if at the end of the BB.
  PointerUnion<Instruction *, BasicBlock *> NextInstrOrBB;

public:
  RemoveFromParent(Instruction *RemovedI);
  void revert(Tracker &Tracker) final;
  void accept() final {};
  Instruction *getInstruction() const { return RemovedI; }
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "RemoveFromParent"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

/// This class can be used for tracking most instruction setters.
/// The two template arguments are:
/// - GetterFn: The getter member function pointer (e.g., `&Foo::get`)
/// - SetterFn: The setter member function pointer (e.g., `&Foo::set`)
/// Upon construction, it saves a copy of the original value by calling the
/// getter function. Revert sets the value back to the one saved, using the
/// setter function provided.
///
/// Example:
///  Tracker.track(std::make_unique<
///                GenericSetter<&FooInst::get, &FooInst::set>>(I, Tracker));
///
template <auto GetterFn, auto SetterFn>
class GenericSetter final : public IRChangeBase {
  /// Traits for getting the class type from GetterFn type.
  template <typename> struct GetClassTypeFromGetter;
  template <typename RetT, typename ClassT>
  struct GetClassTypeFromGetter<RetT (ClassT::*)() const> {
    using ClassType = ClassT;
  };
  using InstrT = typename GetClassTypeFromGetter<decltype(GetterFn)>::ClassType;
  using SavedValT = std::invoke_result_t<decltype(GetterFn), InstrT>;
  InstrT *I;
  SavedValT OrigVal;

public:
  GenericSetter(InstrT *I) : I(I), OrigVal((I->*GetterFn)()) {}
  void revert(Tracker &Tracker) final { (I->*SetterFn)(OrigVal); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "GenericSetter"; }
  LLVM_DUMP_METHOD void dump() const final {
    dump(dbgs());
    dbgs() << "\n";
  }
#endif
};

/// Similar to GenericSetter but the setters/getters have an index as their
/// first argument. This is commont in cases like: getOperand(unsigned Idx)
template <auto GetterFn, auto SetterFn>
class GenericSetterWithIdx final : public IRChangeBase {
  /// Helper for getting the class type from the getter
  template <typename ClassT, typename RetT>
  static ClassT getClassTypeFromGetter(RetT (ClassT::*Fn)(unsigned) const);
  template <typename ClassT, typename RetT>
  static ClassT getClassTypeFromGetter(RetT (ClassT::*Fn)(unsigned));

  using InstrT = decltype(getClassTypeFromGetter(GetterFn));
  using SavedValT = std::invoke_result_t<decltype(GetterFn), InstrT, unsigned>;
  InstrT *I;
  SavedValT OrigVal;
  unsigned Idx;

public:
  GenericSetterWithIdx(InstrT *I, unsigned Idx)
      : I(I), OrigVal((I->*GetterFn)(Idx)), Idx(Idx) {}
  void revert(Tracker &Tracker) final { (I->*SetterFn)(Idx, OrigVal); }
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "GenericSetterWithIdx"; }
  LLVM_DUMP_METHOD void dump() const final {
    dump(dbgs());
    dbgs() << "\n";
  }
#endif
};

class CatchSwitchAddHandler : public IRChangeBase {
  CatchSwitchInst *CSI;
  unsigned HandlerIdx;

public:
  CatchSwitchAddHandler(CatchSwitchInst *CSI);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "CatchSwitchAddHandler"; }
  LLVM_DUMP_METHOD void dump() const final {
    dump(dbgs());
    dbgs() << "\n";
  }
#endif // NDEBUG
};

class SwitchAddCase : public IRChangeBase {
  SwitchInst *Switch;
  ConstantInt *Val;

public:
  SwitchAddCase(SwitchInst *Switch, ConstantInt *Val)
      : Switch(Switch), Val(Val) {}
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "SwitchAddCase"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

class SwitchRemoveCase : public IRChangeBase {
  SwitchInst *Switch;
  struct Case {
    ConstantInt *Val;
    BasicBlock *Dest;
  };
  SmallVector<Case> Cases;

public:
  SwitchRemoveCase(SwitchInst *Switch);

  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "SwitchRemoveCase"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

class MoveInstr : public IRChangeBase {
  /// The instruction that moved.
  Instruction *MovedI;
  /// This is either the next instruction in the block, or the parent BB if at
  /// the end of the BB.
  PointerUnion<Instruction *, BasicBlock *> NextInstrOrBB;

public:
  MoveInstr(sandboxir::Instruction *I);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "MoveInstr"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

class InsertIntoBB final : public IRChangeBase {
  Instruction *InsertedI = nullptr;

public:
  InsertIntoBB(Instruction *InsertedI);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "InsertIntoBB"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif // NDEBUG
};

class CreateAndInsertInst final : public IRChangeBase {
  Instruction *NewI = nullptr;

public:
  CreateAndInsertInst(Instruction *NewI) : NewI(NewI) {}
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "CreateAndInsertInst"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

class ShuffleVectorSetMask final : public IRChangeBase {
  ShuffleVectorInst *SVI;
  SmallVector<int, 8> PrevMask;

public:
  ShuffleVectorSetMask(ShuffleVectorInst *SVI);
  void revert(Tracker &Tracker) final;
  void accept() final {}
#ifndef NDEBUG
  void dump(raw_ostream &OS) const final { OS << "ShuffleVectorSetMask"; }
  LLVM_DUMP_METHOD void dump() const final;
#endif
};

/// The tracker collects all the change objects and implements the main API for
/// saving / reverting / accepting.
class Tracker {
public:
  enum class TrackerState {
    Disabled, ///> Tracking is disabled
    Record,   ///> Tracking changes
  };

private:
  /// The list of changes that are being tracked.
  SmallVector<std::unique_ptr<IRChangeBase>> Changes;
  /// The current state of the tracker.
  TrackerState State = TrackerState::Disabled;
  Context &Ctx;

#ifndef NDEBUG
  IRSnapshotChecker SnapshotChecker;
#endif

public:
#ifndef NDEBUG
  /// Helps catch bugs where we are creating new change objects while in the
  /// middle of creating other change objects.
  bool InMiddleOfCreatingChange = false;
#endif // NDEBUG

  explicit Tracker(Context &Ctx)
      : Ctx(Ctx)
#ifndef NDEBUG
        ,
        SnapshotChecker(Ctx)
#endif
  {
  }

  ~Tracker();
  Context &getContext() const { return Ctx; }
  /// Record \p Change and take ownership. This is the main function used to
  /// track Sandbox IR changes.
  void track(std::unique_ptr<IRChangeBase> &&Change) {
    assert(State == TrackerState::Record && "The tracker should be tracking!");
#ifndef NDEBUG
    assert(!InMiddleOfCreatingChange &&
           "We are in the middle of creating another change!");
    if (isTracking())
      InMiddleOfCreatingChange = true;
#endif // NDEBUG
    Changes.push_back(std::move(Change));

#ifndef NDEBUG
    InMiddleOfCreatingChange = false;
#endif
  }
  /// A convenience wrapper for `track()` that constructs and tracks the Change
  /// object if tracking is enabled. \Returns true if tracking is enabled.
  template <typename ChangeT, typename... ArgsT>
  bool emplaceIfTracking(ArgsT... Args) {
    if (!isTracking())
      return false;
    track(std::make_unique<ChangeT>(Args...));
    return true;
  }
  /// \Returns true if the tracker is recording changes.
  bool isTracking() const { return State == TrackerState::Record; }
  /// \Returns the current state of the tracker.
  TrackerState getState() const { return State; }
  /// Turns on IR tracking.
  void save();
  /// Stops tracking and accept changes.
  void accept();
  /// Stops tracking and reverts to saved state.
  void revert();

#ifndef NDEBUG
  void dump(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
  friend raw_ostream &operator<<(raw_ostream &OS, const Tracker &Tracker) {
    Tracker.dump(OS);
    return OS;
  }
#endif // NDEBUG
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_TRACKER_H
