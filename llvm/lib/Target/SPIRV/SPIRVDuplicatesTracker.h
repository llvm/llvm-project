//===-- SPIRVDuplicatesTracker.h - SPIR-V Duplicates Tracker ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// General infrastructure for keeping track of the values that according to
// the SPIR-V binary layout should be global to the whole module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVDUPLICATESTRACKER_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVDUPLICATESTRACKER_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

#include <type_traits>

namespace llvm {
namespace SPIRV {
// NOTE: using MapVector instead of DenseMap because it helps getting
// everything ordered in a stable manner for a price of extra (NumKeys)*PtrSize
// memory and expensive removals which do not happen anyway.
class DTSortableEntry : public MapVector<const MachineFunction *, Register> {
  SmallVector<DTSortableEntry *, 2> Deps;

  struct FlagsTy {
    unsigned IsFunc : 1;
    unsigned IsGV : 1;
    // NOTE: bit-field default init is a C++20 feature.
    FlagsTy() : IsFunc(0), IsGV(0) {}
  };
  FlagsTy Flags;

public:
  // Common hoisting utility doesn't support function, because their hoisting
  // require hoisting of params as well.
  bool getIsFunc() const { return Flags.IsFunc; }
  bool getIsGV() const { return Flags.IsGV; }
  void setIsFunc(bool V) { Flags.IsFunc = V; }
  void setIsGV(bool V) { Flags.IsGV = V; }

  const SmallVector<DTSortableEntry *, 2> &getDeps() const { return Deps; }
  void addDep(DTSortableEntry *E) { Deps.push_back(E); }
};
} // namespace SPIRV

template <typename KeyTy> class SPIRVDuplicatesTrackerBase {
public:
  // NOTE: using MapVector instead of DenseMap helps getting everything ordered
  // in a stable manner for a price of extra (NumKeys)*PtrSize memory and
  // expensive removals which don't happen anyway.
  using StorageTy = MapVector<KeyTy, SPIRV::DTSortableEntry>;

private:
  StorageTy Storage;

public:
  void add(KeyTy V, const MachineFunction *MF, Register R) {
    if (find(V, MF).isValid())
      return;

    Storage[V][MF] = R;
    if (std::is_same<Function,
                     typename std::remove_const<
                         typename std::remove_pointer<KeyTy>::type>::type>() ||
        std::is_same<Argument,
                     typename std::remove_const<
                         typename std::remove_pointer<KeyTy>::type>::type>())
      Storage[V].setIsFunc(true);
    if (std::is_same<GlobalVariable,
                     typename std::remove_const<
                         typename std::remove_pointer<KeyTy>::type>::type>())
      Storage[V].setIsGV(true);
  }

  Register find(KeyTy V, const MachineFunction *MF) const {
    auto iter = Storage.find(V);
    if (iter != Storage.end()) {
      auto Map = iter->second;
      auto iter2 = Map.find(MF);
      if (iter2 != Map.end())
        return iter2->second;
    }
    return Register();
  }

  const StorageTy &getAllUses() const { return Storage; }

private:
  StorageTy &getAllUses() { return Storage; }

  // The friend class needs to have access to the internal storage
  // to be able to build dependency graph, can't declare only one
  // function a 'friend' due to the incomplete declaration at this point
  // and mutual dependency problems.
  friend class SPIRVGeneralDuplicatesTracker;
};

template <typename T>
class SPIRVDuplicatesTracker : public SPIRVDuplicatesTrackerBase<const T *> {};

class SPIRVGeneralDuplicatesTracker {
  SPIRVDuplicatesTracker<Type> TT;
  SPIRVDuplicatesTracker<Constant> CT;
  SPIRVDuplicatesTracker<GlobalVariable> GT;
  SPIRVDuplicatesTracker<Function> FT;
  SPIRVDuplicatesTracker<Argument> AT;

  // NOTE: using MOs instead of regs to get rid of MF dependency to be able
  // to use flat data structure.
  // NOTE: replacing DenseMap with MapVector doesn't affect overall correctness
  // but makes LITs more stable, should prefer DenseMap still due to
  // significant perf difference.
  using SPIRVReg2EntryTy =
      MapVector<MachineOperand *, SPIRV::DTSortableEntry *>;

  template <typename T>
  void prebuildReg2Entry(SPIRVDuplicatesTracker<T> &DT,
                         SPIRVReg2EntryTy &Reg2Entry);

public:
  void buildDepsGraph(std::vector<SPIRV::DTSortableEntry *> &Graph,
                      MachineModuleInfo *MMI);

  void add(const Type *T, const MachineFunction *MF, Register R) {
    TT.add(T, MF, R);
  }

  void add(const Constant *C, const MachineFunction *MF, Register R) {
    CT.add(C, MF, R);
  }

  void add(const GlobalVariable *GV, const MachineFunction *MF, Register R) {
    GT.add(GV, MF, R);
  }

  void add(const Function *F, const MachineFunction *MF, Register R) {
    FT.add(F, MF, R);
  }

  void add(const Argument *Arg, const MachineFunction *MF, Register R) {
    AT.add(Arg, MF, R);
  }

  Register find(const Type *T, const MachineFunction *MF) {
    return TT.find(const_cast<Type *>(T), MF);
  }

  Register find(const Constant *C, const MachineFunction *MF) {
    return CT.find(const_cast<Constant *>(C), MF);
  }

  Register find(const GlobalVariable *GV, const MachineFunction *MF) {
    return GT.find(const_cast<GlobalVariable *>(GV), MF);
  }

  Register find(const Function *F, const MachineFunction *MF) {
    return FT.find(const_cast<Function *>(F), MF);
  }

  Register find(const Argument *Arg, const MachineFunction *MF) {
    return AT.find(const_cast<Argument *>(Arg), MF);
  }
};
} // namespace llvm
#endif
