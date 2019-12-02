//===-- llvm/Analysis/TapirRaceDetect.h ----------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TapirRaceDetect is an LLVM pass that analyses Tapir tasks and dependences
// between memory accesses to find accesses that might race.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_TAPIRRACEDETECT_H
#define LLVM_ANALYSIS_TAPIRRACEDETECT_H

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

namespace llvm {

class Loop;
class LoopInfo;
class RuntimePointerChecking;
class ScalarEvolution;
class StratABIList;
class TargetLibraryInfo;
class TaskInfo;

/// RaceInfo
class RaceInfo {
public:
  // Possible conditions for a race:
  //
  // 1) Within the function, two instructions that might execute in parallel
  // access aliasing locations, and at least one performs a write.
  //
  // 2) An instruction reads or writes a location that might alias a global
  // variable or function argument.  In this case, the race would occur via an
  // ancestor of the invocation of this function.
  enum RaceType
    {
     None = 0,
     Local = 1, // Possible race via local pointer or control flow.
     ViaAncestorRef = 2, // Possible race with ref in caller (e.g., via function
                         // parameter or global)
     ViaAncestorMod = 4, // Possible race with mod inf caller (e.g., via function
                         // parameter or global)
     Opaque = 8, // Possible race via unknown program state (e.g., global data)
    };

  static RaceType setLocalRace(const RaceType RT) {
    return RaceType(static_cast<int>(RT) | static_cast<int>(Local));
  }
  static RaceType setRaceViaAncestorRef(const RaceType RT) {
    return RaceType(static_cast<int>(RT) | static_cast<int>(ViaAncestorRef));
  }
  static RaceType setRaceViaAncestorMod(const RaceType RT) {
    return RaceType(static_cast<int>(RT) | static_cast<int>(ViaAncestorMod));
  }
  static RaceType setOpaqueRace(const RaceType RT) {
    return RaceType(static_cast<int>(RT) | static_cast<int>(Opaque));
  }
  static RaceType clearOpaqueRace(const RaceType RT) {
    return RaceType(static_cast<int>(RT) & ~static_cast<int>(Opaque));
  }
  static RaceType unionRaceTypes(const RaceType RT1, const RaceType RT2) {
    return RaceType(static_cast<int>(RT1) | static_cast<int>(RT2));
  }

  static bool isRace(const RaceType RT) {
    return (RaceType::None != RT);
  }
  static bool isLocalRace(const RaceType RT) {
    return (static_cast<int>(RT) & static_cast<int>(RaceType::Local)) ==
      static_cast<int>(RaceType::Local);
  }
  static bool isRaceViaAncestor(const RaceType RT) {
    return isRaceViaAncestorRef(RT) || isRaceViaAncestorMod(RT);
  }
  static bool isRaceViaAncestorRef(const RaceType RT) {
    return (static_cast<int>(RT) &
            static_cast<int>(RaceType::ViaAncestorRef)) ==
      static_cast<int>(RaceType::ViaAncestorRef);
  }
  static bool isRaceViaAncestorMod(const RaceType RT) {
    return (static_cast<int>(RT) &
            static_cast<int>(RaceType::ViaAncestorMod)) ==
      static_cast<int>(RaceType::ViaAncestorMod);
  }
  static bool isOpaqueRace(const RaceType RT) {
    return (static_cast<int>(RT) & static_cast<int>(RaceType::Opaque)) ==
      static_cast<int>(RaceType::Opaque);
  }
  static void printRaceType(RaceInfo::RaceType RT, raw_ostream &OS) {
    if (RaceInfo::isLocalRace(RT))
      OS << "Local";
    if (RaceInfo::isRaceViaAncestor(RT)) {
      if (RaceInfo::isLocalRace(RT))
        OS << ", ";
      OS << "Via Ancestor";
      if (RaceInfo::isRaceViaAncestorMod(RT))
        OS << " Mod";
      if (RaceInfo::isRaceViaAncestorRef(RT))
        OS << " Ref";
    }
    if (RaceInfo::isOpaqueRace(RT)) {
      if (RaceInfo::isLocalRace(RT) || RaceInfo::isRaceViaAncestor(RT))
        OS << ", ";
      OS << "Opaque";
    }
  }

  using MemAccessInfo = PointerIntPair<const Value *, 1, bool>;

  // Struct to store data about a race.
  struct RaceData {
    MemAccessInfo Access = { nullptr, false };
    unsigned OperandNum = static_cast<unsigned>(-1);
    RaceType Type = RaceType::None;
    GeneralAccess Racer;

    RaceData() = default;
    RaceData(MemAccessInfo Access, unsigned OperandNum, const RaceType RT,
             GeneralAccess Racer = GeneralAccess())
        : Access(Access), OperandNum(OperandNum), Type(RT),
          Racer(Racer) {}

    const Value *getPtr() const { return Access.getPointer(); }
  };

  // Map to store race results.
  struct ResultTy :
    public DenseMap<const Instruction *, SmallVector<RaceData, 4>> {

    void recordRace(const Instruction *I, MemAccessInfo Access,
                    unsigned OperandNum, const RaceType RT,
                    const GeneralAccess &Racer) {
      if (!count(I)) {
        (*this)[I].push_back(RaceData(Access, OperandNum, RT, Racer));
        return;
      }
      for (RaceData &RD : (*this)[I])
        if ((RD.Access == Access) && (RD.OperandNum == OperandNum) &&
            (RD.Racer == Racer)) {
          RD.Type = unionRaceTypes(RD.Type, RT);
          return;
        }
      (*this)[I].push_back(RaceData(Access, OperandNum, RT, Racer));
    }
    void recordLocalRace(const GeneralAccess &GA,
                         const GeneralAccess &Racer) {
      recordRace(GA.I, MemAccessInfo(GA.getPtr(), GA.isMod()), GA.OperandNum,
                 RaceType::Local, Racer);
    }
    void recordRaceViaAncestorRef(const GeneralAccess &GA,
                                  const GeneralAccess &Racer) {
      recordRace(GA.I, MemAccessInfo(GA.getPtr(), GA.isMod()), GA.OperandNum,
                 RaceType::ViaAncestorRef, Racer);
    }
    void recordRaceViaAncestorMod(const GeneralAccess &GA,
                                  const GeneralAccess &Racer) {
      recordRace(GA.I, MemAccessInfo(GA.getPtr(), GA.isMod()), GA.OperandNum,
                 RaceType::ViaAncestorMod, Racer);
    }
    void recordOpaqueRace(const GeneralAccess &GA,
                          const GeneralAccess &Racer) {
      recordRace(GA.I, MemAccessInfo(GA.getPtr(), GA.isMod()), GA.OperandNum,
                 RaceType::Opaque, Racer);
    }

    RaceType getRaceType(const Instruction *I,
        const SmallPtrSetImpl<const Value *> *Filter = nullptr) const {
      if (!count(I))
        return RaceType::None;
      RaceType RT = RaceType::None;

      // Union the recorded race types
      for (RaceData &RD : lookup(I)) {
        if (Filter && RD.Racer.isValid() && Filter->count(RD.Racer.I))
          continue;
        RT = unionRaceTypes(RD.Type, RT);
      }
      return RT;
    }
  };
  using ObjectMRTy = DenseMap<const Value *, ModRefInfo>;
  using PtrChecksTy =
    DenseMap<Loop *, std::unique_ptr<RuntimePointerChecking>>;
  using AccessToUnderlyingObjMap =
    DenseMap<MemAccessInfo, SmallPtrSet<Value *, 1>>;

  using FilterTy = const SmallPtrSetImpl<const Value *>;

  RaceInfo(Function *F, DominatorTree &DT, LoopInfo &LI, TaskInfo &TI,
           DependenceInfo &DI, ScalarEvolution &SE,
           const TargetLibraryInfo *TLI);

  const SmallVectorImpl<RaceData> &getRaceData(const Instruction *I) {
    return Result[I];
  }

  RaceType getRaceType(const Instruction *I, FilterTy *Filter = nullptr) const {
    return Result.getRaceType(I, Filter);
  }
  bool mightRace(const Instruction *I, FilterTy *Filter = nullptr) const {
    return isRace(getRaceType(I, Filter));
  }
  bool mightRaceLocally(const Instruction *I,
                        FilterTy *Filter = nullptr) const {
    return isLocalRace(getRaceType(I, Filter));
  }
  bool mightRaceViaAncestor(const Instruction *I,
                            FilterTy *Filter = nullptr) const {
    return isRaceViaAncestor(getRaceType(I, Filter));
  }
  bool mightRaceViaAncestorRef(const Instruction *I,
                               FilterTy *Filter = nullptr) const {
    return isRaceViaAncestorRef(getRaceType(I, Filter));
  }
  bool mightRaceViaAncestorMod(const Instruction *I,
                               FilterTy *Filter = nullptr) const {
    return isRaceViaAncestorMod(getRaceType(I, Filter));
  }
  bool mightRaceOpaquely(const Instruction *I,
                         FilterTy *Filter = nullptr) const {
    return isOpaqueRace(getRaceType(I, Filter));
  }

  const ObjectMRTy &getObjectMRForRace() const {
    return ObjectMRForRace;
  }
  bool ObjectInvolvedInRace(const Value *V) const {
    return ObjectMRForRace.count(V);
  }
  ModRefInfo GetObjectMRForRace(const Value *V) const {
    if (!ObjectInvolvedInRace(V))
      return ModRefInfo::NoModRef;
    return ObjectMRForRace.lookup(V);
  }

  RaceType getOverallRaceType() const {
    RaceType RT = RaceType::None;
    for (auto Res : Result)
      for (auto &RD : Res.second)
        RT = unionRaceTypes(RT, RD.Type);
    return RT;
  }

  void getObjectsFor(Instruction *I, SmallPtrSetImpl<Value *> &Objects);
  void getObjectsFor(MemAccessInfo Access, SmallPtrSetImpl<Value *> &Objects);

  void print(raw_ostream &) const;

  AliasAnalysis *getAA() const { return DI.getAA(); }

private:
  void analyzeFunction();

  Function *F;

  // Analyses
  DominatorTree &DT;
  LoopInfo &LI;
  TaskInfo &TI;
  DependenceInfo &DI;
  ScalarEvolution &SE;
  const TargetLibraryInfo *TLI;

  ResultTy Result;
  // Map from underlying objects to mod/ref behavior necessary for potential
  // race.
  ObjectMRTy ObjectMRForRace;
  PtrChecksTy AllPtrRtChecks;

  AccessToUnderlyingObjMap AccessToObjs;
};

// AnalysisPass
class TapirRaceDetect : public AnalysisInfoMixin<TapirRaceDetect> {
public:
  using Result = RaceInfo;
  Result run(Function &F, FunctionAnalysisManager &FAM);

private:
  static AnalysisKey Key;
  friend struct AnalysisInfoMixin<TapirRaceDetect>;
}; // class TapirRaceDetect

// Printer pass
class TapirRaceDetectPrinterPass
  : public PassInfoMixin<TapirRaceDetectPrinterPass> {
public:
  TapirRaceDetectPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);

private:
  raw_ostream &OS;
}; // class TapirRaceDetectPrinterPass

// Legacy pass manager pass
class TapirRaceDetectWrapperPass : public FunctionPass {
public:
  static char ID;

  TapirRaceDetectWrapperPass() : FunctionPass(ID) {
    initializeTapirRaceDetectWrapperPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  void releaseMemory() override;
  void getAnalysisUsage(AnalysisUsage &) const override;
  void print(raw_ostream &, const Module * = nullptr) const override;
  RaceInfo &getRaceInfo() const;

private:
  std::unique_ptr<RaceInfo> Info;
}; // class TapirRaceDetectWrapperPass

// createTapirRaceDetectWrapperPass - This creates an instance of the
// TapirRaceDetect wrapper pass.
FunctionPass *createTapirRaceDetectWrapperPass();

} // namespace llvm

#endif
