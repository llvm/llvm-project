//===- DFAPacketizerEmitter.cpp - Packetization DFA for a VLIW machine ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class parses the Schedule.td file and produces an API that can be used
// to reason about whether an instruction can be added to a packet on a VLIW
// architecture. The class internally generates a deterministic finite
// automaton (DFA) that models all possible mappings of machine instructions
// to functional units as instructions are added to a packet.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dfa-emitter"

#include "CodeGenTarget.h"
#include "DFAEmitter.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

// --------------------------------------------------------------------
// Definitions shared between DFAPacketizer.cpp and DFAPacketizerEmitter.cpp

// DFA_MAX_RESTERMS * DFA_MAX_RESOURCES must fit within sizeof DFAInput.
// This is verified in DFAPacketizer.cpp:DFAPacketizer::DFAPacketizer.
//
// e.g. terms x resource bit combinations that fit in uint32_t:
//      4 terms x 8  bits = 32 bits
//      3 terms x 10 bits = 30 bits
//      2 terms x 16 bits = 32 bits
//
// e.g. terms x resource bit combinations that fit in uint64_t:
//      8 terms x 8  bits = 64 bits
//      7 terms x 9  bits = 63 bits
//      6 terms x 10 bits = 60 bits
//      5 terms x 12 bits = 60 bits
//      4 terms x 16 bits = 64 bits <--- current
//      3 terms x 21 bits = 63 bits
//      2 terms x 32 bits = 64 bits
//
#define DFA_MAX_RESTERMS        4   // The max # of AND'ed resource terms.
#define DFA_MAX_RESOURCES       16  // The max # of resource bits in one term.

typedef uint64_t                DFAInput;
typedef int64_t                 DFAStateInput;
#define DFA_TBLTYPE             "int64_t" // For generating DFAStateInputTable.

namespace {

  DFAInput addDFAFuncUnits(DFAInput Inp, unsigned FuncUnits) {
    return (Inp << DFA_MAX_RESOURCES) | FuncUnits;
  }

  /// Return the DFAInput for an instruction class input vector.
  /// This function is used in both DFAPacketizer.cpp and in
  /// DFAPacketizerEmitter.cpp.
  DFAInput getDFAInsnInput(const std::vector<unsigned> &InsnClass) {
    DFAInput InsnInput = 0;
    assert((InsnClass.size() <= DFA_MAX_RESTERMS) &&
           "Exceeded maximum number of DFA terms");
    for (auto U : InsnClass)
      InsnInput = addDFAFuncUnits(InsnInput, U);
    return InsnInput;
  }

} // end anonymous namespace

// --------------------------------------------------------------------

#ifndef NDEBUG
// To enable debugging, run llvm-tblgen with: "-debug-only dfa-emitter".
//
// dbgsInsnClass - When debugging, print instruction class stages.
//
void dbgsInsnClass(const std::vector<unsigned> &InsnClass);
//
// dbgsStateInfo - When debugging, print the set of state info.
//
void dbgsStateInfo(const std::set<unsigned> &stateInfo);
//
// dbgsIndent - When debugging, indent by the specified amount.
//
void dbgsIndent(unsigned indent);
#endif

//
// class DFAPacketizerEmitter: class that generates and prints out the DFA
// for resource tracking.
//
namespace {

class DFAPacketizerEmitter {
private:
  std::string TargetName;
  //
  // allInsnClasses is the set of all possible resources consumed by an
  // InstrStage.
  //
  std::vector<std::vector<unsigned>> allInsnClasses;
  RecordKeeper &Records;

public:
  DFAPacketizerEmitter(RecordKeeper &R);

  //
  // collectAllFuncUnits - Construct a map of function unit names to bits.
  //
  int collectAllFuncUnits(std::vector<Record*> &ProcItinList,
                           std::map<std::string, unsigned> &FUNameToBitsMap,
                           int &maxResources,
                           raw_ostream &OS);

  //
  // collectAllComboFuncs - Construct a map from a combo function unit bit to
  //                        the bits of all included functional units.
  //
  int collectAllComboFuncs(std::vector<Record*> &ComboFuncList,
                           std::map<std::string, unsigned> &FUNameToBitsMap,
                           std::map<unsigned, unsigned> &ComboBitToBitsMap,
                           raw_ostream &OS);

  //
  // collectOneInsnClass - Populate allInsnClasses with one instruction class.
  //
  int collectOneInsnClass(const std::string &ProcName,
                           std::vector<Record*> &ProcItinList,
                           std::map<std::string, unsigned> &FUNameToBitsMap,
                           Record *ItinData,
                           raw_ostream &OS);

  //
  // collectAllInsnClasses - Populate allInsnClasses which is a set of units
  // used in each stage.
  //
  int collectAllInsnClasses(const std::string &ProcName,
                           std::vector<Record*> &ProcItinList,
                           std::map<std::string, unsigned> &FUNameToBitsMap,
                           std::vector<Record*> &ItinDataList,
                           int &maxStages,
                           raw_ostream &OS);

  // Emit code for a subset of itineraries.
  void emitForItineraries(raw_ostream &OS,
                          std::vector<Record *> &ProcItinList,
                          std::string DFAName);

  void run(raw_ostream &OS);
};
} // end anonymous namespace

#ifndef NDEBUG
// To enable debugging, run llvm-tblgen with: "-debug-only dfa-emitter".
//
// dbgsInsnClass - When debugging, print instruction class stages.
//
void dbgsInsnClass(const std::vector<unsigned> &InsnClass) {
  LLVM_DEBUG(dbgs() << "InsnClass: ");
  for (unsigned i = 0; i < InsnClass.size(); ++i) {
    if (i > 0) {
      LLVM_DEBUG(dbgs() << ", ");
    }
    LLVM_DEBUG(dbgs() << "0x" << Twine::utohexstr(InsnClass[i]));
  }
  DFAInput InsnInput = getDFAInsnInput(InsnClass);
  LLVM_DEBUG(dbgs() << " (input: 0x" << Twine::utohexstr(InsnInput) << ")");
}

//
// dbgsIndent - When debugging, indent by the specified amount.
//
void dbgsIndent(unsigned indent) {
  for (unsigned i = 0; i < indent; ++i) {
    LLVM_DEBUG(dbgs() << " ");
  }
}
#endif // NDEBUG

DFAPacketizerEmitter::DFAPacketizerEmitter(RecordKeeper &R):
  TargetName(CodeGenTarget(R).getName()), Records(R) {}

//
// collectAllFuncUnits - Construct a map of function unit names to bits.
//
int DFAPacketizerEmitter::collectAllFuncUnits(
                            std::vector<Record*> &ProcItinList,
                            std::map<std::string, unsigned> &FUNameToBitsMap,
                            int &maxFUs,
                            raw_ostream &OS) {
  LLVM_DEBUG(dbgs() << "-------------------------------------------------------"
                       "----------------------\n");
  LLVM_DEBUG(dbgs() << "collectAllFuncUnits");
  LLVM_DEBUG(dbgs() << " (" << ProcItinList.size() << " itineraries)\n");

  int totalFUs = 0;
  // Parse functional units for all the itineraries.
  for (unsigned i = 0, N = ProcItinList.size(); i < N; ++i) {
    Record *Proc = ProcItinList[i];
    std::vector<Record*> FUs = Proc->getValueAsListOfDefs("FU");

    LLVM_DEBUG(dbgs() << "    FU:" << i << " (" << FUs.size() << " FUs) "
                      << Proc->getName());

    // Convert macros to bits for each stage.
    unsigned numFUs = FUs.size();
    for (unsigned j = 0; j < numFUs; ++j) {
      assert ((j < DFA_MAX_RESOURCES) &&
                      "Exceeded maximum number of representable resources");
      unsigned FuncResources = (unsigned) (1U << j);
      FUNameToBitsMap[FUs[j]->getName()] = FuncResources;
      LLVM_DEBUG(dbgs() << " " << FUs[j]->getName() << ":0x"
                        << Twine::utohexstr(FuncResources));
    }
    if (((int) numFUs) > maxFUs) {
      maxFUs = numFUs;
    }
    totalFUs += numFUs;
    LLVM_DEBUG(dbgs() << "\n");
  }
  return totalFUs;
}

//
// collectAllComboFuncs - Construct a map from a combo function unit bit to
//                        the bits of all included functional units.
//
int DFAPacketizerEmitter::collectAllComboFuncs(
                            std::vector<Record*> &ComboFuncList,
                            std::map<std::string, unsigned> &FUNameToBitsMap,
                            std::map<unsigned, unsigned> &ComboBitToBitsMap,
                            raw_ostream &OS) {
  LLVM_DEBUG(dbgs() << "-------------------------------------------------------"
                       "----------------------\n");
  LLVM_DEBUG(dbgs() << "collectAllComboFuncs");
  LLVM_DEBUG(dbgs() << " (" << ComboFuncList.size() << " sets)\n");

  int numCombos = 0;
  for (unsigned i = 0, N = ComboFuncList.size(); i < N; ++i) {
    Record *Func = ComboFuncList[i];
    std::vector<Record*> FUs = Func->getValueAsListOfDefs("CFD");

    LLVM_DEBUG(dbgs() << "    CFD:" << i << " (" << FUs.size() << " combo FUs) "
                      << Func->getName() << "\n");

    // Convert macros to bits for each stage.
    for (unsigned j = 0, N = FUs.size(); j < N; ++j) {
      assert ((j < DFA_MAX_RESOURCES) &&
                      "Exceeded maximum number of DFA resources");
      Record *FuncData = FUs[j];
      Record *ComboFunc = FuncData->getValueAsDef("TheComboFunc");
      const std::vector<Record*> &FuncList =
                                   FuncData->getValueAsListOfDefs("FuncList");
      const std::string &ComboFuncName = ComboFunc->getName();
      unsigned ComboBit = FUNameToBitsMap[ComboFuncName];
      unsigned ComboResources = ComboBit;
      LLVM_DEBUG(dbgs() << "      combo: " << ComboFuncName << ":0x"
                        << Twine::utohexstr(ComboResources) << "\n");
      for (unsigned k = 0, M = FuncList.size(); k < M; ++k) {
        std::string FuncName = FuncList[k]->getName();
        unsigned FuncResources = FUNameToBitsMap[FuncName];
        LLVM_DEBUG(dbgs() << "        " << FuncName << ":0x"
                          << Twine::utohexstr(FuncResources) << "\n");
        ComboResources |= FuncResources;
      }
      ComboBitToBitsMap[ComboBit] = ComboResources;
      numCombos++;
      LLVM_DEBUG(dbgs() << "          => combo bits: " << ComboFuncName << ":0x"
                        << Twine::utohexstr(ComboBit) << " = 0x"
                        << Twine::utohexstr(ComboResources) << "\n");
    }
  }
  return numCombos;
}

//
// collectOneInsnClass - Populate allInsnClasses with one instruction class
//
int DFAPacketizerEmitter::collectOneInsnClass(const std::string &ProcName,
                        std::vector<Record*> &ProcItinList,
                        std::map<std::string, unsigned> &FUNameToBitsMap,
                        Record *ItinData,
                        raw_ostream &OS) {
  const std::vector<Record*> &StageList =
    ItinData->getValueAsListOfDefs("Stages");

  // The number of stages.
  unsigned NStages = StageList.size();

  LLVM_DEBUG(dbgs() << "    " << ItinData->getValueAsDef("TheClass")->getName()
                    << "\n");

  std::vector<unsigned> UnitBits;

  // Compute the bitwise or of each unit used in this stage.
  for (unsigned i = 0; i < NStages; ++i) {
    const Record *Stage = StageList[i];

    // Get unit list.
    const std::vector<Record*> &UnitList =
      Stage->getValueAsListOfDefs("Units");

    LLVM_DEBUG(dbgs() << "        stage:" << i << " [" << UnitList.size()
                      << " units]:");
    unsigned dbglen = 26;  // cursor after stage dbgs

    // Compute the bitwise or of each unit used in this stage.
    unsigned UnitBitValue = 0;
    for (unsigned j = 0, M = UnitList.size(); j < M; ++j) {
      // Conduct bitwise or.
      std::string UnitName = UnitList[j]->getName();
      LLVM_DEBUG(dbgs() << " " << j << ":" << UnitName);
      dbglen += 3 + UnitName.length();
      assert(FUNameToBitsMap.count(UnitName));
      UnitBitValue |= FUNameToBitsMap[UnitName];
    }

    if (UnitBitValue != 0)
      UnitBits.push_back(UnitBitValue);

    while (dbglen <= 64) {   // line up bits dbgs
        dbglen += 8;
        LLVM_DEBUG(dbgs() << "\t");
    }
    LLVM_DEBUG(dbgs() << " (bits: 0x" << Twine::utohexstr(UnitBitValue)
                      << ")\n");
  }

  if (!UnitBits.empty())
    allInsnClasses.push_back(UnitBits);

  LLVM_DEBUG({
    dbgs() << "        ";
    dbgsInsnClass(UnitBits);
    dbgs() << "\n";
  });

  return NStages;
}

//
// collectAllInsnClasses - Populate allInsnClasses which is a set of units
// used in each stage.
//
int DFAPacketizerEmitter::collectAllInsnClasses(const std::string &ProcName,
                            std::vector<Record*> &ProcItinList,
                            std::map<std::string, unsigned> &FUNameToBitsMap,
                            std::vector<Record*> &ItinDataList,
                            int &maxStages,
                            raw_ostream &OS) {
  // Collect all instruction classes.
  unsigned M = ItinDataList.size();

  int numInsnClasses = 0;
  LLVM_DEBUG(dbgs() << "-------------------------------------------------------"
                       "----------------------\n"
                    << "collectAllInsnClasses " << ProcName << " (" << M
                    << " classes)\n");

  // Collect stages for each instruction class for all itinerary data
  for (unsigned j = 0; j < M; j++) {
    Record *ItinData = ItinDataList[j];
    int NStages = collectOneInsnClass(ProcName, ProcItinList,
                                      FUNameToBitsMap, ItinData, OS);
    if (NStages > maxStages) {
      maxStages = NStages;
    }
    numInsnClasses++;
  }
  return numInsnClasses;
}

//
// Run the worklist algorithm to generate the DFA.
//
void DFAPacketizerEmitter::run(raw_ostream &OS) {
  OS << "\n"
     << "#include \"llvm/CodeGen/DFAPacketizer.h\"\n";
  OS << "namespace llvm {\n";

  OS << "\n// Input format:\n";
  OS << "#define DFA_MAX_RESTERMS        " << DFA_MAX_RESTERMS
     << "\t// maximum AND'ed resource terms\n";
  OS << "#define DFA_MAX_RESOURCES       " << DFA_MAX_RESOURCES
     << "\t// maximum resource bits in one term\n";

  // Collect processor iteraries.
  std::vector<Record*> ProcItinList =
    Records.getAllDerivedDefinitions("ProcessorItineraries");

  std::unordered_map<std::string, std::vector<Record*>> ItinsByNamespace;
  for (Record *R : ProcItinList)
    ItinsByNamespace[R->getValueAsString("PacketizerNamespace")].push_back(R);

  for (auto &KV : ItinsByNamespace)
    emitForItineraries(OS, KV.second, KV.first);
  OS << "} // end namespace llvm\n";
}

void DFAPacketizerEmitter::emitForItineraries(
    raw_ostream &OS, std::vector<Record *> &ProcItinList,
    std::string DFAName) {
  //
  // Collect the Functional units.
  //
  std::map<std::string, unsigned> FUNameToBitsMap;
  int maxResources = 0;
  collectAllFuncUnits(ProcItinList,
                              FUNameToBitsMap, maxResources, OS);

  //
  // Collect the Combo Functional units.
  //
  std::map<unsigned, unsigned> ComboBitToBitsMap;
  std::vector<Record*> ComboFuncList =
    Records.getAllDerivedDefinitions("ComboFuncUnits");
  collectAllComboFuncs(ComboFuncList, FUNameToBitsMap, ComboBitToBitsMap, OS);

  //
  // Collect the itineraries.
  //
  int maxStages = 0;
  int numInsnClasses = 0;
  for (unsigned i = 0, N = ProcItinList.size(); i < N; i++) {
    Record *Proc = ProcItinList[i];

    // Get processor itinerary name.
    const std::string &ProcName = Proc->getName();

    // Skip default.
    if (ProcName == "NoItineraries")
      continue;

    // Sanity check for at least one instruction itinerary class.
    unsigned NItinClasses =
      Records.getAllDerivedDefinitions("InstrItinClass").size();
    if (NItinClasses == 0)
      return;

    // Get itinerary data list.
    std::vector<Record*> ItinDataList = Proc->getValueAsListOfDefs("IID");

    // Collect all instruction classes
    numInsnClasses += collectAllInsnClasses(ProcName, ProcItinList,
                          FUNameToBitsMap, ItinDataList, maxStages, OS);
  }

  // The type of a state in the nondeterministic automaton we're defining.
  using NfaStateTy = unsigned;

  // Given a resource state, return all resource states by applying
  // InsnClass.
  auto applyInsnClass = [&](ArrayRef<unsigned> InsnClass,
                            NfaStateTy State) -> std::deque<unsigned> {
    std::deque<unsigned> V(1, State);
    // Apply every stage in the class individually.
    for (unsigned Stage : InsnClass) {
      // Apply this stage to every existing member of V in turn.
      size_t Sz = V.size();
      for (unsigned I = 0; I < Sz; ++I) {
        unsigned S = V.front();
        V.pop_front();

        // For this stage, state combination, try all possible resources.
        for (unsigned J = 0; J < DFA_MAX_RESOURCES; ++J) {
          unsigned ResourceMask = 1U << J;
          if ((ResourceMask & Stage) == 0)
            // This resource isn't required by this stage.
            continue;
          unsigned Combo = ComboBitToBitsMap[ResourceMask];
          if (Combo && ((~S & Combo) != Combo))
            // This combo units bits are not available.
            continue;
          unsigned ResultingResourceState = S | ResourceMask | Combo;
          if (ResultingResourceState == S)
            continue;
          V.push_back(ResultingResourceState);
        }
      }
    }
    return V;
  };

  // Given a resource state, return a quick (conservative) guess as to whether
  // InsnClass can be applied. This is a filter for the more heavyweight
  // applyInsnClass.
  auto canApplyInsnClass = [](ArrayRef<unsigned> InsnClass,
                              NfaStateTy State) -> bool {
    for (unsigned Resources : InsnClass) {
      if ((State | Resources) == State)
        return false;
    }
    return true;
  };

  DfaEmitter Emitter;
  std::deque<NfaStateTy> Worklist(1, 0);
  std::set<NfaStateTy> SeenStates;
  SeenStates.insert(Worklist.front());
  while (!Worklist.empty()) {
    NfaStateTy State = Worklist.front();
    Worklist.pop_front();
    for (unsigned i = 0; i < allInsnClasses.size(); i++) {
      const std::vector<unsigned> &InsnClass = allInsnClasses[i];
      if (!canApplyInsnClass(InsnClass, State))
        continue;
      for (unsigned NewState : applyInsnClass(InsnClass, State)) {
        if (SeenStates.emplace(NewState).second)
          Worklist.emplace_back(NewState);
        Emitter.addTransition(State, NewState, getDFAInsnInput(InsnClass));
      }
    }
  }

  OS << "} // end namespace llvm\n\n";
  OS << "namespace {\n";
  std::string TargetAndDFAName = TargetName + DFAName;
  Emitter.emit(TargetAndDFAName, OS);
  OS << "} // end anonymous namespace\n\n";

  std::string SubTargetClassName = TargetName + "GenSubtargetInfo";
  OS << "namespace llvm {\n";
  OS << "DFAPacketizer *" << SubTargetClassName << "::"
     << "create" << DFAName
     << "DFAPacketizer(const InstrItineraryData *IID) const {\n"
     << "  static Automaton<uint64_t> A(ArrayRef<" << TargetAndDFAName
     << "Transition>(" << TargetAndDFAName << "Transitions), "
     << TargetAndDFAName << "TransitionInfo);\n"
     << "  return new DFAPacketizer(IID, A);\n"
     << "\n}\n\n";
}

namespace llvm {

void EmitDFAPacketizer(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Target DFA Packetizer Tables", OS);
  DFAPacketizerEmitter(RK).run(OS);
}

} // end namespace llvm
