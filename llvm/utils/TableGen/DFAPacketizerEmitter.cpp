//===- DFAPacketizerEmitter.cpp - Packetization DFA for a VLIW machine-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#include "CodeGenTarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <list>
#include <map>
#include <string>
using namespace llvm;

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
  DenseSet<unsigned> allInsnClasses;
  RecordKeeper &Records;

public:
  DFAPacketizerEmitter(RecordKeeper &R);

  //
  // collectAllInsnClasses: Populate allInsnClasses which is a set of units
  // used in each stage.
  //
  void collectAllInsnClasses(const std::string &Name,
                             Record *ItinData,
                             unsigned &NStages,
                             raw_ostream &OS);

  void run(raw_ostream &OS);
};
} // End anonymous namespace.

//
//
// State represents the usage of machine resources if the packet contains
// a set of instruction classes.
//
// Specifically, currentState is a set of bit-masks.
// The nth bit in a bit-mask indicates whether the nth resource is being used
// by this state. The set of bit-masks in a state represent the different
// possible outcomes of transitioning to this state.
// For example: consider a two resource architecture: resource L and resource M
// with three instruction classes: L, M, and L_or_M.
// From the initial state (currentState = 0x00), if we add instruction class
// L_or_M we will transition to a state with currentState = [0x01, 0x10]. This
// represents the possible resource states that can result from adding a L_or_M
// instruction
//
// Another way of thinking about this transition is we are mapping a NDFA with
// two states [0x01] and [0x10] into a DFA with a single state [0x01, 0x10].
//
// A State instance also contains a collection of transitions from that state:
// a map from inputs to new states.
//
namespace {
class State {
 public:
  static int currentStateNum;
  // stateNum is the only member used for equality/ordering, all other members
  // can be mutated even in const State objects.
  const int stateNum;
  mutable bool isInitial;
  mutable std::set<unsigned> stateInfo;
  typedef std::map<unsigned, const State *> TransitionMap;
  mutable TransitionMap Transitions;

  State();

  bool operator<(const State &s) const {
    return stateNum < s.stateNum;
  }

  //
  // canAddInsnClass - Returns true if an instruction of type InsnClass is a
  // valid transition from this state, i.e., can an instruction of type InsnClass
  // be added to the packet represented by this state.
  //
  // PossibleStates is the set of valid resource states that ensue from valid
  // transitions.
  //
  bool canAddInsnClass(unsigned InsnClass) const;
  //
  // AddInsnClass - Return all combinations of resource reservation
  // which are possible from this state (PossibleStates).
  //
  void AddInsnClass(unsigned InsnClass, std::set<unsigned> &PossibleStates) const;
  // 
  // addTransition - Add a transition from this state given the input InsnClass
  //
  void addTransition(unsigned InsnClass, const State *To) const;
  //
  // hasTransition - Returns true if there is a transition from this state
  // given the input InsnClass
  //
  bool hasTransition(unsigned InsnClass) const;
};
} // End anonymous namespace.

//
// class DFA: deterministic finite automaton for processor resource tracking.
//
namespace {
class DFA {
public:
  DFA();

  // Set of states. Need to keep this sorted to emit the transition table.
  typedef std::set<State> StateSet;
  StateSet states;

  State *currentState;

  //
  // Modify the DFA.
  //
  const State &newState();

  //
  // writeTable: Print out a table representing the DFA.
  //
  void writeTableAndAPI(raw_ostream &OS, const std::string &ClassName);
};
} // End anonymous namespace.


//
// Constructors and destructors for State and DFA
//
State::State() :
  stateNum(currentStateNum++), isInitial(false) {}

DFA::DFA(): currentState(nullptr) {}

// 
// addTransition - Add a transition from this state given the input InsnClass
//
void State::addTransition(unsigned InsnClass, const State *To) const {
  assert(!Transitions.count(InsnClass) &&
      "Cannot have multiple transitions for the same input");
  Transitions[InsnClass] = To;
}

//
// hasTransition - Returns true if there is a transition from this state
// given the input InsnClass
//
bool State::hasTransition(unsigned InsnClass) const {
  return Transitions.count(InsnClass) > 0;
}

//
// AddInsnClass - Return all combinations of resource reservation
// which are possible from this state (PossibleStates).
//
void State::AddInsnClass(unsigned InsnClass,
                            std::set<unsigned> &PossibleStates) const {
  //
  // Iterate over all resource states in currentState.
  //

  for (std::set<unsigned>::iterator SI = stateInfo.begin();
       SI != stateInfo.end(); ++SI) {
    unsigned thisState = *SI;

    //
    // Iterate over all possible resources used in InsnClass.
    // For ex: for InsnClass = 0x11, all resources = {0x01, 0x10}.
    //

    DenseSet<unsigned> VisitedResourceStates;
    for (unsigned int j = 0; j < sizeof(InsnClass) * 8; ++j) {
      if ((0x1 << j) & InsnClass) {
        //
        // For each possible resource used in InsnClass, generate the
        // resource state if that resource was used.
        //
        unsigned ResultingResourceState = thisState | (0x1 << j);
        //
        // Check if the resulting resource state can be accommodated in this
        // packet.
        // We compute ResultingResourceState OR thisState.
        // If the result of the OR is different than thisState, it implies
        // that there is at least one resource that can be used to schedule
        // InsnClass in the current packet.
        // Insert ResultingResourceState into PossibleStates only if we haven't
        // processed ResultingResourceState before.
        //
        if ((ResultingResourceState != thisState) &&
            (VisitedResourceStates.count(ResultingResourceState) == 0)) {
          VisitedResourceStates.insert(ResultingResourceState);
          PossibleStates.insert(ResultingResourceState);
        }
      }
    }
  }

}


//
// canAddInsnClass - Quickly verifies if an instruction of type InsnClass is a
// valid transition from this state i.e., can an instruction of type InsnClass
// be added to the packet represented by this state.
//
bool State::canAddInsnClass(unsigned InsnClass) const {
  for (std::set<unsigned>::const_iterator SI = stateInfo.begin();
       SI != stateInfo.end(); ++SI) {
    if (~*SI & InsnClass)
      return true;
  }
  return false;
}


const State &DFA::newState() {
  auto IterPair = states.insert(State());
  assert(IterPair.second && "State already exists");
  return *IterPair.first;
}


int State::currentStateNum = 0;

DFAPacketizerEmitter::DFAPacketizerEmitter(RecordKeeper &R):
  TargetName(CodeGenTarget(R).getName()),
  allInsnClasses(), Records(R) {}


//
// writeTableAndAPI - Print out a table representing the DFA and the
// associated API to create a DFA packetizer.
//
// Format:
// DFAStateInputTable[][2] = pairs of <Input, Transition> for all valid
//                           transitions.
// DFAStateEntryTable[i] = Index of the first entry in DFAStateInputTable for
//                         the ith state.
//
//
void DFA::writeTableAndAPI(raw_ostream &OS, const std::string &TargetName) {
  static const std::string SentinelEntry = "{-1, -1}";
  DFA::StateSet::iterator SI = states.begin();
  // This table provides a map to the beginning of the transitions for State s
  // in DFAStateInputTable.
  std::vector<int> StateEntry(states.size());

  OS << "namespace llvm {\n\n";
  OS << "const int " << TargetName << "DFAStateInputTable[][2] = {\n";

  // Tracks the total valid transitions encountered so far. It is used
  // to construct the StateEntry table.
  int ValidTransitions = 0;
  for (unsigned i = 0; i < states.size(); ++i, ++SI) {
    assert ((SI->stateNum == (int) i) && "Mismatch in state numbers");
    StateEntry[i] = ValidTransitions;
    for (State::TransitionMap::iterator
        II = SI->Transitions.begin(), IE = SI->Transitions.end();
        II != IE; ++II) {
      OS << "{" << II->first << ", "
         << II->second->stateNum
         << "},    ";
    }
    ValidTransitions += SI->Transitions.size();

    // If there are no valid transitions from this stage, we need a sentinel
    // transition.
    if (ValidTransitions == StateEntry[i]) {
      OS << SentinelEntry << ",";
      ++ValidTransitions;
    }

    OS << "\n";
  }

  // Print out a sentinel entry at the end of the StateInputTable. This is
  // needed to iterate over StateInputTable in DFAPacketizer::ReadTable()
  OS << SentinelEntry << "\n";
  
  OS << "};\n\n";
  OS << "const unsigned int " << TargetName << "DFAStateEntryTable[] = {\n";

  // Multiply i by 2 since each entry in DFAStateInputTable is a set of
  // two numbers.
  for (unsigned i = 0; i < states.size(); ++i)
    OS << StateEntry[i] << ", ";

  // Print out the index to the sentinel entry in StateInputTable
  OS << ValidTransitions << ", ";

  OS << "\n};\n";
  OS << "} // namespace\n";


  //
  // Emit DFA Packetizer tables if the target is a VLIW machine.
  //
  std::string SubTargetClassName = TargetName + "GenSubtargetInfo";
  OS << "\n" << "#include \"llvm/CodeGen/DFAPacketizer.h\"\n";
  OS << "namespace llvm {\n";
  OS << "DFAPacketizer *" << SubTargetClassName << "::"
     << "createDFAPacketizer(const InstrItineraryData *IID) const {\n"
     << "   return new DFAPacketizer(IID, " << TargetName
     << "DFAStateInputTable, " << TargetName << "DFAStateEntryTable);\n}\n\n";
  OS << "} // End llvm namespace \n";
}


//
// collectAllInsnClasses - Populate allInsnClasses which is a set of units
// used in each stage.
//
void DFAPacketizerEmitter::collectAllInsnClasses(const std::string &Name,
                                  Record *ItinData,
                                  unsigned &NStages,
                                  raw_ostream &OS) {
  // Collect processor itineraries.
  std::vector<Record*> ProcItinList =
    Records.getAllDerivedDefinitions("ProcessorItineraries");

  // If just no itinerary then don't bother.
  if (ProcItinList.size() < 2)
    return;
  std::map<std::string, unsigned> NameToBitsMap;

  // Parse functional units for all the itineraries.
  for (unsigned i = 0, N = ProcItinList.size(); i < N; ++i) {
    Record *Proc = ProcItinList[i];
    std::vector<Record*> FUs = Proc->getValueAsListOfDefs("FU");

    // Convert macros to bits for each stage.
    for (unsigned i = 0, N = FUs.size(); i < N; ++i)
      NameToBitsMap[FUs[i]->getName()] = (unsigned) (1U << i);
  }

  const std::vector<Record*> &StageList =
    ItinData->getValueAsListOfDefs("Stages");

  // The number of stages.
  NStages = StageList.size();

  // For each unit.
  unsigned UnitBitValue = 0;

  // Compute the bitwise or of each unit used in this stage.
  for (unsigned i = 0; i < NStages; ++i) {
    const Record *Stage = StageList[i];

    // Get unit list.
    const std::vector<Record*> &UnitList =
      Stage->getValueAsListOfDefs("Units");

    for (unsigned j = 0, M = UnitList.size(); j < M; ++j) {
      // Conduct bitwise or.
      std::string UnitName = UnitList[j]->getName();
      assert(NameToBitsMap.count(UnitName));
      UnitBitValue |= NameToBitsMap[UnitName];
    }

    if (UnitBitValue != 0)
      allInsnClasses.insert(UnitBitValue);
  }
}


//
// Run the worklist algorithm to generate the DFA.
//
void DFAPacketizerEmitter::run(raw_ostream &OS) {

  // Collect processor iteraries.
  std::vector<Record*> ProcItinList =
    Records.getAllDerivedDefinitions("ProcessorItineraries");

  //
  // Collect the instruction classes.
  //
  for (unsigned i = 0, N = ProcItinList.size(); i < N; i++) {
    Record *Proc = ProcItinList[i];

    // Get processor itinerary name.
    const std::string &Name = Proc->getName();

    // Skip default.
    if (Name == "NoItineraries")
      continue;

    // Sanity check for at least one instruction itinerary class.
    unsigned NItinClasses =
      Records.getAllDerivedDefinitions("InstrItinClass").size();
    if (NItinClasses == 0)
      return;

    // Get itinerary data list.
    std::vector<Record*> ItinDataList = Proc->getValueAsListOfDefs("IID");

    // Collect instruction classes for all itinerary data.
    for (unsigned j = 0, M = ItinDataList.size(); j < M; j++) {
      Record *ItinData = ItinDataList[j];
      unsigned NStages;
      collectAllInsnClasses(Name, ItinData, NStages, OS);
    }
  }


  //
  // Run a worklist algorithm to generate the DFA.
  //
  DFA D;
  const State *Initial = &D.newState();
  Initial->isInitial = true;
  Initial->stateInfo.insert(0x0);
  SmallVector<const State*, 32> WorkList;
  std::map<std::set<unsigned>, const State*> Visited;

  WorkList.push_back(Initial);

  //
  // Worklist algorithm to create a DFA for processor resource tracking.
  // C = {set of InsnClasses}
  // Begin with initial node in worklist. Initial node does not have
  // any consumed resources,
  //     ResourceState = 0x0
  // Visited = {}
  // While worklist != empty
  //    S = first element of worklist
  //    For every instruction class C
  //      if we can accommodate C in S:
  //          S' = state with resource states = {S Union C}
  //          Add a new transition: S x C -> S'
  //          If S' is not in Visited:
  //             Add S' to worklist
  //             Add S' to Visited
  //
  while (!WorkList.empty()) {
    const State *current = WorkList.pop_back_val();
    for (DenseSet<unsigned>::iterator CI = allInsnClasses.begin(),
           CE = allInsnClasses.end(); CI != CE; ++CI) {
      unsigned InsnClass = *CI;

      std::set<unsigned> NewStateResources;
      //
      // If we haven't already created a transition for this input
      // and the state can accommodate this InsnClass, create a transition.
      //
      if (!current->hasTransition(InsnClass) &&
          current->canAddInsnClass(InsnClass)) {
        const State *NewState;
        current->AddInsnClass(InsnClass, NewStateResources);
        assert(!NewStateResources.empty() && "New states must be generated");

        //
        // If we have seen this state before, then do not create a new state.
        //
        //
        auto VI = Visited.find(NewStateResources);
        if (VI != Visited.end())
          NewState = VI->second;
        else {
          NewState = &D.newState();
          NewState->stateInfo = NewStateResources;
          Visited[NewStateResources] = NewState;
          WorkList.push_back(NewState);
        }
        
        current->addTransition(InsnClass, NewState);
      }
    }
  }

  // Print out the table.
  D.writeTableAndAPI(OS, TargetName);
}

namespace llvm {

void EmitDFAPacketizer(RecordKeeper &RK, raw_ostream &OS) {
  emitSourceFileHeader("Target DFA Packetizer Tables", OS);
  DFAPacketizerEmitter(RK).run(OS);
}

} // End llvm namespace
