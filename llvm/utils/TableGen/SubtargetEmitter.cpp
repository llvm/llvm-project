//===- SubtargetEmitter.cpp - Generate subtarget enumerations -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits subtarget enumerations.
//
//===----------------------------------------------------------------------===//

#include "CodeGenTarget.h"
#include "CodeGenSchedule.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <map>
#include <string>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "subtarget-emitter"

namespace {

class SubtargetEmitter {
  // Each processor has a SchedClassDesc table with an entry for each SchedClass.
  // The SchedClassDesc table indexes into a global write resource table, write
  // latency table, and read advance table.
  struct SchedClassTables {
    std::vector<std::vector<MCSchedClassDesc>> ProcSchedClasses;
    std::vector<MCWriteProcResEntry> WriteProcResources;
    std::vector<MCWriteLatencyEntry> WriteLatencies;
    std::vector<std::string> WriterNames;
    std::vector<MCReadAdvanceEntry> ReadAdvanceEntries;

    // Reserve an invalid entry at index 0
    SchedClassTables() {
      ProcSchedClasses.resize(1);
      WriteProcResources.resize(1);
      WriteLatencies.resize(1);
      WriterNames.push_back("InvalidWrite");
      ReadAdvanceEntries.resize(1);
    }
  };

  struct LessWriteProcResources {
    bool operator()(const MCWriteProcResEntry &LHS,
                    const MCWriteProcResEntry &RHS) {
      return LHS.ProcResourceIdx < RHS.ProcResourceIdx;
    }
  };

  RecordKeeper &Records;
  CodeGenSchedModels &SchedModels;
  std::string Target;

  void Enumeration(raw_ostream &OS);
  unsigned FeatureKeyValues(raw_ostream &OS);
  unsigned CPUKeyValues(raw_ostream &OS);
  void FormItineraryStageString(const std::string &Names,
                                Record *ItinData, std::string &ItinString,
                                unsigned &NStages);
  void FormItineraryOperandCycleString(Record *ItinData, std::string &ItinString,
                                       unsigned &NOperandCycles);
  void FormItineraryBypassString(const std::string &Names,
                                 Record *ItinData,
                                 std::string &ItinString, unsigned NOperandCycles);
  void EmitStageAndOperandCycleData(raw_ostream &OS,
                                    std::vector<std::vector<InstrItinerary>>
                                      &ProcItinLists);
  void EmitItineraries(raw_ostream &OS,
                       std::vector<std::vector<InstrItinerary>>
                         &ProcItinLists);
  void EmitProcessorProp(raw_ostream &OS, const Record *R, StringRef Name,
                         char Separator);
  void EmitProcessorResources(const CodeGenProcModel &ProcModel,
                              raw_ostream &OS);
  Record *FindWriteResources(const CodeGenSchedRW &SchedWrite,
                             const CodeGenProcModel &ProcModel);
  Record *FindReadAdvance(const CodeGenSchedRW &SchedRead,
                          const CodeGenProcModel &ProcModel);
  void ExpandProcResources(RecVec &PRVec, std::vector<int64_t> &Cycles,
                           const CodeGenProcModel &ProcModel);
  void GenSchedClassTables(const CodeGenProcModel &ProcModel,
                           SchedClassTables &SchedTables);
  void EmitSchedClassTables(SchedClassTables &SchedTables, raw_ostream &OS);
  void EmitProcessorModels(raw_ostream &OS);
  void EmitProcessorLookup(raw_ostream &OS);
  void EmitSchedModelHelpers(const std::string &ClassName, raw_ostream &OS);
  void EmitSchedModel(raw_ostream &OS);
  void ParseFeaturesFunction(raw_ostream &OS, unsigned NumFeatures,
                             unsigned NumProcs);

public:
  SubtargetEmitter(RecordKeeper &R, CodeGenTarget &TGT):
    Records(R), SchedModels(TGT.getSchedModels()), Target(TGT.getName()) {}

  void run(raw_ostream &o);
};

} // end anonymous namespace

//
// Enumeration - Emit the specified class as an enumeration.
//
void SubtargetEmitter::Enumeration(raw_ostream &OS) {
  // Get all records of class and sort
  std::vector<Record*> DefList =
    Records.getAllDerivedDefinitions("SubtargetFeature");
  std::sort(DefList.begin(), DefList.end(), LessRecord());

  unsigned N = DefList.size();
  if (N == 0)
    return;
  if (N > MAX_SUBTARGET_FEATURES)
    PrintFatalError("Too many subtarget features! Bump MAX_SUBTARGET_FEATURES.");

  OS << "namespace " << Target << " {\n";

  // Open enumeration.
  OS << "enum {\n";

  // For each record
  for (unsigned i = 0; i < N;) {
    // Next record
    Record *Def = DefList[i];

    // Get and emit name
    OS << "  " << Def->getName() << " = " << i;
    if (++i < N) OS << ",";

    OS << "\n";
  }

  // Close enumeration and namespace
  OS << "};\n";
  OS << "} // end namespace " << Target << "\n";
}

//
// FeatureKeyValues - Emit data of all the subtarget features.  Used by the
// command line.
//
unsigned SubtargetEmitter::FeatureKeyValues(raw_ostream &OS) {
  // Gather and sort all the features
  std::vector<Record*> FeatureList =
                           Records.getAllDerivedDefinitions("SubtargetFeature");

  if (FeatureList.empty())
    return 0;

  std::sort(FeatureList.begin(), FeatureList.end(), LessRecordFieldName());

  // Begin feature table
  OS << "// Sorted (by key) array of values for CPU features.\n"
     << "extern const llvm::SubtargetFeatureKV " << Target
     << "FeatureKV[] = {\n";

  // For each feature
  unsigned NumFeatures = 0;
  for (unsigned i = 0, N = FeatureList.size(); i < N; ++i) {
    // Next feature
    Record *Feature = FeatureList[i];

    const std::string &Name = Feature->getName();
    const std::string &CommandLineName = Feature->getValueAsString("Name");
    const std::string &Desc = Feature->getValueAsString("Desc");

    if (CommandLineName.empty()) continue;

    // Emit as { "feature", "description", { featureEnum }, { i1 , i2 , ... , in } }
    OS << "  { "
       << "\"" << CommandLineName << "\", "
       << "\"" << Desc << "\", "
       << "{ " << Target << "::" << Name << " }, ";

    const std::vector<Record*> &ImpliesList =
      Feature->getValueAsListOfDefs("Implies");

    OS << "{";
    for (unsigned j = 0, M = ImpliesList.size(); j < M;) {
      OS << " " << Target << "::" << ImpliesList[j]->getName();
      if (++j < M) OS << ",";
    }
    OS << " }";

    OS << " }";
    ++NumFeatures;

    // Depending on 'if more in the list' emit comma
    if ((i + 1) < N) OS << ",";

    OS << "\n";
  }

  // End feature table
  OS << "};\n";

  return NumFeatures;
}

//
// CPUKeyValues - Emit data of all the subtarget processors.  Used by command
// line.
//
unsigned SubtargetEmitter::CPUKeyValues(raw_ostream &OS) {
  // Gather and sort processor information
  std::vector<Record*> ProcessorList =
                          Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcessorList.begin(), ProcessorList.end(), LessRecordFieldName());

  // Begin processor table
  OS << "// Sorted (by key) array of values for CPU subtype.\n"
     << "extern const llvm::SubtargetFeatureKV " << Target
     << "SubTypeKV[] = {\n";

  // For each processor
  for (unsigned i = 0, N = ProcessorList.size(); i < N;) {
    // Next processor
    Record *Processor = ProcessorList[i];

    const std::string &Name = Processor->getValueAsString("Name");
    const std::vector<Record*> &FeatureList =
      Processor->getValueAsListOfDefs("Features");

    // Emit as { "cpu", "description", { f1 , f2 , ... fn } },
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"Select the " << Name << " processor\", ";

    OS << "{";
    for (unsigned j = 0, M = FeatureList.size(); j < M;) {
      OS << " " << Target << "::" << FeatureList[j]->getName();
      if (++j < M) OS << ",";
    }
    OS << " }";

    // The { } is for the "implies" section of this data structure.
    OS << ", { } }";

    // Depending on 'if more in the list' emit comma
    if (++i < N) OS << ",";

    OS << "\n";
  }

  // End processor table
  OS << "};\n";

  return ProcessorList.size();
}

//
// FormItineraryStageString - Compose a string containing the stage
// data initialization for the specified itinerary.  N is the number
// of stages.
//
void SubtargetEmitter::FormItineraryStageString(const std::string &Name,
                                                Record *ItinData,
                                                std::string &ItinString,
                                                unsigned &NStages) {
  // Get states list
  const std::vector<Record*> &StageList =
    ItinData->getValueAsListOfDefs("Stages");

  // For each stage
  unsigned N = NStages = StageList.size();
  for (unsigned i = 0; i < N;) {
    // Next stage
    const Record *Stage = StageList[i];

    // Form string as ,{ cycles, u1 | u2 | ... | un, timeinc, kind }
    int Cycles = Stage->getValueAsInt("Cycles");
    ItinString += "  { " + itostr(Cycles) + ", ";

    // Get unit list
    const std::vector<Record*> &UnitList = Stage->getValueAsListOfDefs("Units");

    // For each unit
    for (unsigned j = 0, M = UnitList.size(); j < M;) {
      // Add name and bitwise or
      ItinString += Name + "FU::" + UnitList[j]->getName().str();
      if (++j < M) ItinString += " | ";
    }

    int TimeInc = Stage->getValueAsInt("TimeInc");
    ItinString += ", " + itostr(TimeInc);

    int Kind = Stage->getValueAsInt("Kind");
    ItinString += ", (llvm::InstrStage::ReservationKinds)" + itostr(Kind);

    // Close off stage
    ItinString += " }";
    if (++i < N) ItinString += ", ";
  }
}

//
// FormItineraryOperandCycleString - Compose a string containing the
// operand cycle initialization for the specified itinerary.  N is the
// number of operands that has cycles specified.
//
void SubtargetEmitter::FormItineraryOperandCycleString(Record *ItinData,
                         std::string &ItinString, unsigned &NOperandCycles) {
  // Get operand cycle list
  const std::vector<int64_t> &OperandCycleList =
    ItinData->getValueAsListOfInts("OperandCycles");

  // For each operand cycle
  unsigned N = NOperandCycles = OperandCycleList.size();
  for (unsigned i = 0; i < N;) {
    // Next operand cycle
    const int OCycle = OperandCycleList[i];

    ItinString += "  " + itostr(OCycle);
    if (++i < N) ItinString += ", ";
  }
}

void SubtargetEmitter::FormItineraryBypassString(const std::string &Name,
                                                 Record *ItinData,
                                                 std::string &ItinString,
                                                 unsigned NOperandCycles) {
  const std::vector<Record*> &BypassList =
    ItinData->getValueAsListOfDefs("Bypasses");
  unsigned N = BypassList.size();
  unsigned i = 0;
  for (; i < N;) {
    ItinString += Name + "Bypass::" + BypassList[i]->getName().str();
    if (++i < NOperandCycles) ItinString += ", ";
  }
  for (; i < NOperandCycles;) {
    ItinString += " 0";
    if (++i < NOperandCycles) ItinString += ", ";
  }
}

//
// EmitStageAndOperandCycleData - Generate unique itinerary stages and operand
// cycle tables. Create a list of InstrItinerary objects (ProcItinLists) indexed
// by CodeGenSchedClass::Index.
//
void SubtargetEmitter::
EmitStageAndOperandCycleData(raw_ostream &OS,
                             std::vector<std::vector<InstrItinerary>>
                               &ProcItinLists) {
  // Multiple processor models may share an itinerary record. Emit it once.
  SmallPtrSet<Record*, 8> ItinsDefSet;

  // Emit functional units for all the itineraries.
  for (const CodeGenProcModel &ProcModel : SchedModels.procModels()) {

    if (!ItinsDefSet.insert(ProcModel.ItinsDef).second)
      continue;

    std::vector<Record*> FUs = ProcModel.ItinsDef->getValueAsListOfDefs("FU");
    if (FUs.empty())
      continue;

    const std::string &Name = ProcModel.ItinsDef->getName();
    OS << "\n// Functional units for \"" << Name << "\"\n"
       << "namespace " << Name << "FU {\n";

    for (unsigned j = 0, FUN = FUs.size(); j < FUN; ++j)
      OS << "  const unsigned " << FUs[j]->getName()
         << " = 1 << " << j << ";\n";

    OS << "} // end namespace " << Name << "FU\n";

    std::vector<Record*> BPs = ProcModel.ItinsDef->getValueAsListOfDefs("BP");
    if (!BPs.empty()) {
      OS << "\n// Pipeline forwarding pathes for itineraries \"" << Name
         << "\"\n" << "namespace " << Name << "Bypass {\n";

      OS << "  const unsigned NoBypass = 0;\n";
      for (unsigned j = 0, BPN = BPs.size(); j < BPN; ++j)
        OS << "  const unsigned " << BPs[j]->getName()
           << " = 1 << " << j << ";\n";

      OS << "} // end namespace " << Name << "Bypass\n";
    }
  }

  // Begin stages table
  std::string StageTable = "\nextern const llvm::InstrStage " + Target +
                           "Stages[] = {\n";
  StageTable += "  { 0, 0, 0, llvm::InstrStage::Required }, // No itinerary\n";

  // Begin operand cycle table
  std::string OperandCycleTable = "extern const unsigned " + Target +
    "OperandCycles[] = {\n";
  OperandCycleTable += "  0, // No itinerary\n";

  // Begin pipeline bypass table
  std::string BypassTable = "extern const unsigned " + Target +
    "ForwardingPaths[] = {\n";
  BypassTable += " 0, // No itinerary\n";

  // For each Itinerary across all processors, add a unique entry to the stages,
  // operand cycles, and pipeline bypass tables. Then add the new Itinerary
  // object with computed offsets to the ProcItinLists result.
  unsigned StageCount = 1, OperandCycleCount = 1;
  std::map<std::string, unsigned> ItinStageMap, ItinOperandMap;
  for (const CodeGenProcModel &ProcModel : SchedModels.procModels()) {
    // Add process itinerary to the list.
    ProcItinLists.resize(ProcItinLists.size()+1);

    // If this processor defines no itineraries, then leave the itinerary list
    // empty.
    std::vector<InstrItinerary> &ItinList = ProcItinLists.back();
    if (!ProcModel.hasItineraries())
      continue;

    const std::string &Name = ProcModel.ItinsDef->getName();

    ItinList.resize(SchedModels.numInstrSchedClasses());
    assert(ProcModel.ItinDefList.size() == ItinList.size() && "bad Itins");

    for (unsigned SchedClassIdx = 0, SchedClassEnd = ItinList.size();
         SchedClassIdx < SchedClassEnd; ++SchedClassIdx) {

      // Next itinerary data
      Record *ItinData = ProcModel.ItinDefList[SchedClassIdx];

      // Get string and stage count
      std::string ItinStageString;
      unsigned NStages = 0;
      if (ItinData)
        FormItineraryStageString(Name, ItinData, ItinStageString, NStages);

      // Get string and operand cycle count
      std::string ItinOperandCycleString;
      unsigned NOperandCycles = 0;
      std::string ItinBypassString;
      if (ItinData) {
        FormItineraryOperandCycleString(ItinData, ItinOperandCycleString,
                                        NOperandCycles);

        FormItineraryBypassString(Name, ItinData, ItinBypassString,
                                  NOperandCycles);
      }

      // Check to see if stage already exists and create if it doesn't
      unsigned FindStage = 0;
      if (NStages > 0) {
        FindStage = ItinStageMap[ItinStageString];
        if (FindStage == 0) {
          // Emit as { cycles, u1 | u2 | ... | un, timeinc }, // indices
          StageTable += ItinStageString + ", // " + itostr(StageCount);
          if (NStages > 1)
            StageTable += "-" + itostr(StageCount + NStages - 1);
          StageTable += "\n";
          // Record Itin class number.
          ItinStageMap[ItinStageString] = FindStage = StageCount;
          StageCount += NStages;
        }
      }

      // Check to see if operand cycle already exists and create if it doesn't
      unsigned FindOperandCycle = 0;
      if (NOperandCycles > 0) {
        std::string ItinOperandString = ItinOperandCycleString+ItinBypassString;
        FindOperandCycle = ItinOperandMap[ItinOperandString];
        if (FindOperandCycle == 0) {
          // Emit as  cycle, // index
          OperandCycleTable += ItinOperandCycleString + ", // ";
          std::string OperandIdxComment = itostr(OperandCycleCount);
          if (NOperandCycles > 1)
            OperandIdxComment += "-"
              + itostr(OperandCycleCount + NOperandCycles - 1);
          OperandCycleTable += OperandIdxComment + "\n";
          // Record Itin class number.
          ItinOperandMap[ItinOperandCycleString] =
            FindOperandCycle = OperandCycleCount;
          // Emit as bypass, // index
          BypassTable += ItinBypassString + ", // " + OperandIdxComment + "\n";
          OperandCycleCount += NOperandCycles;
        }
      }

      // Set up itinerary as location and location + stage count
      int NumUOps = ItinData ? ItinData->getValueAsInt("NumMicroOps") : 0;
      InstrItinerary Intinerary = { NumUOps, FindStage, FindStage + NStages,
                                    FindOperandCycle,
                                    FindOperandCycle + NOperandCycles };

      // Inject - empty slots will be 0, 0
      ItinList[SchedClassIdx] = Intinerary;
    }
  }

  // Closing stage
  StageTable += "  { 0, 0, 0, llvm::InstrStage::Required } // End stages\n";
  StageTable += "};\n";

  // Closing operand cycles
  OperandCycleTable += "  0 // End operand cycles\n";
  OperandCycleTable += "};\n";

  BypassTable += " 0 // End bypass tables\n";
  BypassTable += "};\n";

  // Emit tables.
  OS << StageTable;
  OS << OperandCycleTable;
  OS << BypassTable;
}

//
// EmitProcessorData - Generate data for processor itineraries that were
// computed during EmitStageAndOperandCycleData(). ProcItinLists lists all
// Itineraries for each processor. The Itinerary lists are indexed on
// CodeGenSchedClass::Index.
//
void SubtargetEmitter::
EmitItineraries(raw_ostream &OS,
                std::vector<std::vector<InstrItinerary>> &ProcItinLists) {
  // Multiple processor models may share an itinerary record. Emit it once.
  SmallPtrSet<Record*, 8> ItinsDefSet;

  // For each processor's machine model
  std::vector<std::vector<InstrItinerary>>::iterator
      ProcItinListsIter = ProcItinLists.begin();
  for (CodeGenSchedModels::ProcIter PI = SchedModels.procModelBegin(),
         PE = SchedModels.procModelEnd(); PI != PE; ++PI, ++ProcItinListsIter) {

    Record *ItinsDef = PI->ItinsDef;
    if (!ItinsDefSet.insert(ItinsDef).second)
      continue;

    // Get processor itinerary name
    const std::string &Name = ItinsDef->getName();

    // Get the itinerary list for the processor.
    assert(ProcItinListsIter != ProcItinLists.end() && "bad iterator");
    std::vector<InstrItinerary> &ItinList = *ProcItinListsIter;

    // Empty itineraries aren't referenced anywhere in the tablegen output
    // so don't emit them.
    if (ItinList.empty())
      continue;

    OS << "\n";
    OS << "static const llvm::InstrItinerary ";

    // Begin processor itinerary table
    OS << Name << "[] = {\n";

    // For each itinerary class in CodeGenSchedClass::Index order.
    for (unsigned j = 0, M = ItinList.size(); j < M; ++j) {
      InstrItinerary &Intinerary = ItinList[j];

      // Emit Itinerary in the form of
      // { firstStage, lastStage, firstCycle, lastCycle } // index
      OS << "  { " <<
        Intinerary.NumMicroOps << ", " <<
        Intinerary.FirstStage << ", " <<
        Intinerary.LastStage << ", " <<
        Intinerary.FirstOperandCycle << ", " <<
        Intinerary.LastOperandCycle << " }" <<
        ", // " << j << " " << SchedModels.getSchedClass(j).Name << "\n";
    }
    // End processor itinerary table
    OS << "  { 0, ~0U, ~0U, ~0U, ~0U } // end marker\n";
    OS << "};\n";
  }
}

// Emit either the value defined in the TableGen Record, or the default
// value defined in the C++ header. The Record is null if the processor does not
// define a model.
void SubtargetEmitter::EmitProcessorProp(raw_ostream &OS, const Record *R,
                                         StringRef Name, char Separator) {
  OS << "  ";
  int V = R ? R->getValueAsInt(Name) : -1;
  if (V >= 0)
    OS << V << Separator << " // " << Name;
  else
    OS << "MCSchedModel::Default" << Name << Separator;
  OS << '\n';
}

void SubtargetEmitter::EmitProcessorResources(const CodeGenProcModel &ProcModel,
                                              raw_ostream &OS) {
  char Sep = ProcModel.ProcResourceDefs.empty() ? ' ' : ',';

  OS << "\n// {Name, NumUnits, SuperIdx, IsBuffered}\n";
  OS << "static const llvm::MCProcResourceDesc "
     << ProcModel.ModelName << "ProcResources" << "[] = {\n"
     << "  {DBGFIELD(\"InvalidUnit\")     0, 0, 0}" << Sep << "\n";

  for (unsigned i = 0, e = ProcModel.ProcResourceDefs.size(); i < e; ++i) {
    Record *PRDef = ProcModel.ProcResourceDefs[i];

    Record *SuperDef = nullptr;
    unsigned SuperIdx = 0;
    unsigned NumUnits = 0;
    int BufferSize = PRDef->getValueAsInt("BufferSize");
    if (PRDef->isSubClassOf("ProcResGroup")) {
      RecVec ResUnits = PRDef->getValueAsListOfDefs("Resources");
      for (Record *RU : ResUnits) {
        NumUnits += RU->getValueAsInt("NumUnits");
      }
    }
    else {
      // Find the SuperIdx
      if (PRDef->getValueInit("Super")->isComplete()) {
        SuperDef = SchedModels.findProcResUnits(
          PRDef->getValueAsDef("Super"), ProcModel);
        SuperIdx = ProcModel.getProcResourceIdx(SuperDef);
      }
      NumUnits = PRDef->getValueAsInt("NumUnits");
    }
    // Emit the ProcResourceDesc
    if (i+1 == e)
      Sep = ' ';
    OS << "  {DBGFIELD(\"" << PRDef->getName() << "\") ";
    if (PRDef->getName().size() < 15)
      OS.indent(15 - PRDef->getName().size());
    OS << NumUnits << ", " << SuperIdx << ", "
       << BufferSize << "}" << Sep << " // #" << i+1;
    if (SuperDef)
      OS << ", Super=" << SuperDef->getName();
    OS << "\n";
  }
  OS << "};\n";
}

// Find the WriteRes Record that defines processor resources for this
// SchedWrite.
Record *SubtargetEmitter::FindWriteResources(
  const CodeGenSchedRW &SchedWrite, const CodeGenProcModel &ProcModel) {

  // Check if the SchedWrite is already subtarget-specific and directly
  // specifies a set of processor resources.
  if (SchedWrite.TheDef->isSubClassOf("SchedWriteRes"))
    return SchedWrite.TheDef;

  Record *AliasDef = nullptr;
  for (Record *A : SchedWrite.Aliases) {
    const CodeGenSchedRW &AliasRW =
      SchedModels.getSchedRW(A->getValueAsDef("AliasRW"));
    if (AliasRW.TheDef->getValueInit("SchedModel")->isComplete()) {
      Record *ModelDef = AliasRW.TheDef->getValueAsDef("SchedModel");
      if (&SchedModels.getProcModel(ModelDef) != &ProcModel)
        continue;
    }
    if (AliasDef)
      PrintFatalError(AliasRW.TheDef->getLoc(), "Multiple aliases "
                    "defined for processor " + ProcModel.ModelName +
                    " Ensure only one SchedAlias exists per RW.");
    AliasDef = AliasRW.TheDef;
  }
  if (AliasDef && AliasDef->isSubClassOf("SchedWriteRes"))
    return AliasDef;

  // Check this processor's list of write resources.
  Record *ResDef = nullptr;
  for (Record *WR : ProcModel.WriteResDefs) {
    if (!WR->isSubClassOf("WriteRes"))
      continue;
    if (AliasDef == WR->getValueAsDef("WriteType")
        || SchedWrite.TheDef == WR->getValueAsDef("WriteType")) {
      if (ResDef) {
        PrintFatalError(WR->getLoc(), "Resources are defined for both "
                      "SchedWrite and its alias on processor " +
                      ProcModel.ModelName);
      }
      ResDef = WR;
    }
  }
  // TODO: If ProcModel has a base model (previous generation processor),
  // then call FindWriteResources recursively with that model here.
  if (!ResDef) {
    PrintFatalError(ProcModel.ModelDef->getLoc(),
                  std::string("Processor does not define resources for ")
                  + SchedWrite.TheDef->getName());
  }
  return ResDef;
}

/// Find the ReadAdvance record for the given SchedRead on this processor or
/// return NULL.
Record *SubtargetEmitter::FindReadAdvance(const CodeGenSchedRW &SchedRead,
                                          const CodeGenProcModel &ProcModel) {
  // Check for SchedReads that directly specify a ReadAdvance.
  if (SchedRead.TheDef->isSubClassOf("SchedReadAdvance"))
    return SchedRead.TheDef;

  // Check this processor's list of aliases for SchedRead.
  Record *AliasDef = nullptr;
  for (Record *A : SchedRead.Aliases) {
    const CodeGenSchedRW &AliasRW =
      SchedModels.getSchedRW(A->getValueAsDef("AliasRW"));
    if (AliasRW.TheDef->getValueInit("SchedModel")->isComplete()) {
      Record *ModelDef = AliasRW.TheDef->getValueAsDef("SchedModel");
      if (&SchedModels.getProcModel(ModelDef) != &ProcModel)
        continue;
    }
    if (AliasDef)
      PrintFatalError(AliasRW.TheDef->getLoc(), "Multiple aliases "
                    "defined for processor " + ProcModel.ModelName +
                    " Ensure only one SchedAlias exists per RW.");
    AliasDef = AliasRW.TheDef;
  }
  if (AliasDef && AliasDef->isSubClassOf("SchedReadAdvance"))
    return AliasDef;

  // Check this processor's ReadAdvanceList.
  Record *ResDef = nullptr;
  for (Record *RA : ProcModel.ReadAdvanceDefs) {
    if (!RA->isSubClassOf("ReadAdvance"))
      continue;
    if (AliasDef == RA->getValueAsDef("ReadType")
        || SchedRead.TheDef == RA->getValueAsDef("ReadType")) {
      if (ResDef) {
        PrintFatalError(RA->getLoc(), "Resources are defined for both "
                      "SchedRead and its alias on processor " +
                      ProcModel.ModelName);
      }
      ResDef = RA;
    }
  }
  // TODO: If ProcModel has a base model (previous generation processor),
  // then call FindReadAdvance recursively with that model here.
  if (!ResDef && SchedRead.TheDef->getName() != "ReadDefault") {
    PrintFatalError(ProcModel.ModelDef->getLoc(),
                  std::string("Processor does not define resources for ")
                  + SchedRead.TheDef->getName());
  }
  return ResDef;
}

// Expand an explicit list of processor resources into a full list of implied
// resource groups and super resources that cover them.
void SubtargetEmitter::ExpandProcResources(RecVec &PRVec,
                                           std::vector<int64_t> &Cycles,
                                           const CodeGenProcModel &PM) {
  // Default to 1 resource cycle.
  Cycles.resize(PRVec.size(), 1);
  for (unsigned i = 0, e = PRVec.size(); i != e; ++i) {
    Record *PRDef = PRVec[i];
    RecVec SubResources;
    if (PRDef->isSubClassOf("ProcResGroup"))
      SubResources = PRDef->getValueAsListOfDefs("Resources");
    else {
      SubResources.push_back(PRDef);
      PRDef = SchedModels.findProcResUnits(PRVec[i], PM);
      for (Record *SubDef = PRDef;
           SubDef->getValueInit("Super")->isComplete();) {
        if (SubDef->isSubClassOf("ProcResGroup")) {
          // Disallow this for simplicitly.
          PrintFatalError(SubDef->getLoc(), "Processor resource group "
                          " cannot be a super resources.");
        }
        Record *SuperDef =
          SchedModels.findProcResUnits(SubDef->getValueAsDef("Super"), PM);
        PRVec.push_back(SuperDef);
        Cycles.push_back(Cycles[i]);
        SubDef = SuperDef;
      }
    }
    for (Record *PR : PM.ProcResourceDefs) {
      if (PR == PRDef || !PR->isSubClassOf("ProcResGroup"))
        continue;
      RecVec SuperResources = PR->getValueAsListOfDefs("Resources");
      RecIter SubI = SubResources.begin(), SubE = SubResources.end();
      for( ; SubI != SubE; ++SubI) {
        if (!is_contained(SuperResources, *SubI)) {
          break;
        }
      }
      if (SubI == SubE) {
        PRVec.push_back(PR);
        Cycles.push_back(Cycles[i]);
      }
    }
  }
}

// Generate the SchedClass table for this processor and update global
// tables. Must be called for each processor in order.
void SubtargetEmitter::GenSchedClassTables(const CodeGenProcModel &ProcModel,
                                           SchedClassTables &SchedTables) {
  SchedTables.ProcSchedClasses.resize(SchedTables.ProcSchedClasses.size() + 1);
  if (!ProcModel.hasInstrSchedModel())
    return;

  std::vector<MCSchedClassDesc> &SCTab = SchedTables.ProcSchedClasses.back();
  for (const CodeGenSchedClass &SC : SchedModels.schedClasses()) {
    DEBUG(SC.dump(&SchedModels));

    SCTab.resize(SCTab.size() + 1);
    MCSchedClassDesc &SCDesc = SCTab.back();
    // SCDesc.Name is guarded by NDEBUG
    SCDesc.NumMicroOps = 0;
    SCDesc.BeginGroup = false;
    SCDesc.EndGroup = false;
    SCDesc.WriteProcResIdx = 0;
    SCDesc.WriteLatencyIdx = 0;
    SCDesc.ReadAdvanceIdx = 0;

    // A Variant SchedClass has no resources of its own.
    bool HasVariants = false;
    for (std::vector<CodeGenSchedTransition>::const_iterator
           TI = SC.Transitions.begin(), TE = SC.Transitions.end();
         TI != TE; ++TI) {
      if (TI->ProcIndices[0] == 0) {
        HasVariants = true;
        break;
      }
      if (is_contained(TI->ProcIndices, ProcModel.Index)) {
        HasVariants = true;
        break;
      }
    }
    if (HasVariants) {
      SCDesc.NumMicroOps = MCSchedClassDesc::VariantNumMicroOps;
      continue;
    }

    // Determine if the SchedClass is actually reachable on this processor. If
    // not don't try to locate the processor resources, it will fail.
    // If ProcIndices contains 0, this class applies to all processors.
    assert(!SC.ProcIndices.empty() && "expect at least one procidx");
    if (SC.ProcIndices[0] != 0) {
      if (!is_contained(SC.ProcIndices, ProcModel.Index))
        continue;
    }
    IdxVec Writes = SC.Writes;
    IdxVec Reads = SC.Reads;
    if (!SC.InstRWs.empty()) {
      // This class has a default ReadWrite list which can be overriden by
      // InstRW definitions.
      Record *RWDef = nullptr;
      for (Record *RW : SC.InstRWs) {
        Record *RWModelDef = RW->getValueAsDef("SchedModel");
        if (&ProcModel == &SchedModels.getProcModel(RWModelDef)) {
          RWDef = RW;
          break;
        }
      }
      if (RWDef) {
        Writes.clear();
        Reads.clear();
        SchedModels.findRWs(RWDef->getValueAsListOfDefs("OperandReadWrites"),
                            Writes, Reads);
      }
    }
    if (Writes.empty()) {
      // Check this processor's itinerary class resources.
      for (Record *I : ProcModel.ItinRWDefs) {
        RecVec Matched = I->getValueAsListOfDefs("MatchedItinClasses");
        if (is_contained(Matched, SC.ItinClassDef)) {
          SchedModels.findRWs(I->getValueAsListOfDefs("OperandReadWrites"),
                              Writes, Reads);
          break;
        }
      }
      if (Writes.empty()) {
        DEBUG(dbgs() << ProcModel.ModelName
              << " does not have resources for class " << SC.Name << '\n');
      }
    }
    // Sum resources across all operand writes.
    std::vector<MCWriteProcResEntry> WriteProcResources;
    std::vector<MCWriteLatencyEntry> WriteLatencies;
    std::vector<std::string> WriterNames;
    std::vector<MCReadAdvanceEntry> ReadAdvanceEntries;
    for (unsigned W : Writes) {
      IdxVec WriteSeq;
      SchedModels.expandRWSeqForProc(W, WriteSeq, /*IsRead=*/false,
                                     ProcModel);

      // For each operand, create a latency entry.
      MCWriteLatencyEntry WLEntry;
      WLEntry.Cycles = 0;
      unsigned WriteID = WriteSeq.back();
      WriterNames.push_back(SchedModels.getSchedWrite(WriteID).Name);
      // If this Write is not referenced by a ReadAdvance, don't distinguish it
      // from other WriteLatency entries.
      if (!SchedModels.hasReadOfWrite(
            SchedModels.getSchedWrite(WriteID).TheDef)) {
        WriteID = 0;
      }
      WLEntry.WriteResourceID = WriteID;

      for (unsigned WS : WriteSeq) {

        Record *WriteRes =
          FindWriteResources(SchedModels.getSchedWrite(WS), ProcModel);

        // Mark the parent class as invalid for unsupported write types.
        if (WriteRes->getValueAsBit("Unsupported")) {
          SCDesc.NumMicroOps = MCSchedClassDesc::InvalidNumMicroOps;
          break;
        }
        WLEntry.Cycles += WriteRes->getValueAsInt("Latency");
        SCDesc.NumMicroOps += WriteRes->getValueAsInt("NumMicroOps");
        SCDesc.BeginGroup |= WriteRes->getValueAsBit("BeginGroup");
        SCDesc.EndGroup |= WriteRes->getValueAsBit("EndGroup");
        SCDesc.BeginGroup |= WriteRes->getValueAsBit("SingleIssue");
        SCDesc.EndGroup |= WriteRes->getValueAsBit("SingleIssue");

        // Create an entry for each ProcResource listed in WriteRes.
        RecVec PRVec = WriteRes->getValueAsListOfDefs("ProcResources");
        std::vector<int64_t> Cycles =
          WriteRes->getValueAsListOfInts("ResourceCycles");

        ExpandProcResources(PRVec, Cycles, ProcModel);

        for (unsigned PRIdx = 0, PREnd = PRVec.size();
             PRIdx != PREnd; ++PRIdx) {
          MCWriteProcResEntry WPREntry;
          WPREntry.ProcResourceIdx = ProcModel.getProcResourceIdx(PRVec[PRIdx]);
          assert(WPREntry.ProcResourceIdx && "Bad ProcResourceIdx");
          WPREntry.Cycles = Cycles[PRIdx];
          // If this resource is already used in this sequence, add the current
          // entry's cycles so that the same resource appears to be used
          // serially, rather than multiple parallel uses. This is important for
          // in-order machine where the resource consumption is a hazard.
          unsigned WPRIdx = 0, WPREnd = WriteProcResources.size();
          for( ; WPRIdx != WPREnd; ++WPRIdx) {
            if (WriteProcResources[WPRIdx].ProcResourceIdx
                == WPREntry.ProcResourceIdx) {
              WriteProcResources[WPRIdx].Cycles += WPREntry.Cycles;
              break;
            }
          }
          if (WPRIdx == WPREnd)
            WriteProcResources.push_back(WPREntry);
        }
      }
      WriteLatencies.push_back(WLEntry);
    }
    // Create an entry for each operand Read in this SchedClass.
    // Entries must be sorted first by UseIdx then by WriteResourceID.
    for (unsigned UseIdx = 0, EndIdx = Reads.size();
         UseIdx != EndIdx; ++UseIdx) {
      Record *ReadAdvance =
        FindReadAdvance(SchedModels.getSchedRead(Reads[UseIdx]), ProcModel);
      if (!ReadAdvance)
        continue;

      // Mark the parent class as invalid for unsupported write types.
      if (ReadAdvance->getValueAsBit("Unsupported")) {
        SCDesc.NumMicroOps = MCSchedClassDesc::InvalidNumMicroOps;
        break;
      }
      RecVec ValidWrites = ReadAdvance->getValueAsListOfDefs("ValidWrites");
      IdxVec WriteIDs;
      if (ValidWrites.empty())
        WriteIDs.push_back(0);
      else {
        for (Record *VW : ValidWrites) {
          WriteIDs.push_back(SchedModels.getSchedRWIdx(VW, /*IsRead=*/false));
        }
      }
      std::sort(WriteIDs.begin(), WriteIDs.end());
      for(unsigned W : WriteIDs) {
        MCReadAdvanceEntry RAEntry;
        RAEntry.UseIdx = UseIdx;
        RAEntry.WriteResourceID = W;
        RAEntry.Cycles = ReadAdvance->getValueAsInt("Cycles");
        ReadAdvanceEntries.push_back(RAEntry);
      }
    }
    if (SCDesc.NumMicroOps == MCSchedClassDesc::InvalidNumMicroOps) {
      WriteProcResources.clear();
      WriteLatencies.clear();
      ReadAdvanceEntries.clear();
    }
    // Add the information for this SchedClass to the global tables using basic
    // compression.
    //
    // WritePrecRes entries are sorted by ProcResIdx.
    std::sort(WriteProcResources.begin(), WriteProcResources.end(),
              LessWriteProcResources());

    SCDesc.NumWriteProcResEntries = WriteProcResources.size();
    std::vector<MCWriteProcResEntry>::iterator WPRPos =
      std::search(SchedTables.WriteProcResources.begin(),
                  SchedTables.WriteProcResources.end(),
                  WriteProcResources.begin(), WriteProcResources.end());
    if (WPRPos != SchedTables.WriteProcResources.end())
      SCDesc.WriteProcResIdx = WPRPos - SchedTables.WriteProcResources.begin();
    else {
      SCDesc.WriteProcResIdx = SchedTables.WriteProcResources.size();
      SchedTables.WriteProcResources.insert(WPRPos, WriteProcResources.begin(),
                                            WriteProcResources.end());
    }
    // Latency entries must remain in operand order.
    SCDesc.NumWriteLatencyEntries = WriteLatencies.size();
    std::vector<MCWriteLatencyEntry>::iterator WLPos =
      std::search(SchedTables.WriteLatencies.begin(),
                  SchedTables.WriteLatencies.end(),
                  WriteLatencies.begin(), WriteLatencies.end());
    if (WLPos != SchedTables.WriteLatencies.end()) {
      unsigned idx = WLPos - SchedTables.WriteLatencies.begin();
      SCDesc.WriteLatencyIdx = idx;
      for (unsigned i = 0, e = WriteLatencies.size(); i < e; ++i)
        if (SchedTables.WriterNames[idx + i].find(WriterNames[i]) ==
            std::string::npos) {
          SchedTables.WriterNames[idx + i] += std::string("_") + WriterNames[i];
        }
    }
    else {
      SCDesc.WriteLatencyIdx = SchedTables.WriteLatencies.size();
      SchedTables.WriteLatencies.insert(SchedTables.WriteLatencies.end(),
                                        WriteLatencies.begin(),
                                        WriteLatencies.end());
      SchedTables.WriterNames.insert(SchedTables.WriterNames.end(),
                                     WriterNames.begin(), WriterNames.end());
    }
    // ReadAdvanceEntries must remain in operand order.
    SCDesc.NumReadAdvanceEntries = ReadAdvanceEntries.size();
    std::vector<MCReadAdvanceEntry>::iterator RAPos =
      std::search(SchedTables.ReadAdvanceEntries.begin(),
                  SchedTables.ReadAdvanceEntries.end(),
                  ReadAdvanceEntries.begin(), ReadAdvanceEntries.end());
    if (RAPos != SchedTables.ReadAdvanceEntries.end())
      SCDesc.ReadAdvanceIdx = RAPos - SchedTables.ReadAdvanceEntries.begin();
    else {
      SCDesc.ReadAdvanceIdx = SchedTables.ReadAdvanceEntries.size();
      SchedTables.ReadAdvanceEntries.insert(RAPos, ReadAdvanceEntries.begin(),
                                            ReadAdvanceEntries.end());
    }
  }
}

// Emit SchedClass tables for all processors and associated global tables.
void SubtargetEmitter::EmitSchedClassTables(SchedClassTables &SchedTables,
                                            raw_ostream &OS) {
  // Emit global WriteProcResTable.
  OS << "\n// {ProcResourceIdx, Cycles}\n"
     << "extern const llvm::MCWriteProcResEntry "
     << Target << "WriteProcResTable[] = {\n"
     << "  { 0,  0}, // Invalid\n";
  for (unsigned WPRIdx = 1, WPREnd = SchedTables.WriteProcResources.size();
       WPRIdx != WPREnd; ++WPRIdx) {
    MCWriteProcResEntry &WPREntry = SchedTables.WriteProcResources[WPRIdx];
    OS << "  {" << format("%2d", WPREntry.ProcResourceIdx) << ", "
       << format("%2d", WPREntry.Cycles) << "}";
    if (WPRIdx + 1 < WPREnd)
      OS << ',';
    OS << " // #" << WPRIdx << '\n';
  }
  OS << "}; // " << Target << "WriteProcResTable\n";

  // Emit global WriteLatencyTable.
  OS << "\n// {Cycles, WriteResourceID}\n"
     << "extern const llvm::MCWriteLatencyEntry "
     << Target << "WriteLatencyTable[] = {\n"
     << "  { 0,  0}, // Invalid\n";
  for (unsigned WLIdx = 1, WLEnd = SchedTables.WriteLatencies.size();
       WLIdx != WLEnd; ++WLIdx) {
    MCWriteLatencyEntry &WLEntry = SchedTables.WriteLatencies[WLIdx];
    OS << "  {" << format("%2d", WLEntry.Cycles) << ", "
       << format("%2d", WLEntry.WriteResourceID) << "}";
    if (WLIdx + 1 < WLEnd)
      OS << ',';
    OS << " // #" << WLIdx << " " << SchedTables.WriterNames[WLIdx] << '\n';
  }
  OS << "}; // " << Target << "WriteLatencyTable\n";

  // Emit global ReadAdvanceTable.
  OS << "\n// {UseIdx, WriteResourceID, Cycles}\n"
     << "extern const llvm::MCReadAdvanceEntry "
     << Target << "ReadAdvanceTable[] = {\n"
     << "  {0,  0,  0}, // Invalid\n";
  for (unsigned RAIdx = 1, RAEnd = SchedTables.ReadAdvanceEntries.size();
       RAIdx != RAEnd; ++RAIdx) {
    MCReadAdvanceEntry &RAEntry = SchedTables.ReadAdvanceEntries[RAIdx];
    OS << "  {" << RAEntry.UseIdx << ", "
       << format("%2d", RAEntry.WriteResourceID) << ", "
       << format("%2d", RAEntry.Cycles) << "}";
    if (RAIdx + 1 < RAEnd)
      OS << ',';
    OS << " // #" << RAIdx << '\n';
  }
  OS << "}; // " << Target << "ReadAdvanceTable\n";

  // Emit a SchedClass table for each processor.
  for (CodeGenSchedModels::ProcIter PI = SchedModels.procModelBegin(),
         PE = SchedModels.procModelEnd(); PI != PE; ++PI) {
    if (!PI->hasInstrSchedModel())
      continue;

    std::vector<MCSchedClassDesc> &SCTab =
      SchedTables.ProcSchedClasses[1 + (PI - SchedModels.procModelBegin())];

    OS << "\n// {Name, NumMicroOps, BeginGroup, EndGroup,"
       << " WriteProcResIdx,#, WriteLatencyIdx,#, ReadAdvanceIdx,#}\n";
    OS << "static const llvm::MCSchedClassDesc "
       << PI->ModelName << "SchedClasses[] = {\n";

    // The first class is always invalid. We no way to distinguish it except by
    // name and position.
    assert(SchedModels.getSchedClass(0).Name == "NoInstrModel"
           && "invalid class not first");
    OS << "  {DBGFIELD(\"InvalidSchedClass\")  "
       << MCSchedClassDesc::InvalidNumMicroOps
       << ", false, false,  0, 0,  0, 0,  0, 0},\n";

    for (unsigned SCIdx = 1, SCEnd = SCTab.size(); SCIdx != SCEnd; ++SCIdx) {
      MCSchedClassDesc &MCDesc = SCTab[SCIdx];
      const CodeGenSchedClass &SchedClass = SchedModels.getSchedClass(SCIdx);
      OS << "  {DBGFIELD(\"" << SchedClass.Name << "\") ";
      if (SchedClass.Name.size() < 18)
        OS.indent(18 - SchedClass.Name.size());
      OS << MCDesc.NumMicroOps
         << ", " << ( MCDesc.BeginGroup ? "true" : "false" )
         << ", " << ( MCDesc.EndGroup ? "true" : "false" )
         << ", " << format("%2d", MCDesc.WriteProcResIdx)
         << ", " << MCDesc.NumWriteProcResEntries
         << ", " << format("%2d", MCDesc.WriteLatencyIdx)
         << ", " << MCDesc.NumWriteLatencyEntries
         << ", " << format("%2d", MCDesc.ReadAdvanceIdx)
         << ", " << MCDesc.NumReadAdvanceEntries << "}";
      if (SCIdx + 1 < SCEnd)
        OS << ',';
      OS << " // #" << SCIdx << '\n';
    }
    OS << "}; // " << PI->ModelName << "SchedClasses\n";
  }
}

void SubtargetEmitter::EmitProcessorModels(raw_ostream &OS) {
  // For each processor model.
  for (const CodeGenProcModel &PM : SchedModels.procModels()) {
    // Emit processor resource table.
    if (PM.hasInstrSchedModel())
      EmitProcessorResources(PM, OS);
    else if(!PM.ProcResourceDefs.empty())
      PrintFatalError(PM.ModelDef->getLoc(), "SchedMachineModel defines "
                    "ProcResources without defining WriteRes SchedWriteRes");

    // Begin processor itinerary properties
    OS << "\n";
    OS << "static const llvm::MCSchedModel " << PM.ModelName << " = {\n";
    EmitProcessorProp(OS, PM.ModelDef, "IssueWidth", ',');
    EmitProcessorProp(OS, PM.ModelDef, "MicroOpBufferSize", ',');
    EmitProcessorProp(OS, PM.ModelDef, "LoopMicroOpBufferSize", ',');
    EmitProcessorProp(OS, PM.ModelDef, "LoadLatency", ',');
    EmitProcessorProp(OS, PM.ModelDef, "HighLatency", ',');
    EmitProcessorProp(OS, PM.ModelDef, "MispredictPenalty", ',');

    bool PostRAScheduler =
      (PM.ModelDef ? PM.ModelDef->getValueAsBit("PostRAScheduler") : false);

    OS << "  " << (PostRAScheduler ? "true" : "false")  << ", // "
       << "PostRAScheduler\n";

    bool CompleteModel =
      (PM.ModelDef ? PM.ModelDef->getValueAsBit("CompleteModel") : false);

    OS << "  " << (CompleteModel ? "true" : "false") << ", // "
       << "CompleteModel\n";

    OS << "  " << PM.Index << ", // Processor ID\n";
    if (PM.hasInstrSchedModel())
      OS << "  " << PM.ModelName << "ProcResources" << ",\n"
         << "  " << PM.ModelName << "SchedClasses" << ",\n"
         << "  " << PM.ProcResourceDefs.size()+1 << ",\n"
         << "  " << (SchedModels.schedClassEnd()
                     - SchedModels.schedClassBegin()) << ",\n";
    else
      OS << "  nullptr, nullptr, 0, 0,"
         << " // No instruction-level machine model.\n";
    if (PM.hasItineraries())
      OS << "  " << PM.ItinsDef->getName() << "};\n";
    else
      OS << "  nullptr}; // No Itinerary\n";
  }
}

//
// EmitProcessorLookup - generate cpu name to itinerary lookup table.
//
void SubtargetEmitter::EmitProcessorLookup(raw_ostream &OS) {
  // Gather and sort processor information
  std::vector<Record*> ProcessorList =
                          Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcessorList.begin(), ProcessorList.end(), LessRecordFieldName());

  // Begin processor table
  OS << "\n";
  OS << "// Sorted (by key) array of itineraries for CPU subtype.\n"
     << "extern const llvm::SubtargetInfoKV "
     << Target << "ProcSchedKV[] = {\n";

  // For each processor
  for (unsigned i = 0, N = ProcessorList.size(); i < N;) {
    // Next processor
    Record *Processor = ProcessorList[i];

    const std::string &Name = Processor->getValueAsString("Name");
    const std::string &ProcModelName =
      SchedModels.getModelForProc(Processor).ModelName;

    // Emit as { "cpu", procinit },
    OS << "  { \"" << Name << "\", (const void *)&" << ProcModelName << " }";

    // Depending on ''if more in the list'' emit comma
    if (++i < N) OS << ",";

    OS << "\n";
  }

  // End processor table
  OS << "};\n";
}

//
// EmitSchedModel - Emits all scheduling model tables, folding common patterns.
//
void SubtargetEmitter::EmitSchedModel(raw_ostream &OS) {
  OS << "#ifdef DBGFIELD\n"
     << "#error \"<target>GenSubtargetInfo.inc requires a DBGFIELD macro\"\n"
     << "#endif\n"
     << "#ifndef NDEBUG\n"
     << "#define DBGFIELD(x) x,\n"
     << "#else\n"
     << "#define DBGFIELD(x)\n"
     << "#endif\n";

  if (SchedModels.hasItineraries()) {
    std::vector<std::vector<InstrItinerary>> ProcItinLists;
    // Emit the stage data
    EmitStageAndOperandCycleData(OS, ProcItinLists);
    EmitItineraries(OS, ProcItinLists);
  }
  OS << "\n// ===============================================================\n"
     << "// Data tables for the new per-operand machine model.\n";

  SchedClassTables SchedTables;
  for (const CodeGenProcModel &ProcModel : SchedModels.procModels()) {
    GenSchedClassTables(ProcModel, SchedTables);
  }
  EmitSchedClassTables(SchedTables, OS);

  // Emit the processor machine model
  EmitProcessorModels(OS);
  // Emit the processor lookup data
  EmitProcessorLookup(OS);

  OS << "#undef DBGFIELD";
}

void SubtargetEmitter::EmitSchedModelHelpers(const std::string &ClassName,
                                             raw_ostream &OS) {
  OS << "unsigned " << ClassName
     << "\n::resolveSchedClass(unsigned SchedClass, const MachineInstr *MI,"
     << " const TargetSchedModel *SchedModel) const {\n";

  std::vector<Record*> Prologs = Records.getAllDerivedDefinitions("PredicateProlog");
  std::sort(Prologs.begin(), Prologs.end(), LessRecord());
  for (Record *P : Prologs) {
    OS << P->getValueAsString("Code") << '\n';
  }
  IdxVec VariantClasses;
  for (const CodeGenSchedClass &SC : SchedModels.schedClasses()) {
    if (SC.Transitions.empty())
      continue;
    VariantClasses.push_back(SC.Index);
  }
  if (!VariantClasses.empty()) {
    OS << "  switch (SchedClass) {\n";
    for (unsigned VC : VariantClasses) {
      const CodeGenSchedClass &SC = SchedModels.getSchedClass(VC);
      OS << "  case " << VC << ": // " << SC.Name << '\n';
      IdxVec ProcIndices;
      for (const CodeGenSchedTransition &T : SC.Transitions) {
        IdxVec PI;
        std::set_union(T.ProcIndices.begin(), T.ProcIndices.end(),
                       ProcIndices.begin(), ProcIndices.end(),
                       std::back_inserter(PI));
        ProcIndices.swap(PI);
      }
      for (unsigned PI : ProcIndices) {
        OS << "    ";
        if (PI != 0)
          OS << "if (SchedModel->getProcessorID() == " << PI << ") ";
        OS << "{ // " << (SchedModels.procModelBegin() + PI)->ModelName
           << '\n';
        for (const CodeGenSchedTransition &T : SC.Transitions) {
          if (PI != 0 && !std::count(T.ProcIndices.begin(),
                                     T.ProcIndices.end(), PI)) {
              continue;
          }
          OS << "      if (";
          for (RecIter RI = T.PredTerm.begin(), RE = T.PredTerm.end();
               RI != RE; ++RI) {
            if (RI != T.PredTerm.begin())
              OS << "\n          && ";
            OS << "(" << (*RI)->getValueAsString("Predicate") << ")";
          }
          OS << ")\n"
             << "        return " << T.ToClassIdx << "; // "
             << SchedModels.getSchedClass(T.ToClassIdx).Name << '\n';
        }
        OS << "    }\n";
        if (PI == 0)
          break;
      }
      if (SC.isInferred())
        OS << "    return " << SC.Index << ";\n";
      OS << "    break;\n";
    }
    OS << "  };\n";
  }
  OS << "  report_fatal_error(\"Expected a variant SchedClass\");\n"
     << "} // " << ClassName << "::resolveSchedClass\n";
}

//
// ParseFeaturesFunction - Produces a subtarget specific function for parsing
// the subtarget features string.
//
void SubtargetEmitter::ParseFeaturesFunction(raw_ostream &OS,
                                             unsigned NumFeatures,
                                             unsigned NumProcs) {
  std::vector<Record*> Features =
                       Records.getAllDerivedDefinitions("SubtargetFeature");
  std::sort(Features.begin(), Features.end(), LessRecord());

  OS << "// ParseSubtargetFeatures - Parses features string setting specified\n"
     << "// subtarget options.\n"
     << "void llvm::";
  OS << Target;
  OS << "Subtarget::ParseSubtargetFeatures(StringRef CPU, StringRef FS) {\n"
     << "  DEBUG(dbgs() << \"\\nFeatures:\" << FS);\n"
     << "  DEBUG(dbgs() << \"\\nCPU:\" << CPU << \"\\n\\n\");\n";

  if (Features.empty()) {
    OS << "}\n";
    return;
  }

  OS << "  InitMCProcessorInfo(CPU, FS);\n"
     << "  const FeatureBitset& Bits = getFeatureBits();\n";

  for (Record *R : Features) {
    // Next record
    const std::string &Instance = R->getName();
    const std::string &Value = R->getValueAsString("Value");
    const std::string &Attribute = R->getValueAsString("Attribute");

    if (Value=="true" || Value=="false")
      OS << "  if (Bits[" << Target << "::"
         << Instance << "]) "
         << Attribute << " = " << Value << ";\n";
    else
      OS << "  if (Bits[" << Target << "::"
         << Instance << "] && "
         << Attribute << " < " << Value << ") "
         << Attribute << " = " << Value << ";\n";
  }

  OS << "}\n";
}

//
// SubtargetEmitter::run - Main subtarget enumeration emitter.
//
void SubtargetEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Subtarget Enumeration Source Fragment", OS);

  OS << "\n#ifdef GET_SUBTARGETINFO_ENUM\n";
  OS << "#undef GET_SUBTARGETINFO_ENUM\n\n";

  OS << "namespace llvm {\n";
  Enumeration(OS);
  OS << "} // end namespace llvm\n\n";
  OS << "#endif // GET_SUBTARGETINFO_ENUM\n\n";

  OS << "\n#ifdef GET_SUBTARGETINFO_MC_DESC\n";
  OS << "#undef GET_SUBTARGETINFO_MC_DESC\n\n";

  OS << "namespace llvm {\n";
#if 0
  OS << "namespace {\n";
#endif
  unsigned NumFeatures = FeatureKeyValues(OS);
  OS << "\n";
  unsigned NumProcs = CPUKeyValues(OS);
  OS << "\n";
  EmitSchedModel(OS);
  OS << "\n";
#if 0
  OS << "} // end anonymous namespace\n\n";
#endif

  // MCInstrInfo initialization routine.
  OS << "static inline MCSubtargetInfo *create" << Target
     << "MCSubtargetInfoImpl("
     << "const Triple &TT, StringRef CPU, StringRef FS) {\n";
  OS << "  return new MCSubtargetInfo(TT, CPU, FS, ";
  if (NumFeatures)
    OS << Target << "FeatureKV, ";
  else
    OS << "None, ";
  if (NumProcs)
    OS << Target << "SubTypeKV, ";
  else
    OS << "None, ";
  OS << '\n'; OS.indent(22);
  OS << Target << "ProcSchedKV, "
     << Target << "WriteProcResTable, "
     << Target << "WriteLatencyTable, "
     << Target << "ReadAdvanceTable, ";
  OS << '\n'; OS.indent(22);
  if (SchedModels.hasItineraries()) {
    OS << Target << "Stages, "
       << Target << "OperandCycles, "
       << Target << "ForwardingPaths";
  } else
    OS << "nullptr, nullptr, nullptr";
  OS << ");\n}\n\n";

  OS << "} // end namespace llvm\n\n";

  OS << "#endif // GET_SUBTARGETINFO_MC_DESC\n\n";

  OS << "\n#ifdef GET_SUBTARGETINFO_TARGET_DESC\n";
  OS << "#undef GET_SUBTARGETINFO_TARGET_DESC\n\n";

  OS << "#include \"llvm/Support/Debug.h\"\n";
  OS << "#include \"llvm/Support/raw_ostream.h\"\n\n";
  ParseFeaturesFunction(OS, NumFeatures, NumProcs);

  OS << "#endif // GET_SUBTARGETINFO_TARGET_DESC\n\n";

  // Create a TargetSubtargetInfo subclass to hide the MC layer initialization.
  OS << "\n#ifdef GET_SUBTARGETINFO_HEADER\n";
  OS << "#undef GET_SUBTARGETINFO_HEADER\n\n";

  std::string ClassName = Target + "GenSubtargetInfo";
  OS << "namespace llvm {\n";
  OS << "class DFAPacketizer;\n";
  OS << "struct " << ClassName << " : public TargetSubtargetInfo {\n"
     << "  explicit " << ClassName << "(const Triple &TT, StringRef CPU, "
     << "StringRef FS);\n"
     << "public:\n"
     << "  unsigned resolveSchedClass(unsigned SchedClass, "
     << " const MachineInstr *DefMI,"
     << " const TargetSchedModel *SchedModel) const override;\n"
     << "  DFAPacketizer *createDFAPacketizer(const InstrItineraryData *IID)"
     << " const;\n"
     << "};\n";
  OS << "} // end namespace llvm\n\n";

  OS << "#endif // GET_SUBTARGETINFO_HEADER\n\n";

  OS << "\n#ifdef GET_SUBTARGETINFO_CTOR\n";
  OS << "#undef GET_SUBTARGETINFO_CTOR\n\n";

  OS << "#include \"llvm/CodeGen/TargetSchedule.h\"\n\n";
  OS << "namespace llvm {\n";
  OS << "extern const llvm::SubtargetFeatureKV " << Target << "FeatureKV[];\n";
  OS << "extern const llvm::SubtargetFeatureKV " << Target << "SubTypeKV[];\n";
  OS << "extern const llvm::SubtargetInfoKV " << Target << "ProcSchedKV[];\n";
  OS << "extern const llvm::MCWriteProcResEntry "
     << Target << "WriteProcResTable[];\n";
  OS << "extern const llvm::MCWriteLatencyEntry "
     << Target << "WriteLatencyTable[];\n";
  OS << "extern const llvm::MCReadAdvanceEntry "
     << Target << "ReadAdvanceTable[];\n";

  if (SchedModels.hasItineraries()) {
    OS << "extern const llvm::InstrStage " << Target << "Stages[];\n";
    OS << "extern const unsigned " << Target << "OperandCycles[];\n";
    OS << "extern const unsigned " << Target << "ForwardingPaths[];\n";
  }

  OS << ClassName << "::" << ClassName << "(const Triple &TT, StringRef CPU, "
     << "StringRef FS)\n"
     << "  : TargetSubtargetInfo(TT, CPU, FS, ";
  if (NumFeatures)
    OS << "makeArrayRef(" << Target << "FeatureKV, " << NumFeatures << "), ";
  else
    OS << "None, ";
  if (NumProcs)
    OS << "makeArrayRef(" << Target << "SubTypeKV, " << NumProcs << "), ";
  else
    OS << "None, ";
  OS << '\n'; OS.indent(24);
  OS << Target << "ProcSchedKV, "
     << Target << "WriteProcResTable, "
     << Target << "WriteLatencyTable, "
     << Target << "ReadAdvanceTable, ";
  OS << '\n'; OS.indent(24);
  if (SchedModels.hasItineraries()) {
    OS << Target << "Stages, "
       << Target << "OperandCycles, "
       << Target << "ForwardingPaths";
  } else
    OS << "nullptr, nullptr, nullptr";
  OS << ") {}\n\n";

  EmitSchedModelHelpers(ClassName, OS);

  OS << "} // end namespace llvm\n\n";

  OS << "#endif // GET_SUBTARGETINFO_CTOR\n\n";
}

namespace llvm {

void EmitSubtarget(RecordKeeper &RK, raw_ostream &OS) {
  CodeGenTarget CGTarget(RK);
  SubtargetEmitter(RK, CGTarget).run(OS);
}

} // end namespace llvm
