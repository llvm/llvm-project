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

#include "SubtargetEmitter.h"
#include "CodeGenTarget.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
using namespace llvm;

//
// Enumeration - Emit the specified class as an enumeration.
//
void SubtargetEmitter::Enumeration(std::ostream &OS,
                                   const char *ClassName,
                                   bool isBits) {
  // Get all records of class and sort
  std::vector<Record*> DefList = Records.getAllDerivedDefinitions(ClassName);
  std::sort(DefList.begin(), DefList.end(), LessRecord());

  // Open enumeration
  OS << "enum {\n";
  
  // For each record
  for (unsigned i = 0, N = DefList.size(); i < N;) {
    // Next record
    Record *Def = DefList[i];
    
    // Get and emit name
    OS << "  " << Def->getName();
    
    // If bit flags then emit expression (1 << i)
    if (isBits)  OS << " = " << " 1 << " << i;

    // Depending on 'if more in the list' emit comma
    if (++i < N) OS << ",";
    
    OS << "\n";
  }
  
  // Close enumeration
  OS << "};\n";
}

//
// FeatureKeyValues - Emit data of all the subtarget features.  Used by the
// command line.
//
void SubtargetEmitter::FeatureKeyValues(std::ostream &OS) {
  // Gather and sort all the features
  std::vector<Record*> FeatureList =
                           Records.getAllDerivedDefinitions("SubtargetFeature");
  std::sort(FeatureList.begin(), FeatureList.end(), LessRecordFieldName());

  // Begin feature table
  OS << "// Sorted (by key) array of values for CPU features.\n"
     << "static const llvm::SubtargetFeatureKV FeatureKV[] = {\n";
  
  // For each feature
  for (unsigned i = 0, N = FeatureList.size(); i < N; ++i) {
    // Next feature
    Record *Feature = FeatureList[i];

    const std::string &Name = Feature->getName();
    const std::string &CommandLineName = Feature->getValueAsString("Name");
    const std::string &Desc = Feature->getValueAsString("Desc");
    
    if (CommandLineName.empty()) continue;
    
    // Emit as { "feature", "decription", feactureEnum, i1 | i2 | ... | in }
    OS << "  { "
       << "\"" << CommandLineName << "\", "
       << "\"" << Desc << "\", "
       << Name << ", ";

    const std::vector<Record*> &ImpliesList = 
      Feature->getValueAsListOfDefs("Implies");
    
    if (ImpliesList.empty()) {
      OS << "0";
    } else {
      for (unsigned j = 0, M = ImpliesList.size(); j < M;) {
        OS << ImpliesList[j]->getName();
        if (++j < M) OS << " | ";
      }
    }

    OS << " }";
    
    // Depending on 'if more in the list' emit comma
    if ((i + 1) < N) OS << ",";
    
    OS << "\n";
  }
  
  // End feature table
  OS << "};\n";

  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  FeatureKVSize = sizeof(FeatureKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
}

//
// CPUKeyValues - Emit data of all the subtarget processors.  Used by command
// line.
//
void SubtargetEmitter::CPUKeyValues(std::ostream &OS) {
  // Gather and sort processor information
  std::vector<Record*> ProcessorList =
                          Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcessorList.begin(), ProcessorList.end(), LessRecordFieldName());

  // Begin processor table
  OS << "// Sorted (by key) array of values for CPU subtype.\n"
     << "static const llvm::SubtargetFeatureKV SubTypeKV[] = {\n";
     
  // For each processor
  for (unsigned i = 0, N = ProcessorList.size(); i < N;) {
    // Next processor
    Record *Processor = ProcessorList[i];

    const std::string &Name = Processor->getValueAsString("Name");
    const std::vector<Record*> &FeatureList = 
      Processor->getValueAsListOfDefs("Features");
    
    // Emit as { "cpu", "description", f1 | f2 | ... fn },
    OS << "  { "
       << "\"" << Name << "\", "
       << "\"Select the " << Name << " processor\", ";
    
    if (FeatureList.empty()) {
      OS << "0";
    } else {
      for (unsigned j = 0, M = FeatureList.size(); j < M;) {
        OS << FeatureList[j]->getName();
        if (++j < M) OS << " | ";
      }
    }
    
    // The "0" is for the "implies" section of this data structure.
    OS << ", 0 }";
    
    // Depending on 'if more in the list' emit comma
    if (++i < N) OS << ",";
    
    OS << "\n";
  }
  
  // End processor table
  OS << "};\n";

  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  SubTypeKVSize = sizeof(SubTypeKV)/sizeof(llvm::SubtargetFeatureKV)\n";
  OS<<"};\n";
}

//
// CollectAllItinClasses - Gathers and enumerates all the itinerary classes.
// Returns itinerary class count.
//
unsigned SubtargetEmitter::CollectAllItinClasses(std::ostream &OS,
                              std::map<std::string, unsigned> &ItinClassesMap) {
  // Gather and sort all itinerary classes
  std::vector<Record*> ItinClassList =
                            Records.getAllDerivedDefinitions("InstrItinClass");
  std::sort(ItinClassList.begin(), ItinClassList.end(), LessRecord());

  // For each itinerary class
  unsigned N = ItinClassList.size();
  for (unsigned i = 0; i < N; i++) {
    // Next itinerary class
    const Record *ItinClass = ItinClassList[i];
    // Get name of itinerary class
    // Assign itinerary class a unique number
    ItinClassesMap[ItinClass->getName()] = i;
  }
  
  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  ItinClassesSize = " << N << "\n";
  OS<<"};\n";

  // Return itinerary class count
  return N;
}

//
// FormItineraryString - Compose a string containing the data initialization
// for the specified itinerary.  N is the number of stages.
//
void SubtargetEmitter::FormItineraryString(Record *ItinData,
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
  
    // Form string as ,{ cycles, u1 | u2 | ... | un }
    int Cycles = Stage->getValueAsInt("Cycles");
    ItinString += "  { " + itostr(Cycles) + ", ";
    
    // Get unit list
    const std::vector<Record*> &UnitList = Stage->getValueAsListOfDefs("Units");
    
    // For each unit
    for (unsigned j = 0, M = UnitList.size(); j < M;) {
      // Add name and bitwise or
      ItinString += UnitList[j]->getName();
      if (++j < M) ItinString += " | ";
    }
    
    // Close off stage
    ItinString += " }";
    if (++i < N) ItinString += ", ";
  }
}

//
// EmitStageData - Generate unique itinerary stages.  Record itineraries for 
// processors.
//
void SubtargetEmitter::EmitStageData(std::ostream &OS,
       unsigned NItinClasses,
       std::map<std::string, unsigned> &ItinClassesMap, 
       std::vector<std::vector<InstrItinerary> > &ProcList) {
  // Gather processor iteraries
  std::vector<Record*> ProcItinList =
                       Records.getAllDerivedDefinitions("ProcessorItineraries");
  
  // If just no itinerary then don't bother
  if (ProcItinList.size() < 2) return;

  // Begin stages table
  OS << "static const llvm::InstrStage Stages[] = {\n"
        "  { 0, 0 }, // No itinerary\n";
        
  unsigned StageCount = 1;
  unsigned ItinEnum = 1;
  std::map<std::string, unsigned> ItinMap;
  for (unsigned i = 0, N = ProcItinList.size(); i < N; i++) {
    // Next record
    Record *Proc = ProcItinList[i];
    
    // Get processor itinerary name
    const std::string &Name = Proc->getName();
    
    // Skip default
    if (Name == "NoItineraries") continue;
    
    // Create and expand processor itinerary to cover all itinerary classes
    std::vector<InstrItinerary> ItinList;
    ItinList.resize(NItinClasses);
    
    // Get itinerary data list
    std::vector<Record*> ItinDataList = Proc->getValueAsListOfDefs("IID");
    
    // For each itinerary data
    for (unsigned j = 0, M = ItinDataList.size(); j < M; j++) {
      // Next itinerary data
      Record *ItinData = ItinDataList[j];
      
      // Get string and stage count
      std::string ItinString;
      unsigned NStages;
      FormItineraryString(ItinData, ItinString, NStages);

      // Check to see if it already exists
      unsigned Find = ItinMap[ItinString];
      
      // If new itinerary
      if (Find == 0) {
        // Emit as { cycles, u1 | u2 | ... | un }, // index
        OS << ItinString << ", // " << ItinEnum << "\n";
        // Record Itin class number.
        ItinMap[ItinString] = Find = StageCount;
        StageCount += NStages;
        ItinEnum++;
      }
      
      // Set up itinerary as location and location + stage count
      InstrItinerary Intinerary = { Find, Find + NStages };

      // Locate where to inject into processor itinerary table
      const std::string &Name = ItinData->getValueAsDef("TheClass")->getName();
      Find = ItinClassesMap[Name];
      
      // Inject - empty slots will be 0, 0
      ItinList[Find] = Intinerary;
    }
    
    // Add process itinerary to list
    ProcList.push_back(ItinList);
  }
  
  // Closing stage
  OS << "  { 0, 0 } // End itinerary\n";
  // End stages table
  OS << "};\n";
  
  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  StagesSize = sizeof(Stages)/sizeof(llvm::InstrStage)\n";
  OS<<"};\n";
}

//
// EmitProcessorData - Generate data for processor itineraries.
//
void SubtargetEmitter::EmitProcessorData(std::ostream &OS,
      std::vector<std::vector<InstrItinerary> > &ProcList) {
  // Get an iterator for processor itinerary stages
  std::vector<std::vector<InstrItinerary> >::iterator
      ProcListIter = ProcList.begin();
  
  // For each processor itinerary
  std::vector<Record*> Itins =
                       Records.getAllDerivedDefinitions("ProcessorItineraries");
  for (unsigned i = 0, N = Itins.size(); i < N; i++) {
    // Next record
    Record *Itin = Itins[i];

    // Get processor itinerary name
    const std::string &Name = Itin->getName();
    
    // Skip default
    if (Name == "NoItineraries") continue;

    // Begin processor itinerary table
    OS << "\n";
    OS << "static const llvm::InstrItinerary " << Name << "[] = {\n";
    
    // For each itinerary class
    std::vector<InstrItinerary> &ItinList = *ProcListIter++;
    for (unsigned j = 0, M = ItinList.size(); j < M;) {
      InstrItinerary &Intinerary = ItinList[j];
      
      // Emit in the form of { first, last } // index
      if (Intinerary.First == 0) {
        OS << "  { 0, 0 }";
      } else {
        OS << "  { " << Intinerary.First << ", " << Intinerary.Last << " }";
      }
      
      // If more in list add comma
      if (++j < M) OS << ",";
      
      OS << " // " << (j - 1) << "\n";
    }
    
    // End processor itinerary table
    OS << "};\n";
  }
}

//
// EmitProcessorLookup - generate cpu name to itinerary lookup table.
//
void SubtargetEmitter::EmitProcessorLookup(std::ostream &OS) {
  // Gather and sort processor information
  std::vector<Record*> ProcessorList =
                          Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcessorList.begin(), ProcessorList.end(), LessRecordFieldName());

  // Begin processor table
  OS << "\n";
  OS << "// Sorted (by key) array of itineraries for CPU subtype.\n"
     << "static const llvm::SubtargetInfoKV ProcItinKV[] = {\n";
     
  // For each processor
  for (unsigned i = 0, N = ProcessorList.size(); i < N;) {
    // Next processor
    Record *Processor = ProcessorList[i];

    const std::string &Name = Processor->getValueAsString("Name");
    const std::string &ProcItin =
      Processor->getValueAsDef("ProcItin")->getName();
    
    // Emit as { "cpu", procinit },
    OS << "  { "
       << "\"" << Name << "\", "
       << "(void *)&" << ProcItin;
        
    OS << " }";
    
    // Depending on ''if more in the list'' emit comma
    if (++i < N) OS << ",";
    
    OS << "\n";
  }
  
  // End processor table
  OS << "};\n";

  // Emit size of table
  OS<<"\nenum {\n";
  OS<<"  ProcItinKVSize = sizeof(ProcItinKV)/"
                            "sizeof(llvm::SubtargetInfoKV)\n";
  OS<<"};\n";
}

//
// EmitData - Emits all stages and itineries, folding common patterns.
//
void SubtargetEmitter::EmitData(std::ostream &OS) {
  std::map<std::string, unsigned> ItinClassesMap;
  std::vector<std::vector<InstrItinerary> > ProcList;
  
  // Enumerate all the itinerary classes
  unsigned NItinClasses = CollectAllItinClasses(OS, ItinClassesMap);
  // Make sure the rest is worth the effort
  HasItineraries = NItinClasses != 1;   // Ignore NoItinerary.
  
  if (HasItineraries) {
    // Emit the stage data
    EmitStageData(OS, NItinClasses, ItinClassesMap, ProcList);
    // Emit the processor itinerary data
    EmitProcessorData(OS, ProcList);
    // Emit the processor lookup data
    EmitProcessorLookup(OS);
  }
}

//
// ParseFeaturesFunction - Produces a subtarget specific function for parsing
// the subtarget features string.
//
void SubtargetEmitter::ParseFeaturesFunction(std::ostream &OS) {
  std::vector<Record*> Features =
                       Records.getAllDerivedDefinitions("SubtargetFeature");
  std::sort(Features.begin(), Features.end(), LessRecord());

  OS << "// ParseSubtargetFeatures - Parses features string setting specified\n" 
     << "// subtarget options.\n" 
     << "void llvm::";
  OS << Target;
  OS << "Subtarget::ParseSubtargetFeatures(const std::string &FS,\n"
     << "                                  const std::string &CPU) {\n"
     << "  SubtargetFeatures Features(FS);\n"
     << "  Features.setCPUIfNone(CPU);\n"
     << "  uint32_t Bits =  Features.getBits(SubTypeKV, SubTypeKVSize,\n"
     << "                                    FeatureKV, FeatureKVSize);\n";

  for (unsigned i = 0; i < Features.size(); i++) {
    // Next record
    Record *R = Features[i];
    const std::string &Instance = R->getName();
    const std::string &Value = R->getValueAsString("Value");
    const std::string &Attribute = R->getValueAsString("Attribute");

    if (Value=="true" || Value=="false")
      OS << "  if ((Bits & " << Instance << ") != 0) "
         << Attribute << " = " << Value << ";\n";
    else
      OS << "  if ((Bits & " << Instance << ") != 0 && " << Attribute << 
            " < " << Value << ") " << Attribute << " = " << Value << ";\n";
  }

  if (HasItineraries) {
    OS << "\n"
       << "  InstrItinerary *Itinerary = (InstrItinerary *)"
       <<              "Features.getInfo(ProcItinKV, ProcItinKVSize);\n"
       << "  InstrItins = InstrItineraryData(Stages, Itinerary);\n";
  }
  
  OS << "}\n";
}

// 
// SubtargetEmitter::run - Main subtarget enumeration emitter.
//
void SubtargetEmitter::run(std::ostream &OS) {
  Target = CodeGenTarget().getName();

  EmitSourceFileHeader("Subtarget Enumeration Source Fragment", OS);

  OS << "#include \"llvm/Target/SubtargetFeature.h\"\n";
  OS << "#include \"llvm/Target/TargetInstrItineraries.h\"\n\n";
  
  Enumeration(OS, "FuncUnit", true);
  OS<<"\n";
//  Enumeration(OS, "InstrItinClass", false);
//  OS<<"\n";
  Enumeration(OS, "SubtargetFeature", true);
  OS<<"\n";
  FeatureKeyValues(OS);
  OS<<"\n";
  CPUKeyValues(OS);
  OS<<"\n";
  EmitData(OS);
  OS<<"\n";
  ParseFeaturesFunction(OS);
}
