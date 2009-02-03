//===- SubtargetEmitter.h - Generate subtarget enumerations -----*- C++ -*-===//
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

#ifndef SUBTARGET_EMITTER_H
#define SUBTARGET_EMITTER_H

#include "TableGenBackend.h"
#include "llvm/Target/TargetInstrItineraries.h"
#include <vector>
#include <map>
#include <string>


namespace llvm {

class SubtargetEmitter : public TableGenBackend {
  
  RecordKeeper &Records;
  std::string Target;
  bool HasItineraries;
  
  void Enumeration(std::ostream &OS, const char *ClassName, bool isBits);
  void FeatureKeyValues(std::ostream &OS);
  void CPUKeyValues(std::ostream &OS);
  unsigned CollectAllItinClasses(std::ostream &OS,
                               std::map<std::string, unsigned> &ItinClassesMap);
  void FormItineraryString(Record *ItinData, std::string &ItinString,
                           unsigned &NStages);
  void EmitStageData(std::ostream &OS, unsigned NItinClasses,
                     std::map<std::string, unsigned> &ItinClassesMap,
                     std::vector<std::vector<InstrItinerary> > &ProcList);
  void EmitProcessorData(std::ostream &OS,
                       std::vector<std::vector<InstrItinerary> > &ProcList);
  void EmitProcessorLookup(std::ostream &OS);
  void EmitData(std::ostream &OS);
  void ParseFeaturesFunction(std::ostream &OS);
  
public:
  SubtargetEmitter(RecordKeeper &R) : Records(R), HasItineraries(false) {}

  // run - Output the subtarget enumerations, returning true on failure.
  void run(std::ostream &o);

};


} // End llvm namespace

#endif



