//===- TargetSchedule.td - Target Independent Scheduling ---*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent scheduling interfaces which should
// be implemented by each target which is using TableGen based scheduling.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Processor functional unit - These values represent the function units
// available across all chip sets for the target.  Eg., IntUnit, FPUnit, ...
// These may be independent values for each chip set or may be shared across
// all chip sets of the target.  Each functional unit is treated as a resource
// during scheduling and has an affect instruction order based on availability
// during a time interval.
//  
class FuncUnit;

//===----------------------------------------------------------------------===//
// Instruction stage - These values represent a step in the execution of an
// instruction.  The latency represents the number of discrete time slots used
// need to complete the stage.  Units represent the choice of functional units
// that can be used to complete the stage.  Eg. IntUnit1, IntUnit2.
//
class InstrStage<int cycles, list<FuncUnit> units> {
  int Cycles          = cycles;       // length of stage in machine cycles
  list<FuncUnit> Units = units;       // choice of functional units
}

//===----------------------------------------------------------------------===//
// Instruction itinerary - An itinerary represents a sequential series of steps
// required to complete an instruction.  Itineraries are represented as lists of
// instruction stages.
//

//===----------------------------------------------------------------------===//
// Instruction itinerary classes - These values represent 'named' instruction
// itinerary.  Using named itineraries simplifies managing groups of
// instructions across chip sets.  An instruction uses the same itinerary class
// across all chip sets.  Thus a new chip set can be added without modifying
// instruction information.
//
class InstrItinClass;
def NoItinerary : InstrItinClass;

//===----------------------------------------------------------------------===//
// Instruction itinerary data - These values provide a runtime map of an 
// instruction itinerary class (name) to it's itinerary data.
//
class InstrItinData<InstrItinClass Class, list<InstrStage> stages> {
  InstrItinClass TheClass = Class;
  list<InstrStage> Stages = stages;
}

//===----------------------------------------------------------------------===//
// Processor itineraries - These values represent the set of all itinerary
// classes for a given chip set.
//
class ProcessorItineraries<list<InstrItinData> iid> {
  list<InstrItinData> IID = iid;
}

// NoItineraries - A marker that can be used by processors without schedule
// info.
def NoItineraries : ProcessorItineraries<[]>;

