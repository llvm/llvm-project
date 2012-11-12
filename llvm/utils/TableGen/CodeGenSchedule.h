//===- CodeGenSchedule.h - Scheduling Machine Models ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate the machine model as decribed in
// the target description.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_SCHEDULE_H
#define CODEGEN_SCHEDULE_H

#include "SetTheory.h"
#include "llvm/TableGen/Record.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {

class CodeGenTarget;
class CodeGenSchedModels;
class CodeGenInstruction;

typedef std::vector<Record*> RecVec;
typedef std::vector<Record*>::const_iterator RecIter;

typedef std::vector<unsigned> IdxVec;
typedef std::vector<unsigned>::const_iterator IdxIter;

void splitSchedReadWrites(const RecVec &RWDefs,
                          RecVec &WriteDefs, RecVec &ReadDefs);

/// We have two kinds of SchedReadWrites. Explicitly defined and inferred
/// sequences.  TheDef is nonnull for explicit SchedWrites, but Sequence may or
/// may not be empty. TheDef is null for inferred sequences, and Sequence must
/// be nonempty.
///
/// IsVariadic controls whether the variants are expanded into multiple operands
/// or a sequence of writes on one operand.
struct CodeGenSchedRW {
  unsigned Index;
  std::string Name;
  Record *TheDef;
  bool IsRead;
  bool IsAlias;
  bool HasVariants;
  bool IsVariadic;
  bool IsSequence;
  IdxVec Sequence;
  RecVec Aliases;

  CodeGenSchedRW(): Index(0), TheDef(0), IsAlias(false), HasVariants(false),
                    IsVariadic(false), IsSequence(false) {}
  CodeGenSchedRW(unsigned Idx, Record *Def): Index(Idx), TheDef(Def),
                                             IsAlias(false), IsVariadic(false) {
    Name = Def->getName();
    IsRead = Def->isSubClassOf("SchedRead");
    HasVariants = Def->isSubClassOf("SchedVariant");
    if (HasVariants)
      IsVariadic = Def->getValueAsBit("Variadic");

    // Read records don't currently have sequences, but it can be easily
    // added. Note that implicit Reads (from ReadVariant) may have a Sequence
    // (but no record).
    IsSequence = Def->isSubClassOf("WriteSequence");
  }

  CodeGenSchedRW(unsigned Idx, bool Read, const IdxVec &Seq,
                 const std::string &Name):
    Index(Idx), Name(Name), TheDef(0), IsRead(Read), IsAlias(false),
    HasVariants(false), IsVariadic(false), IsSequence(true), Sequence(Seq) {
    assert(Sequence.size() > 1 && "implied sequence needs >1 RWs");
  }

  bool isValid() const {
    assert((!HasVariants || TheDef) && "Variant write needs record def");
    assert((!IsVariadic || HasVariants) && "Variadic write needs variants");
    assert((!IsSequence || !HasVariants) && "Sequence can't have variant");
    assert((!IsSequence || !Sequence.empty()) && "Sequence should be nonempty");
    assert((!IsAlias || Aliases.empty()) && "Alias cannot have aliases");
    return TheDef || !Sequence.empty();
  }

#ifndef NDEBUG
  void dump() const;
#endif
};

/// Represent a transition between SchedClasses induced by SchedVariant.
struct CodeGenSchedTransition {
  unsigned ToClassIdx;
  IdxVec ProcIndices;
  RecVec PredTerm;
};

/// Scheduling class.
///
/// Each instruction description will be mapped to a scheduling class. There are
/// four types of classes:
///
/// 1) An explicitly defined itinerary class with ItinClassDef set.
/// Writes and ReadDefs are empty. ProcIndices contains 0 for any processor.
///
/// 2) An implied class with a list of SchedWrites and SchedReads that are
/// defined in an instruction definition and which are common across all
/// subtargets. ProcIndices contains 0 for any processor.
///
/// 3) An implied class with a list of InstRW records that map instructions to
/// SchedWrites and SchedReads per-processor. InstrClassMap should map the same
/// instructions to this class. ProcIndices contains all the processors that
/// provided InstrRW records for this class. ItinClassDef or Writes/Reads may
/// still be defined for processors with no InstRW entry.
///
/// 4) An inferred class represents a variant of another class that may be
/// resolved at runtime. ProcIndices contains the set of processors that may
/// require the class. ProcIndices are propagated through SchedClasses as
/// variants are expanded. Multiple SchedClasses may be inferred from an
/// itinerary class. Each inherits the processor index from the ItinRW record
/// that mapped the itinerary class to the variant Writes or Reads.
struct CodeGenSchedClass {
  std::string Name;
  Record *ItinClassDef;

  IdxVec Writes;
  IdxVec Reads;
  // Sorted list of ProcIdx, where ProcIdx==0 implies any processor.
  IdxVec ProcIndices;

  std::vector<CodeGenSchedTransition> Transitions;

  // InstRW records associated with this class. These records may refer to an
  // Instruction no longer mapped to this class by InstrClassMap. These
  // Instructions should be ignored by this class because they have been split
  // off to join another inferred class.
  RecVec InstRWs;

  CodeGenSchedClass(): ItinClassDef(0) {}
  CodeGenSchedClass(Record *rec): ItinClassDef(rec) {
    Name = rec->getName();
    ProcIndices.push_back(0);
  }

#ifndef NDEBUG
  void dump(const CodeGenSchedModels *SchedModels) const;
#endif
};

// Processor model.
//
// ModelName is a unique name used to name an instantiation of MCSchedModel.
//
// ModelDef is NULL for inferred Models. This happens when a processor defines
// an itinerary but no machine model. If the processer defines neither a machine
// model nor itinerary, then ModelDef remains pointing to NoModel. NoModel has
// the special "NoModel" field set to true.
//
// ItinsDef always points to a valid record definition, but may point to the
// default NoItineraries. NoItineraries has an empty list of InstrItinData
// records.
//
// ItinDefList orders this processor's InstrItinData records by SchedClass idx.
struct CodeGenProcModel {
  unsigned Index;
  std::string ModelName;
  Record *ModelDef;
  Record *ItinsDef;

  // Derived members...

  // Array of InstrItinData records indexed by a CodeGenSchedClass index.
  // This list is empty if the Processor has no value for Itineraries.
  // Initialized by collectProcItins().
  RecVec ItinDefList;

  // Map itinerary classes to per-operand resources.
  // This list is empty if no ItinRW refers to this Processor.
  RecVec ItinRWDefs;

  // All read/write resources associated with this processor.
  RecVec WriteResDefs;
  RecVec ReadAdvanceDefs;

  // Per-operand machine model resources associated with this processor.
  RecVec ProcResourceDefs;

  CodeGenProcModel(unsigned Idx, const std::string &Name, Record *MDef,
                   Record *IDef) :
    Index(Idx), ModelName(Name), ModelDef(MDef), ItinsDef(IDef) {}

  bool hasInstrSchedModel() const {
    return !WriteResDefs.empty() || !ItinRWDefs.empty();
  }

  unsigned getProcResourceIdx(Record *PRDef) const;

#ifndef NDEBUG
  void dump() const;
#endif
};

/// Top level container for machine model data.
class CodeGenSchedModels {
  RecordKeeper &Records;
  const CodeGenTarget &Target;

  // Map dag expressions to Instruction lists.
  SetTheory Sets;

  // List of unique processor models.
  std::vector<CodeGenProcModel> ProcModels;

  // Map Processor's MachineModel or ProcItin to a CodeGenProcModel index.
  typedef DenseMap<Record*, unsigned> ProcModelMapTy;
  ProcModelMapTy ProcModelMap;

  // Per-operand SchedReadWrite types.
  std::vector<CodeGenSchedRW> SchedWrites;
  std::vector<CodeGenSchedRW> SchedReads;

  // List of unique SchedClasses.
  std::vector<CodeGenSchedClass> SchedClasses;

  // Map SchedClass name to itinerary index.
  // These are either explicit itinerary classes or classes implied by
  // instruction definitions with SchedReadWrite lists.
  StringMap<unsigned> SchedClassIdxMap;

  // SchedClass indices 1 up to and including NumItineraryClasses identify
  // itinerary classes that are explicitly used for this target's instruction
  // definitions. NoItinerary always has index 0 regardless of whether it is
  // explicitly referenced.
  //
  // Any implied SchedClass has an index greater than NumItineraryClasses.
  unsigned NumItineraryClasses;

  // Any inferred SchedClass has an index greater than NumInstrSchedClassses.
  unsigned NumInstrSchedClasses;

  // Map Instruction to SchedClass index. Only for Instructions mentioned in
  // InstRW records.
  typedef DenseMap<Record*, unsigned> InstClassMapTy;
  InstClassMapTy InstrClassMap;

public:
  CodeGenSchedModels(RecordKeeper& RK, const CodeGenTarget &TGT);

  Record *getModelOrItinDef(Record *ProcDef) const {
    Record *ModelDef = ProcDef->getValueAsDef("SchedModel");
    Record *ItinsDef = ProcDef->getValueAsDef("ProcItin");
    if (!ItinsDef->getValueAsListOfDefs("IID").empty()) {
      assert(ModelDef->getValueAsBit("NoModel")
             && "Itineraries must be defined within SchedMachineModel");
      return ItinsDef;
    }
    return ModelDef;
  }

  const CodeGenProcModel &getModelForProc(Record *ProcDef) const {
    Record *ModelDef = getModelOrItinDef(ProcDef);
    ProcModelMapTy::const_iterator I = ProcModelMap.find(ModelDef);
    assert(I != ProcModelMap.end() && "missing machine model");
    return ProcModels[I->second];
  }

  const CodeGenProcModel &getProcModel(Record *ModelDef) const {
    ProcModelMapTy::const_iterator I = ProcModelMap.find(ModelDef);
    assert(I != ProcModelMap.end() && "missing machine model");
    return ProcModels[I->second];
  }

  // Iterate over the unique processor models.
  typedef std::vector<CodeGenProcModel>::const_iterator ProcIter;
  ProcIter procModelBegin() const { return ProcModels.begin(); }
  ProcIter procModelEnd() const { return ProcModels.end(); }

  // Get a SchedWrite from its index.
  const CodeGenSchedRW &getSchedWrite(unsigned Idx) const {
    assert(Idx < SchedWrites.size() && "bad SchedWrite index");
    assert(SchedWrites[Idx].isValid() && "invalid SchedWrite");
    return SchedWrites[Idx];
  }
  // Get a SchedWrite from its index.
  const CodeGenSchedRW &getSchedRead(unsigned Idx) const {
    assert(Idx < SchedReads.size() && "bad SchedRead index");
    assert(SchedReads[Idx].isValid() && "invalid SchedRead");
    return SchedReads[Idx];
  }

  const CodeGenSchedRW &getSchedRW(unsigned Idx, bool IsRead) const {
    return IsRead ? getSchedRead(Idx) : getSchedWrite(Idx);
  }
  CodeGenSchedRW &getSchedRW(Record *Def) {
    bool IsRead = Def->isSubClassOf("SchedRead");
    unsigned Idx = getSchedRWIdx(Def, IsRead);
    return const_cast<CodeGenSchedRW&>(
      IsRead ? getSchedRead(Idx) : getSchedWrite(Idx));
  }
  const CodeGenSchedRW &getSchedRW(Record*Def) const {
    return const_cast<CodeGenSchedModels&>(*this).getSchedRW(Def);
  }

  unsigned getSchedRWIdx(Record *Def, bool IsRead, unsigned After = 0) const;

  // Return true if the given write record is referenced by a ReadAdvance.
  bool hasReadOfWrite(Record *WriteDef) const;

  // Check if any instructions are assigned to an explicit itinerary class other
  // than NoItinerary.
  bool hasItineraryClasses() const { return NumItineraryClasses > 0; }

  // Return the number of itinerary classes in use by this target's instruction
  // descriptions, not including "NoItinerary".
  unsigned numItineraryClasses() const {
    return NumItineraryClasses;
  }

  // Get a SchedClass from its index.
  CodeGenSchedClass &getSchedClass(unsigned Idx) {
    assert(Idx < SchedClasses.size() && "bad SchedClass index");
    return SchedClasses[Idx];
  }
  const CodeGenSchedClass &getSchedClass(unsigned Idx) const {
    assert(Idx < SchedClasses.size() && "bad SchedClass index");
    return SchedClasses[Idx];
  }

  // Get the SchedClass index for an instruction. Instructions with no
  // itinerary, no SchedReadWrites, and no InstrReadWrites references return 0
  // for NoItinerary.
  unsigned getSchedClassIdx(const CodeGenInstruction &Inst) const;

  unsigned getSchedClassIdx(const RecVec &RWDefs) const;

  unsigned getSchedClassIdxForItin(const Record *ItinDef) {
    return SchedClassIdxMap[ItinDef->getName()];
  }

  typedef std::vector<CodeGenSchedClass>::const_iterator SchedClassIter;
  SchedClassIter schedClassBegin() const { return SchedClasses.begin(); }
  SchedClassIter schedClassEnd() const { return SchedClasses.end(); }

  void findRWs(const RecVec &RWDefs, IdxVec &Writes, IdxVec &Reads) const;
  void findRWs(const RecVec &RWDefs, IdxVec &RWs, bool IsRead) const;
  void expandRWSequence(unsigned RWIdx, IdxVec &RWSeq, bool IsRead) const;
  void expandRWSeqForProc(unsigned RWIdx, IdxVec &RWSeq, bool IsRead,
                          const CodeGenProcModel &ProcModel) const;

  unsigned addSchedClass(const IdxVec &OperWrites, const IdxVec &OperReads,
                         const IdxVec &ProcIndices);

  unsigned findOrInsertRW(ArrayRef<unsigned> Seq, bool IsRead);

  unsigned findSchedClassIdx(const IdxVec &Writes, const IdxVec &Reads) const;

  Record *findProcResUnits(Record *ProcResKind,
                           const CodeGenProcModel &PM) const;

private:
  void collectProcModels();

  // Initialize a new processor model if it is unique.
  void addProcModel(Record *ProcDef);

  void collectSchedRW();

  std::string genRWName(const IdxVec& Seq, bool IsRead);
  unsigned findRWForSequence(const IdxVec &Seq, bool IsRead);

  void collectSchedClasses();

  std::string createSchedClassName(const IdxVec &OperWrites,
                                   const IdxVec &OperReads);
  std::string createSchedClassName(const RecVec &InstDefs);
  void createInstRWClass(Record *InstRWDef);

  void collectProcItins();

  void collectProcItinRW();

  void inferSchedClasses();

  void inferFromRW(const IdxVec &OperWrites, const IdxVec &OperReads,
                   unsigned FromClassIdx, const IdxVec &ProcIndices);
  void inferFromItinClass(Record *ItinClassDef, unsigned FromClassIdx);
  void inferFromInstRWs(unsigned SCIdx);

  void collectProcResources();

  void collectItinProcResources(Record *ItinClassDef);

  void collectRWResources(unsigned RWIdx, bool IsRead,
                          const IdxVec &ProcIndices);

  void collectRWResources(const IdxVec &Writes, const IdxVec &Reads,
                          const IdxVec &ProcIndices);

  void addProcResource(Record *ProcResourceKind, CodeGenProcModel &PM);

  void addWriteRes(Record *ProcWriteResDef, unsigned PIdx);

  void addReadAdvance(Record *ProcReadAdvanceDef, unsigned PIdx);
};

} // namespace llvm

#endif
