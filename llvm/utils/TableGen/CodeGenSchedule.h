//===- CodeGenSchedule.h - Scheduling Machine Models ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate the machine model as described in
// the target description.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_CODEGENSCHEDULE_H
#define LLVM_UTILS_TABLEGEN_CODEGENSCHEDULE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"

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

  CodeGenSchedRW()
    : Index(0), TheDef(nullptr), IsRead(false), IsAlias(false),
      HasVariants(false), IsVariadic(false), IsSequence(false) {}
  CodeGenSchedRW(unsigned Idx, Record *Def)
    : Index(Idx), TheDef(Def), IsAlias(false), IsVariadic(false) {
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

  CodeGenSchedRW(unsigned Idx, bool Read, ArrayRef<unsigned> Seq,
                 const std::string &Name)
      : Index(Idx), Name(Name), TheDef(nullptr), IsRead(Read), IsAlias(false),
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
  unsigned Index;
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

  CodeGenSchedClass(): Index(0), ItinClassDef(nullptr) {}

  bool isKeyEqual(Record *IC, ArrayRef<unsigned> W, ArrayRef<unsigned> R) {
    return ItinClassDef == IC && makeArrayRef(Writes) == W &&
           makeArrayRef(Reads) == R;
  }

  // Is this class generated from a variants if existing classes? Instructions
  // are never mapped directly to inferred scheduling classes.
  bool isInferred() const { return !ItinClassDef; }

#ifndef NDEBUG
  void dump(const CodeGenSchedModels *SchedModels) const;
#endif
};

// Processor model.
//
// ModelName is a unique name used to name an instantiation of MCSchedModel.
//
// ModelDef is NULL for inferred Models. This happens when a processor defines
// an itinerary but no machine model. If the processor defines neither a machine
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
  RecVec ProcResGroupDefs;

  CodeGenProcModel(unsigned Idx, const std::string &Name, Record *MDef,
                   Record *IDef) :
    Index(Idx), ModelName(Name), ModelDef(MDef), ItinsDef(IDef) {}

  bool hasItineraries() const {
    return !ItinsDef->getValueAsListOfDefs("IID").empty();
  }

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

  // Any inferred SchedClass has an index greater than NumInstrSchedClassses.
  unsigned NumInstrSchedClasses;

  // Map each instruction to its unique SchedClass index considering the
  // combination of it's itinerary class, SchedRW list, and InstRW records.
  typedef DenseMap<Record*, unsigned> InstClassMapTy;
  InstClassMapTy InstrClassMap;

public:
  CodeGenSchedModels(RecordKeeper& RK, const CodeGenTarget &TGT);

  // iterator access to the scheduling classes.
  typedef std::vector<CodeGenSchedClass>::iterator class_iterator;
  typedef std::vector<CodeGenSchedClass>::const_iterator const_class_iterator;
  class_iterator classes_begin() { return SchedClasses.begin(); }
  const_class_iterator classes_begin() const { return SchedClasses.begin(); }
  class_iterator classes_end() { return SchedClasses.end(); }
  const_class_iterator classes_end() const { return SchedClasses.end(); }
  iterator_range<class_iterator> classes() {
   return make_range(classes_begin(), classes_end());
  }
  iterator_range<const_class_iterator> classes() const {
   return make_range(classes_begin(), classes_end());
  }
  iterator_range<class_iterator> explicit_classes() {
    return make_range(classes_begin(), classes_begin() + NumInstrSchedClasses);
  }
  iterator_range<const_class_iterator> explicit_classes() const {
    return make_range(classes_begin(), classes_begin() + NumInstrSchedClasses);
  }

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

  CodeGenProcModel &getProcModel(Record *ModelDef) {
    ProcModelMapTy::const_iterator I = ProcModelMap.find(ModelDef);
    assert(I != ProcModelMap.end() && "missing machine model");
    return ProcModels[I->second];
  }
  const CodeGenProcModel &getProcModel(Record *ModelDef) const {
    return const_cast<CodeGenSchedModels*>(this)->getProcModel(ModelDef);
  }

  // Iterate over the unique processor models.
  typedef std::vector<CodeGenProcModel>::const_iterator ProcIter;
  ProcIter procModelBegin() const { return ProcModels.begin(); }
  ProcIter procModelEnd() const { return ProcModels.end(); }
  ArrayRef<CodeGenProcModel> procModels() const { return ProcModels; }

  // Return true if any processors have itineraries.
  bool hasItineraries() const;

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

  typedef std::vector<CodeGenSchedClass>::const_iterator SchedClassIter;
  SchedClassIter schedClassBegin() const { return SchedClasses.begin(); }
  SchedClassIter schedClassEnd() const { return SchedClasses.end(); }
  ArrayRef<CodeGenSchedClass> schedClasses() const { return SchedClasses; }

  unsigned numInstrSchedClasses() const { return NumInstrSchedClasses; }

  void findRWs(const RecVec &RWDefs, IdxVec &Writes, IdxVec &Reads) const;
  void findRWs(const RecVec &RWDefs, IdxVec &RWs, bool IsRead) const;
  void expandRWSequence(unsigned RWIdx, IdxVec &RWSeq, bool IsRead) const;
  void expandRWSeqForProc(unsigned RWIdx, IdxVec &RWSeq, bool IsRead,
                          const CodeGenProcModel &ProcModel) const;

  unsigned addSchedClass(Record *ItinDef, ArrayRef<unsigned> OperWrites,
                         ArrayRef<unsigned> OperReads,
                         ArrayRef<unsigned> ProcIndices);

  unsigned findOrInsertRW(ArrayRef<unsigned> Seq, bool IsRead);

  unsigned findSchedClassIdx(Record *ItinClassDef, ArrayRef<unsigned> Writes,
                             ArrayRef<unsigned> Reads) const;

  Record *findProcResUnits(Record *ProcResKind,
                           const CodeGenProcModel &PM) const;

private:
  void collectProcModels();

  // Initialize a new processor model if it is unique.
  void addProcModel(Record *ProcDef);

  void collectSchedRW();

  std::string genRWName(ArrayRef<unsigned> Seq, bool IsRead);
  unsigned findRWForSequence(ArrayRef<unsigned> Seq, bool IsRead);

  void collectSchedClasses();

  std::string createSchedClassName(Record *ItinClassDef,
                                   ArrayRef<unsigned> OperWrites,
                                   ArrayRef<unsigned> OperReads);
  std::string createSchedClassName(const RecVec &InstDefs);
  void createInstRWClass(Record *InstRWDef);

  void collectProcItins();

  void collectProcItinRW();

  void inferSchedClasses();

  void inferFromRW(ArrayRef<unsigned> OperWrites, ArrayRef<unsigned> OperReads,
                   unsigned FromClassIdx, ArrayRef<unsigned> ProcIndices);
  void inferFromItinClass(Record *ItinClassDef, unsigned FromClassIdx);
  void inferFromInstRWs(unsigned SCIdx);

  bool hasSuperGroup(RecVec &SubUnits, CodeGenProcModel &PM);
  void verifyProcResourceGroups(CodeGenProcModel &PM);

  void collectProcResources();

  void collectItinProcResources(Record *ItinClassDef);

  void collectRWResources(unsigned RWIdx, bool IsRead,
                          ArrayRef<unsigned> ProcIndices);

  void collectRWResources(ArrayRef<unsigned> Writes, ArrayRef<unsigned> Reads,
                          ArrayRef<unsigned> ProcIndices);

  void addProcResource(Record *ProcResourceKind, CodeGenProcModel &PM);

  void addWriteRes(Record *ProcWriteResDef, unsigned PIdx);

  void addReadAdvance(Record *ProcReadAdvanceDef, unsigned PIdx);
};

} // namespace llvm

#endif
