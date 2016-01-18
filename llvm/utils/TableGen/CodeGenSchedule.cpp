//===- CodeGenSchedule.cpp - Scheduling MachineModels ---------------------===//
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

#include "CodeGenSchedule.h"
#include "CodeGenTarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;

#define DEBUG_TYPE "subtarget-emitter"

#ifndef NDEBUG
static void dumpIdxVec(ArrayRef<unsigned> V) {
  for (unsigned Idx : V)
    dbgs() << Idx << ", ";
}
#endif

namespace {
// (instrs a, b, ...) Evaluate and union all arguments. Identical to AddOp.
struct InstrsOp : public SetTheory::Operator {
  void apply(SetTheory &ST, DagInit *Expr, SetTheory::RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    ST.evaluate(Expr->arg_begin(), Expr->arg_end(), Elts, Loc);
  }
};

// (instregex "OpcPat",...) Find all instructions matching an opcode pattern.
//
// TODO: Since this is a prefix match, perform a binary search over the
// instruction names using lower_bound. Note that the predefined instrs must be
// scanned linearly first. However, this is only safe if the regex pattern has
// no top-level bars. The DAG already has a list of patterns, so there's no
// reason to use top-level bars, but we need a way to verify they don't exist
// before implementing the optimization.
struct InstRegexOp : public SetTheory::Operator {
  const CodeGenTarget &Target;
  InstRegexOp(const CodeGenTarget &t): Target(t) {}

  void apply(SetTheory &ST, DagInit *Expr, SetTheory::RecSet &Elts,
             ArrayRef<SMLoc> Loc) override {
    SmallVector<Regex, 4> RegexList;
    for (DagInit::const_arg_iterator
           AI = Expr->arg_begin(), AE = Expr->arg_end(); AI != AE; ++AI) {
      StringInit *SI = dyn_cast<StringInit>(*AI);
      if (!SI)
        PrintFatalError(Loc, "instregex requires pattern string: "
          + Expr->getAsString());
      std::string pat = SI->getValue();
      // Implement a python-style prefix match.
      if (pat[0] != '^') {
        pat.insert(0, "^(");
        pat.insert(pat.end(), ')');
      }
      RegexList.push_back(Regex(pat));
    }
    for (const CodeGenInstruction *Inst : Target.instructions()) {
      for (auto &R : RegexList) {
        if (R.match(Inst->TheDef->getName()))
          Elts.insert(Inst->TheDef);
      }
    }
  }
};
} // end anonymous namespace

/// CodeGenModels ctor interprets machine model records and populates maps.
CodeGenSchedModels::CodeGenSchedModels(RecordKeeper &RK,
                                       const CodeGenTarget &TGT):
  Records(RK), Target(TGT) {

  Sets.addFieldExpander("InstRW", "Instrs");

  // Allow Set evaluation to recognize the dags used in InstRW records:
  // (instrs Op1, Op1...)
  Sets.addOperator("instrs", llvm::make_unique<InstrsOp>());
  Sets.addOperator("instregex", llvm::make_unique<InstRegexOp>(Target));

  // Instantiate a CodeGenProcModel for each SchedMachineModel with the values
  // that are explicitly referenced in tablegen records. Resources associated
  // with each processor will be derived later. Populate ProcModelMap with the
  // CodeGenProcModel instances.
  collectProcModels();

  // Instantiate a CodeGenSchedRW for each SchedReadWrite record explicitly
  // defined, and populate SchedReads and SchedWrites vectors. Implicit
  // SchedReadWrites that represent sequences derived from expanded variant will
  // be inferred later.
  collectSchedRW();

  // Instantiate a CodeGenSchedClass for each unique SchedRW signature directly
  // required by an instruction definition, and populate SchedClassIdxMap. Set
  // NumItineraryClasses to the number of explicit itinerary classes referenced
  // by instructions. Set NumInstrSchedClasses to the number of itinerary
  // classes plus any classes implied by instructions that derive from class
  // Sched and provide SchedRW list. This does not infer any new classes from
  // SchedVariant.
  collectSchedClasses();

  // Find instruction itineraries for each processor. Sort and populate
  // CodeGenProcModel::ItinDefList. (Cycle-to-cycle itineraries). This requires
  // all itinerary classes to be discovered.
  collectProcItins();

  // Find ItinRW records for each processor and itinerary class.
  // (For per-operand resources mapped to itinerary classes).
  collectProcItinRW();

  // Infer new SchedClasses from SchedVariant.
  inferSchedClasses();

  // Populate each CodeGenProcModel's WriteResDefs, ReadAdvanceDefs, and
  // ProcResourceDefs.
  collectProcResources();
}

/// Gather all processor models.
void CodeGenSchedModels::collectProcModels() {
  RecVec ProcRecords = Records.getAllDerivedDefinitions("Processor");
  std::sort(ProcRecords.begin(), ProcRecords.end(), LessRecordFieldName());

  // Reserve space because we can. Reallocation would be ok.
  ProcModels.reserve(ProcRecords.size()+1);

  // Use idx=0 for NoModel/NoItineraries.
  Record *NoModelDef = Records.getDef("NoSchedModel");
  Record *NoItinsDef = Records.getDef("NoItineraries");
  ProcModels.emplace_back(0, "NoSchedModel", NoModelDef, NoItinsDef);
  ProcModelMap[NoModelDef] = 0;

  // For each processor, find a unique machine model.
  for (unsigned i = 0, N = ProcRecords.size(); i < N; ++i)
    addProcModel(ProcRecords[i]);
}

/// Get a unique processor model based on the defined MachineModel and
/// ProcessorItineraries.
void CodeGenSchedModels::addProcModel(Record *ProcDef) {
  Record *ModelKey = getModelOrItinDef(ProcDef);
  if (!ProcModelMap.insert(std::make_pair(ModelKey, ProcModels.size())).second)
    return;

  std::string Name = ModelKey->getName();
  if (ModelKey->isSubClassOf("SchedMachineModel")) {
    Record *ItinsDef = ModelKey->getValueAsDef("Itineraries");
    ProcModels.emplace_back(ProcModels.size(), Name, ModelKey, ItinsDef);
  }
  else {
    // An itinerary is defined without a machine model. Infer a new model.
    if (!ModelKey->getValueAsListOfDefs("IID").empty())
      Name = Name + "Model";
    ProcModels.emplace_back(ProcModels.size(), Name,
                            ProcDef->getValueAsDef("SchedModel"), ModelKey);
  }
  DEBUG(ProcModels.back().dump());
}

// Recursively find all reachable SchedReadWrite records.
static void scanSchedRW(Record *RWDef, RecVec &RWDefs,
                        SmallPtrSet<Record*, 16> &RWSet) {
  if (!RWSet.insert(RWDef).second)
    return;
  RWDefs.push_back(RWDef);
  // Reads don't current have sequence records, but it can be added later.
  if (RWDef->isSubClassOf("WriteSequence")) {
    RecVec Seq = RWDef->getValueAsListOfDefs("Writes");
    for (RecIter I = Seq.begin(), E = Seq.end(); I != E; ++I)
      scanSchedRW(*I, RWDefs, RWSet);
  }
  else if (RWDef->isSubClassOf("SchedVariant")) {
    // Visit each variant (guarded by a different predicate).
    RecVec Vars = RWDef->getValueAsListOfDefs("Variants");
    for (RecIter VI = Vars.begin(), VE = Vars.end(); VI != VE; ++VI) {
      // Visit each RW in the sequence selected by the current variant.
      RecVec Selected = (*VI)->getValueAsListOfDefs("Selected");
      for (RecIter I = Selected.begin(), E = Selected.end(); I != E; ++I)
        scanSchedRW(*I, RWDefs, RWSet);
    }
  }
}

// Collect and sort all SchedReadWrites reachable via tablegen records.
// More may be inferred later when inferring new SchedClasses from variants.
void CodeGenSchedModels::collectSchedRW() {
  // Reserve idx=0 for invalid writes/reads.
  SchedWrites.resize(1);
  SchedReads.resize(1);

  SmallPtrSet<Record*, 16> RWSet;

  // Find all SchedReadWrites referenced by instruction defs.
  RecVec SWDefs, SRDefs;
  for (const CodeGenInstruction *Inst : Target.instructions()) {
    Record *SchedDef = Inst->TheDef;
    if (SchedDef->isValueUnset("SchedRW"))
      continue;
    RecVec RWs = SchedDef->getValueAsListOfDefs("SchedRW");
    for (RecIter RWI = RWs.begin(), RWE = RWs.end(); RWI != RWE; ++RWI) {
      if ((*RWI)->isSubClassOf("SchedWrite"))
        scanSchedRW(*RWI, SWDefs, RWSet);
      else {
        assert((*RWI)->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
        scanSchedRW(*RWI, SRDefs, RWSet);
      }
    }
  }
  // Find all ReadWrites referenced by InstRW.
  RecVec InstRWDefs = Records.getAllDerivedDefinitions("InstRW");
  for (RecIter OI = InstRWDefs.begin(), OE = InstRWDefs.end(); OI != OE; ++OI) {
    // For all OperandReadWrites.
    RecVec RWDefs = (*OI)->getValueAsListOfDefs("OperandReadWrites");
    for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end();
         RWI != RWE; ++RWI) {
      if ((*RWI)->isSubClassOf("SchedWrite"))
        scanSchedRW(*RWI, SWDefs, RWSet);
      else {
        assert((*RWI)->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
        scanSchedRW(*RWI, SRDefs, RWSet);
      }
    }
  }
  // Find all ReadWrites referenced by ItinRW.
  RecVec ItinRWDefs = Records.getAllDerivedDefinitions("ItinRW");
  for (RecIter II = ItinRWDefs.begin(), IE = ItinRWDefs.end(); II != IE; ++II) {
    // For all OperandReadWrites.
    RecVec RWDefs = (*II)->getValueAsListOfDefs("OperandReadWrites");
    for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end();
         RWI != RWE; ++RWI) {
      if ((*RWI)->isSubClassOf("SchedWrite"))
        scanSchedRW(*RWI, SWDefs, RWSet);
      else {
        assert((*RWI)->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
        scanSchedRW(*RWI, SRDefs, RWSet);
      }
    }
  }
  // Find all ReadWrites referenced by SchedAlias. AliasDefs needs to be sorted
  // for the loop below that initializes Alias vectors.
  RecVec AliasDefs = Records.getAllDerivedDefinitions("SchedAlias");
  std::sort(AliasDefs.begin(), AliasDefs.end(), LessRecord());
  for (RecIter AI = AliasDefs.begin(), AE = AliasDefs.end(); AI != AE; ++AI) {
    Record *MatchDef = (*AI)->getValueAsDef("MatchRW");
    Record *AliasDef = (*AI)->getValueAsDef("AliasRW");
    if (MatchDef->isSubClassOf("SchedWrite")) {
      if (!AliasDef->isSubClassOf("SchedWrite"))
        PrintFatalError((*AI)->getLoc(), "SchedWrite Alias must be SchedWrite");
      scanSchedRW(AliasDef, SWDefs, RWSet);
    }
    else {
      assert(MatchDef->isSubClassOf("SchedRead") && "Unknown SchedReadWrite");
      if (!AliasDef->isSubClassOf("SchedRead"))
        PrintFatalError((*AI)->getLoc(), "SchedRead Alias must be SchedRead");
      scanSchedRW(AliasDef, SRDefs, RWSet);
    }
  }
  // Sort and add the SchedReadWrites directly referenced by instructions or
  // itinerary resources. Index reads and writes in separate domains.
  std::sort(SWDefs.begin(), SWDefs.end(), LessRecord());
  for (RecIter SWI = SWDefs.begin(), SWE = SWDefs.end(); SWI != SWE; ++SWI) {
    assert(!getSchedRWIdx(*SWI, /*IsRead=*/false) && "duplicate SchedWrite");
    SchedWrites.emplace_back(SchedWrites.size(), *SWI);
  }
  std::sort(SRDefs.begin(), SRDefs.end(), LessRecord());
  for (RecIter SRI = SRDefs.begin(), SRE = SRDefs.end(); SRI != SRE; ++SRI) {
    assert(!getSchedRWIdx(*SRI, /*IsRead-*/true) && "duplicate SchedWrite");
    SchedReads.emplace_back(SchedReads.size(), *SRI);
  }
  // Initialize WriteSequence vectors.
  for (std::vector<CodeGenSchedRW>::iterator WI = SchedWrites.begin(),
         WE = SchedWrites.end(); WI != WE; ++WI) {
    if (!WI->IsSequence)
      continue;
    findRWs(WI->TheDef->getValueAsListOfDefs("Writes"), WI->Sequence,
            /*IsRead=*/false);
  }
  // Initialize Aliases vectors.
  for (RecIter AI = AliasDefs.begin(), AE = AliasDefs.end(); AI != AE; ++AI) {
    Record *AliasDef = (*AI)->getValueAsDef("AliasRW");
    getSchedRW(AliasDef).IsAlias = true;
    Record *MatchDef = (*AI)->getValueAsDef("MatchRW");
    CodeGenSchedRW &RW = getSchedRW(MatchDef);
    if (RW.IsAlias)
      PrintFatalError((*AI)->getLoc(), "Cannot Alias an Alias");
    RW.Aliases.push_back(*AI);
  }
  DEBUG(
    for (unsigned WIdx = 0, WEnd = SchedWrites.size(); WIdx != WEnd; ++WIdx) {
      dbgs() << WIdx << ": ";
      SchedWrites[WIdx].dump();
      dbgs() << '\n';
    }
    for (unsigned RIdx = 0, REnd = SchedReads.size(); RIdx != REnd; ++RIdx) {
      dbgs() << RIdx << ": ";
      SchedReads[RIdx].dump();
      dbgs() << '\n';
    }
    RecVec RWDefs = Records.getAllDerivedDefinitions("SchedReadWrite");
    for (RecIter RI = RWDefs.begin(), RE = RWDefs.end();
         RI != RE; ++RI) {
      if (!getSchedRWIdx(*RI, (*RI)->isSubClassOf("SchedRead"))) {
        const std::string &Name = (*RI)->getName();
        if (Name != "NoWrite" && Name != "ReadDefault")
          dbgs() << "Unused SchedReadWrite " << (*RI)->getName() << '\n';
      }
    });
}

/// Compute a SchedWrite name from a sequence of writes.
std::string CodeGenSchedModels::genRWName(ArrayRef<unsigned> Seq, bool IsRead) {
  std::string Name("(");
  for (auto I = Seq.begin(), E = Seq.end(); I != E; ++I) {
    if (I != Seq.begin())
      Name += '_';
    Name += getSchedRW(*I, IsRead).Name;
  }
  Name += ')';
  return Name;
}

unsigned CodeGenSchedModels::getSchedRWIdx(Record *Def, bool IsRead,
                                           unsigned After) const {
  const std::vector<CodeGenSchedRW> &RWVec = IsRead ? SchedReads : SchedWrites;
  assert(After < RWVec.size() && "start position out of bounds");
  for (std::vector<CodeGenSchedRW>::const_iterator I = RWVec.begin() + After,
         E = RWVec.end(); I != E; ++I) {
    if (I->TheDef == Def)
      return I - RWVec.begin();
  }
  return 0;
}

bool CodeGenSchedModels::hasReadOfWrite(Record *WriteDef) const {
  for (unsigned i = 0, e = SchedReads.size(); i < e; ++i) {
    Record *ReadDef = SchedReads[i].TheDef;
    if (!ReadDef || !ReadDef->isSubClassOf("ProcReadAdvance"))
      continue;

    RecVec ValidWrites = ReadDef->getValueAsListOfDefs("ValidWrites");
    if (std::find(ValidWrites.begin(), ValidWrites.end(), WriteDef)
        != ValidWrites.end()) {
      return true;
    }
  }
  return false;
}

namespace llvm {
void splitSchedReadWrites(const RecVec &RWDefs,
                          RecVec &WriteDefs, RecVec &ReadDefs) {
  for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end(); RWI != RWE; ++RWI) {
    if ((*RWI)->isSubClassOf("SchedWrite"))
      WriteDefs.push_back(*RWI);
    else {
      assert((*RWI)->isSubClassOf("SchedRead") && "unknown SchedReadWrite");
      ReadDefs.push_back(*RWI);
    }
  }
}
} // namespace llvm

// Split the SchedReadWrites defs and call findRWs for each list.
void CodeGenSchedModels::findRWs(const RecVec &RWDefs,
                                 IdxVec &Writes, IdxVec &Reads) const {
    RecVec WriteDefs;
    RecVec ReadDefs;
    splitSchedReadWrites(RWDefs, WriteDefs, ReadDefs);
    findRWs(WriteDefs, Writes, false);
    findRWs(ReadDefs, Reads, true);
}

// Call getSchedRWIdx for all elements in a sequence of SchedRW defs.
void CodeGenSchedModels::findRWs(const RecVec &RWDefs, IdxVec &RWs,
                                 bool IsRead) const {
  for (RecIter RI = RWDefs.begin(), RE = RWDefs.end(); RI != RE; ++RI) {
    unsigned Idx = getSchedRWIdx(*RI, IsRead);
    assert(Idx && "failed to collect SchedReadWrite");
    RWs.push_back(Idx);
  }
}

void CodeGenSchedModels::expandRWSequence(unsigned RWIdx, IdxVec &RWSeq,
                                          bool IsRead) const {
  const CodeGenSchedRW &SchedRW = getSchedRW(RWIdx, IsRead);
  if (!SchedRW.IsSequence) {
    RWSeq.push_back(RWIdx);
    return;
  }
  int Repeat =
    SchedRW.TheDef ? SchedRW.TheDef->getValueAsInt("Repeat") : 1;
  for (int i = 0; i < Repeat; ++i) {
    for (IdxIter I = SchedRW.Sequence.begin(), E = SchedRW.Sequence.end();
         I != E; ++I) {
      expandRWSequence(*I, RWSeq, IsRead);
    }
  }
}

// Expand a SchedWrite as a sequence following any aliases that coincide with
// the given processor model.
void CodeGenSchedModels::expandRWSeqForProc(
  unsigned RWIdx, IdxVec &RWSeq, bool IsRead,
  const CodeGenProcModel &ProcModel) const {

  const CodeGenSchedRW &SchedWrite = getSchedRW(RWIdx, IsRead);
  Record *AliasDef = nullptr;
  for (RecIter AI = SchedWrite.Aliases.begin(), AE = SchedWrite.Aliases.end();
       AI != AE; ++AI) {
    const CodeGenSchedRW &AliasRW = getSchedRW((*AI)->getValueAsDef("AliasRW"));
    if ((*AI)->getValueInit("SchedModel")->isComplete()) {
      Record *ModelDef = (*AI)->getValueAsDef("SchedModel");
      if (&getProcModel(ModelDef) != &ProcModel)
        continue;
    }
    if (AliasDef)
      PrintFatalError(AliasRW.TheDef->getLoc(), "Multiple aliases "
                      "defined for processor " + ProcModel.ModelName +
                      " Ensure only one SchedAlias exists per RW.");
    AliasDef = AliasRW.TheDef;
  }
  if (AliasDef) {
    expandRWSeqForProc(getSchedRWIdx(AliasDef, IsRead),
                       RWSeq, IsRead,ProcModel);
    return;
  }
  if (!SchedWrite.IsSequence) {
    RWSeq.push_back(RWIdx);
    return;
  }
  int Repeat =
    SchedWrite.TheDef ? SchedWrite.TheDef->getValueAsInt("Repeat") : 1;
  for (int i = 0; i < Repeat; ++i) {
    for (IdxIter I = SchedWrite.Sequence.begin(), E = SchedWrite.Sequence.end();
         I != E; ++I) {
      expandRWSeqForProc(*I, RWSeq, IsRead, ProcModel);
    }
  }
}

// Find the existing SchedWrite that models this sequence of writes.
unsigned CodeGenSchedModels::findRWForSequence(ArrayRef<unsigned> Seq,
                                               bool IsRead) {
  std::vector<CodeGenSchedRW> &RWVec = IsRead ? SchedReads : SchedWrites;

  for (std::vector<CodeGenSchedRW>::iterator I = RWVec.begin(), E = RWVec.end();
       I != E; ++I) {
    if (makeArrayRef(I->Sequence) == Seq)
      return I - RWVec.begin();
  }
  // Index zero reserved for invalid RW.
  return 0;
}

/// Add this ReadWrite if it doesn't already exist.
unsigned CodeGenSchedModels::findOrInsertRW(ArrayRef<unsigned> Seq,
                                            bool IsRead) {
  assert(!Seq.empty() && "cannot insert empty sequence");
  if (Seq.size() == 1)
    return Seq.back();

  unsigned Idx = findRWForSequence(Seq, IsRead);
  if (Idx)
    return Idx;

  unsigned RWIdx = IsRead ? SchedReads.size() : SchedWrites.size();
  CodeGenSchedRW SchedRW(RWIdx, IsRead, Seq, genRWName(Seq, IsRead));
  if (IsRead)
    SchedReads.push_back(SchedRW);
  else
    SchedWrites.push_back(SchedRW);
  return RWIdx;
}

/// Visit all the instruction definitions for this target to gather and
/// enumerate the itinerary classes. These are the explicitly specified
/// SchedClasses. More SchedClasses may be inferred.
void CodeGenSchedModels::collectSchedClasses() {

  // NoItinerary is always the first class at Idx=0
  SchedClasses.resize(1);
  SchedClasses.back().Index = 0;
  SchedClasses.back().Name = "NoInstrModel";
  SchedClasses.back().ItinClassDef = Records.getDef("NoItinerary");
  SchedClasses.back().ProcIndices.push_back(0);

  // Create a SchedClass for each unique combination of itinerary class and
  // SchedRW list.
  for (const CodeGenInstruction *Inst : Target.instructions()) {
    Record *ItinDef = Inst->TheDef->getValueAsDef("Itinerary");
    IdxVec Writes, Reads;
    if (!Inst->TheDef->isValueUnset("SchedRW"))
      findRWs(Inst->TheDef->getValueAsListOfDefs("SchedRW"), Writes, Reads);

    // ProcIdx == 0 indicates the class applies to all processors.
    IdxVec ProcIndices(1, 0);

    unsigned SCIdx = addSchedClass(ItinDef, Writes, Reads, ProcIndices);
    InstrClassMap[Inst->TheDef] = SCIdx;
  }
  // Create classes for InstRW defs.
  RecVec InstRWDefs = Records.getAllDerivedDefinitions("InstRW");
  std::sort(InstRWDefs.begin(), InstRWDefs.end(), LessRecord());
  for (RecIter OI = InstRWDefs.begin(), OE = InstRWDefs.end(); OI != OE; ++OI)
    createInstRWClass(*OI);

  NumInstrSchedClasses = SchedClasses.size();

  bool EnableDump = false;
  DEBUG(EnableDump = true);
  if (!EnableDump)
    return;

  for (const CodeGenInstruction *Inst : Target.instructions()) {
    std::string InstName = Inst->TheDef->getName();
    unsigned SCIdx = InstrClassMap.lookup(Inst->TheDef);
    if (!SCIdx) {
      dbgs() << "No machine model for " << Inst->TheDef->getName() << '\n';
      continue;
    }
    CodeGenSchedClass &SC = getSchedClass(SCIdx);
    if (SC.ProcIndices[0] != 0)
      PrintFatalError(Inst->TheDef->getLoc(), "Instruction's sched class "
                      "must not be subtarget specific.");

    IdxVec ProcIndices;
    if (SC.ItinClassDef->getName() != "NoItinerary") {
      ProcIndices.push_back(0);
      dbgs() << "Itinerary for " << InstName << ": "
             << SC.ItinClassDef->getName() << '\n';
    }
    if (!SC.Writes.empty()) {
      ProcIndices.push_back(0);
      dbgs() << "SchedRW machine model for " << InstName;
      for (IdxIter WI = SC.Writes.begin(), WE = SC.Writes.end(); WI != WE; ++WI)
        dbgs() << " " << SchedWrites[*WI].Name;
      for (IdxIter RI = SC.Reads.begin(), RE = SC.Reads.end(); RI != RE; ++RI)
        dbgs() << " " << SchedReads[*RI].Name;
      dbgs() << '\n';
    }
    const RecVec &RWDefs = SchedClasses[SCIdx].InstRWs;
    for (RecIter RWI = RWDefs.begin(), RWE = RWDefs.end();
         RWI != RWE; ++RWI) {
      const CodeGenProcModel &ProcModel =
        getProcModel((*RWI)->getValueAsDef("SchedModel"));
      ProcIndices.push_back(ProcModel.Index);
      dbgs() << "InstRW on " << ProcModel.ModelName << " for " << InstName;
      IdxVec Writes;
      IdxVec Reads;
      findRWs((*RWI)->getValueAsListOfDefs("OperandReadWrites"),
              Writes, Reads);
      for (IdxIter WI = Writes.begin(), WE = Writes.end(); WI != WE; ++WI)
        dbgs() << " " << SchedWrites[*WI].Name;
      for (IdxIter RI = Reads.begin(), RE = Reads.end(); RI != RE; ++RI)
        dbgs() << " " << SchedReads[*RI].Name;
      dbgs() << '\n';
    }
    for (std::vector<CodeGenProcModel>::iterator PI = ProcModels.begin(),
           PE = ProcModels.end(); PI != PE; ++PI) {
      if (!std::count(ProcIndices.begin(), ProcIndices.end(), PI->Index))
        dbgs() << "No machine model for " << Inst->TheDef->getName()
               << " on processor " << PI->ModelName << '\n';
    }
  }
}

/// Find an SchedClass that has been inferred from a per-operand list of
/// SchedWrites and SchedReads.
unsigned CodeGenSchedModels::findSchedClassIdx(Record *ItinClassDef,
                                               ArrayRef<unsigned> Writes,
                                               ArrayRef<unsigned> Reads) const {
  for (SchedClassIter I = schedClassBegin(), E = schedClassEnd(); I != E; ++I) {
    if (I->ItinClassDef == ItinClassDef && makeArrayRef(I->Writes) == Writes &&
        makeArrayRef(I->Reads) == Reads) {
      return I - schedClassBegin();
    }
  }
  return 0;
}

// Get the SchedClass index for an instruction.
unsigned CodeGenSchedModels::getSchedClassIdx(
  const CodeGenInstruction &Inst) const {

  return InstrClassMap.lookup(Inst.TheDef);
}

std::string
CodeGenSchedModels::createSchedClassName(Record *ItinClassDef,
                                         ArrayRef<unsigned> OperWrites,
                                         ArrayRef<unsigned> OperReads) {

  std::string Name;
  if (ItinClassDef && ItinClassDef->getName() != "NoItinerary")
    Name = ItinClassDef->getName();
  for (unsigned Idx : OperWrites) {
    if (!Name.empty())
      Name += '_';
    Name += SchedWrites[Idx].Name;
  }
  for (unsigned Idx : OperReads) {
    Name += '_';
    Name += SchedReads[Idx].Name;
  }
  return Name;
}

std::string CodeGenSchedModels::createSchedClassName(const RecVec &InstDefs) {

  std::string Name;
  for (RecIter I = InstDefs.begin(), E = InstDefs.end(); I != E; ++I) {
    if (I != InstDefs.begin())
      Name += '_';
    Name += (*I)->getName();
  }
  return Name;
}

/// Add an inferred sched class from an itinerary class and per-operand list of
/// SchedWrites and SchedReads. ProcIndices contains the set of IDs of
/// processors that may utilize this class.
unsigned CodeGenSchedModels::addSchedClass(Record *ItinClassDef,
                                           ArrayRef<unsigned> OperWrites,
                                           ArrayRef<unsigned> OperReads,
                                           ArrayRef<unsigned> ProcIndices) {
  assert(!ProcIndices.empty() && "expect at least one ProcIdx");

  unsigned Idx = findSchedClassIdx(ItinClassDef, OperWrites, OperReads);
  if (Idx || SchedClasses[0].isKeyEqual(ItinClassDef, OperWrites, OperReads)) {
    IdxVec PI;
    std::set_union(SchedClasses[Idx].ProcIndices.begin(),
                   SchedClasses[Idx].ProcIndices.end(),
                   ProcIndices.begin(), ProcIndices.end(),
                   std::back_inserter(PI));
    SchedClasses[Idx].ProcIndices.swap(PI);
    return Idx;
  }
  Idx = SchedClasses.size();
  SchedClasses.resize(Idx+1);
  CodeGenSchedClass &SC = SchedClasses.back();
  SC.Index = Idx;
  SC.Name = createSchedClassName(ItinClassDef, OperWrites, OperReads);
  SC.ItinClassDef = ItinClassDef;
  SC.Writes = OperWrites;
  SC.Reads = OperReads;
  SC.ProcIndices = ProcIndices;

  return Idx;
}

// Create classes for each set of opcodes that are in the same InstReadWrite
// definition across all processors.
void CodeGenSchedModels::createInstRWClass(Record *InstRWDef) {
  // ClassInstrs will hold an entry for each subset of Instrs in InstRWDef that
  // intersects with an existing class via a previous InstRWDef. Instrs that do
  // not intersect with an existing class refer back to their former class as
  // determined from ItinDef or SchedRW.
  SmallVector<std::pair<unsigned, SmallVector<Record *, 8> >, 4> ClassInstrs;
  // Sort Instrs into sets.
  const RecVec *InstDefs = Sets.expand(InstRWDef);
  if (InstDefs->empty())
    PrintFatalError(InstRWDef->getLoc(), "No matching instruction opcodes");

  for (RecIter I = InstDefs->begin(), E = InstDefs->end(); I != E; ++I) {
    InstClassMapTy::const_iterator Pos = InstrClassMap.find(*I);
    if (Pos == InstrClassMap.end())
      PrintFatalError((*I)->getLoc(), "No sched class for instruction.");
    unsigned SCIdx = Pos->second;
    unsigned CIdx = 0, CEnd = ClassInstrs.size();
    for (; CIdx != CEnd; ++CIdx) {
      if (ClassInstrs[CIdx].first == SCIdx)
        break;
    }
    if (CIdx == CEnd) {
      ClassInstrs.resize(CEnd + 1);
      ClassInstrs[CIdx].first = SCIdx;
    }
    ClassInstrs[CIdx].second.push_back(*I);
  }
  // For each set of Instrs, create a new class if necessary, and map or remap
  // the Instrs to it.
  unsigned CIdx = 0, CEnd = ClassInstrs.size();
  for (; CIdx != CEnd; ++CIdx) {
    unsigned OldSCIdx = ClassInstrs[CIdx].first;
    ArrayRef<Record*> InstDefs = ClassInstrs[CIdx].second;
    // If the all instrs in the current class are accounted for, then leave
    // them mapped to their old class.
    if (OldSCIdx) {
      const RecVec &RWDefs = SchedClasses[OldSCIdx].InstRWs;
      if (!RWDefs.empty()) {
        const RecVec *OrigInstDefs = Sets.expand(RWDefs[0]);
        unsigned OrigNumInstrs = 0;
        for (RecIter I = OrigInstDefs->begin(), E = OrigInstDefs->end();
             I != E; ++I) {
          if (InstrClassMap[*I] == OldSCIdx)
            ++OrigNumInstrs;
        }
        if (OrigNumInstrs == InstDefs.size()) {
          assert(SchedClasses[OldSCIdx].ProcIndices[0] == 0 &&
                 "expected a generic SchedClass");
          DEBUG(dbgs() << "InstRW: Reuse SC " << OldSCIdx << ":"
                << SchedClasses[OldSCIdx].Name << " on "
                << InstRWDef->getValueAsDef("SchedModel")->getName() << "\n");
          SchedClasses[OldSCIdx].InstRWs.push_back(InstRWDef);
          continue;
        }
      }
    }
    unsigned SCIdx = SchedClasses.size();
    SchedClasses.resize(SCIdx+1);
    CodeGenSchedClass &SC = SchedClasses.back();
    SC.Index = SCIdx;
    SC.Name = createSchedClassName(InstDefs);
    DEBUG(dbgs() << "InstRW: New SC " << SCIdx << ":" << SC.Name << " on "
          << InstRWDef->getValueAsDef("SchedModel")->getName() << "\n");

    // Preserve ItinDef and Writes/Reads for processors without an InstRW entry.
    SC.ItinClassDef = SchedClasses[OldSCIdx].ItinClassDef;
    SC.Writes = SchedClasses[OldSCIdx].Writes;
    SC.Reads = SchedClasses[OldSCIdx].Reads;
    SC.ProcIndices.push_back(0);
    // Map each Instr to this new class.
    // Note that InstDefs may be a smaller list than InstRWDef's "Instrs".
    Record *RWModelDef = InstRWDef->getValueAsDef("SchedModel");
    SmallSet<unsigned, 4> RemappedClassIDs;
    for (ArrayRef<Record*>::const_iterator
           II = InstDefs.begin(), IE = InstDefs.end(); II != IE; ++II) {
      unsigned OldSCIdx = InstrClassMap[*II];
      if (OldSCIdx && RemappedClassIDs.insert(OldSCIdx).second) {
        for (RecIter RI = SchedClasses[OldSCIdx].InstRWs.begin(),
               RE = SchedClasses[OldSCIdx].InstRWs.end(); RI != RE; ++RI) {
          if ((*RI)->getValueAsDef("SchedModel") == RWModelDef) {
            PrintFatalError(InstRWDef->getLoc(), "Overlapping InstRW def " +
                          (*II)->getName() + " also matches " +
                          (*RI)->getValue("Instrs")->getValue()->getAsString());
          }
          assert(*RI != InstRWDef && "SchedClass has duplicate InstRW def");
          SC.InstRWs.push_back(*RI);
        }
      }
      InstrClassMap[*II] = SCIdx;
    }
    SC.InstRWs.push_back(InstRWDef);
  }
}

// True if collectProcItins found anything.
bool CodeGenSchedModels::hasItineraries() const {
  for (CodeGenSchedModels::ProcIter PI = procModelBegin(), PE = procModelEnd();
       PI != PE; ++PI) {
    if (PI->hasItineraries())
      return true;
  }
  return false;
}

// Gather the processor itineraries.
void CodeGenSchedModels::collectProcItins() {
  for (CodeGenProcModel &ProcModel : ProcModels) {
    if (!ProcModel.hasItineraries())
      continue;

    RecVec ItinRecords = ProcModel.ItinsDef->getValueAsListOfDefs("IID");
    assert(!ItinRecords.empty() && "ProcModel.hasItineraries is incorrect");

    // Populate ItinDefList with Itinerary records.
    ProcModel.ItinDefList.resize(NumInstrSchedClasses);

    // Insert each itinerary data record in the correct position within
    // the processor model's ItinDefList.
    for (unsigned i = 0, N = ItinRecords.size(); i < N; i++) {
      Record *ItinData = ItinRecords[i];
      Record *ItinDef = ItinData->getValueAsDef("TheClass");
      bool FoundClass = false;
      for (SchedClassIter SCI = schedClassBegin(), SCE = schedClassEnd();
           SCI != SCE; ++SCI) {
        // Multiple SchedClasses may share an itinerary. Update all of them.
        if (SCI->ItinClassDef == ItinDef) {
          ProcModel.ItinDefList[SCI->Index] = ItinData;
          FoundClass = true;
        }
      }
      if (!FoundClass) {
        DEBUG(dbgs() << ProcModel.ItinsDef->getName()
              << " missing class for itinerary " << ItinDef->getName() << '\n');
      }
    }
    // Check for missing itinerary entries.
    assert(!ProcModel.ItinDefList[0] && "NoItinerary class can't have rec");
    DEBUG(
      for (unsigned i = 1, N = ProcModel.ItinDefList.size(); i < N; ++i) {
        if (!ProcModel.ItinDefList[i])
          dbgs() << ProcModel.ItinsDef->getName()
                 << " missing itinerary for class "
                 << SchedClasses[i].Name << '\n';
      });
  }
}

// Gather the read/write types for each itinerary class.
void CodeGenSchedModels::collectProcItinRW() {
  RecVec ItinRWDefs = Records.getAllDerivedDefinitions("ItinRW");
  std::sort(ItinRWDefs.begin(), ItinRWDefs.end(), LessRecord());
  for (RecIter II = ItinRWDefs.begin(), IE = ItinRWDefs.end(); II != IE; ++II) {
    if (!(*II)->getValueInit("SchedModel")->isComplete())
      PrintFatalError((*II)->getLoc(), "SchedModel is undefined");
    Record *ModelDef = (*II)->getValueAsDef("SchedModel");
    ProcModelMapTy::const_iterator I = ProcModelMap.find(ModelDef);
    if (I == ProcModelMap.end()) {
      PrintFatalError((*II)->getLoc(), "Undefined SchedMachineModel "
                    + ModelDef->getName());
    }
    ProcModels[I->second].ItinRWDefs.push_back(*II);
  }
}

/// Infer new classes from existing classes. In the process, this may create new
/// SchedWrites from sequences of existing SchedWrites.
void CodeGenSchedModels::inferSchedClasses() {
  DEBUG(dbgs() << NumInstrSchedClasses << " instr sched classes.\n");

  // Visit all existing classes and newly created classes.
  for (unsigned Idx = 0; Idx != SchedClasses.size(); ++Idx) {
    assert(SchedClasses[Idx].Index == Idx && "bad SCIdx");

    if (SchedClasses[Idx].ItinClassDef)
      inferFromItinClass(SchedClasses[Idx].ItinClassDef, Idx);
    if (!SchedClasses[Idx].InstRWs.empty())
      inferFromInstRWs(Idx);
    if (!SchedClasses[Idx].Writes.empty()) {
      inferFromRW(SchedClasses[Idx].Writes, SchedClasses[Idx].Reads,
                  Idx, SchedClasses[Idx].ProcIndices);
    }
    assert(SchedClasses.size() < (NumInstrSchedClasses*6) &&
           "too many SchedVariants");
  }
}

/// Infer classes from per-processor itinerary resources.
void CodeGenSchedModels::inferFromItinClass(Record *ItinClassDef,
                                            unsigned FromClassIdx) {
  for (unsigned PIdx = 0, PEnd = ProcModels.size(); PIdx != PEnd; ++PIdx) {
    const CodeGenProcModel &PM = ProcModels[PIdx];
    // For all ItinRW entries.
    bool HasMatch = false;
    for (RecIter II = PM.ItinRWDefs.begin(), IE = PM.ItinRWDefs.end();
         II != IE; ++II) {
      RecVec Matched = (*II)->getValueAsListOfDefs("MatchedItinClasses");
      if (!std::count(Matched.begin(), Matched.end(), ItinClassDef))
        continue;
      if (HasMatch)
        PrintFatalError((*II)->getLoc(), "Duplicate itinerary class "
                      + ItinClassDef->getName()
                      + " in ItinResources for " + PM.ModelName);
      HasMatch = true;
      IdxVec Writes, Reads;
      findRWs((*II)->getValueAsListOfDefs("OperandReadWrites"), Writes, Reads);
      IdxVec ProcIndices(1, PIdx);
      inferFromRW(Writes, Reads, FromClassIdx, ProcIndices);
    }
  }
}

/// Infer classes from per-processor InstReadWrite definitions.
void CodeGenSchedModels::inferFromInstRWs(unsigned SCIdx) {
  for (unsigned I = 0, E = SchedClasses[SCIdx].InstRWs.size(); I != E; ++I) {
    assert(SchedClasses[SCIdx].InstRWs.size() == E && "InstrRWs was mutated!");
    Record *Rec = SchedClasses[SCIdx].InstRWs[I];
    const RecVec *InstDefs = Sets.expand(Rec);
    RecIter II = InstDefs->begin(), IE = InstDefs->end();
    for (; II != IE; ++II) {
      if (InstrClassMap[*II] == SCIdx)
        break;
    }
    // If this class no longer has any instructions mapped to it, it has become
    // irrelevant.
    if (II == IE)
      continue;
    IdxVec Writes, Reads;
    findRWs(Rec->getValueAsListOfDefs("OperandReadWrites"), Writes, Reads);
    unsigned PIdx = getProcModel(Rec->getValueAsDef("SchedModel")).Index;
    IdxVec ProcIndices(1, PIdx);
    inferFromRW(Writes, Reads, SCIdx, ProcIndices); // May mutate SchedClasses.
  }
}

namespace {
// Helper for substituteVariantOperand.
struct TransVariant {
  Record *VarOrSeqDef;  // Variant or sequence.
  unsigned RWIdx;       // Index of this variant or sequence's matched type.
  unsigned ProcIdx;     // Processor model index or zero for any.
  unsigned TransVecIdx; // Index into PredTransitions::TransVec.

  TransVariant(Record *def, unsigned rwi, unsigned pi, unsigned ti):
    VarOrSeqDef(def), RWIdx(rwi), ProcIdx(pi), TransVecIdx(ti) {}
};

// Associate a predicate with the SchedReadWrite that it guards.
// RWIdx is the index of the read/write variant.
struct PredCheck {
  bool IsRead;
  unsigned RWIdx;
  Record *Predicate;

  PredCheck(bool r, unsigned w, Record *p): IsRead(r), RWIdx(w), Predicate(p) {}
};

// A Predicate transition is a list of RW sequences guarded by a PredTerm.
struct PredTransition {
  // A predicate term is a conjunction of PredChecks.
  SmallVector<PredCheck, 4> PredTerm;
  SmallVector<SmallVector<unsigned,4>, 16> WriteSequences;
  SmallVector<SmallVector<unsigned,4>, 16> ReadSequences;
  SmallVector<unsigned, 4> ProcIndices;
};

// Encapsulate a set of partially constructed transitions.
// The results are built by repeated calls to substituteVariants.
class PredTransitions {
  CodeGenSchedModels &SchedModels;

public:
  std::vector<PredTransition> TransVec;

  PredTransitions(CodeGenSchedModels &sm): SchedModels(sm) {}

  void substituteVariantOperand(const SmallVectorImpl<unsigned> &RWSeq,
                                bool IsRead, unsigned StartIdx);

  void substituteVariants(const PredTransition &Trans);

#ifndef NDEBUG
  void dump() const;
#endif

private:
  bool mutuallyExclusive(Record *PredDef, ArrayRef<PredCheck> Term);
  void getIntersectingVariants(
    const CodeGenSchedRW &SchedRW, unsigned TransIdx,
    std::vector<TransVariant> &IntersectingVariants);
  void pushVariant(const TransVariant &VInfo, bool IsRead);
};
} // anonymous

// Return true if this predicate is mutually exclusive with a PredTerm. This
// degenerates into checking if the predicate is mutually exclusive with any
// predicate in the Term's conjunction.
//
// All predicates associated with a given SchedRW are considered mutually
// exclusive. This should work even if the conditions expressed by the
// predicates are not exclusive because the predicates for a given SchedWrite
// are always checked in the order they are defined in the .td file. Later
// conditions implicitly negate any prior condition.
bool PredTransitions::mutuallyExclusive(Record *PredDef,
                                        ArrayRef<PredCheck> Term) {

  for (ArrayRef<PredCheck>::iterator I = Term.begin(), E = Term.end();
       I != E; ++I) {
    if (I->Predicate == PredDef)
      return false;

    const CodeGenSchedRW &SchedRW = SchedModels.getSchedRW(I->RWIdx, I->IsRead);
    assert(SchedRW.HasVariants && "PredCheck must refer to a SchedVariant");
    RecVec Variants = SchedRW.TheDef->getValueAsListOfDefs("Variants");
    for (RecIter VI = Variants.begin(), VE = Variants.end(); VI != VE; ++VI) {
      if ((*VI)->getValueAsDef("Predicate") == PredDef)
        return true;
    }
  }
  return false;
}

static bool hasAliasedVariants(const CodeGenSchedRW &RW,
                               CodeGenSchedModels &SchedModels) {
  if (RW.HasVariants)
    return true;

  for (RecIter I = RW.Aliases.begin(), E = RW.Aliases.end(); I != E; ++I) {
    const CodeGenSchedRW &AliasRW =
      SchedModels.getSchedRW((*I)->getValueAsDef("AliasRW"));
    if (AliasRW.HasVariants)
      return true;
    if (AliasRW.IsSequence) {
      IdxVec ExpandedRWs;
      SchedModels.expandRWSequence(AliasRW.Index, ExpandedRWs, AliasRW.IsRead);
      for (IdxIter SI = ExpandedRWs.begin(), SE = ExpandedRWs.end();
           SI != SE; ++SI) {
        if (hasAliasedVariants(SchedModels.getSchedRW(*SI, AliasRW.IsRead),
                               SchedModels)) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool hasVariant(ArrayRef<PredTransition> Transitions,
                       CodeGenSchedModels &SchedModels) {
  for (ArrayRef<PredTransition>::iterator
         PTI = Transitions.begin(), PTE = Transitions.end();
       PTI != PTE; ++PTI) {
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           WSI = PTI->WriteSequences.begin(), WSE = PTI->WriteSequences.end();
         WSI != WSE; ++WSI) {
      for (SmallVectorImpl<unsigned>::const_iterator
             WI = WSI->begin(), WE = WSI->end(); WI != WE; ++WI) {
        if (hasAliasedVariants(SchedModels.getSchedWrite(*WI), SchedModels))
          return true;
      }
    }
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           RSI = PTI->ReadSequences.begin(), RSE = PTI->ReadSequences.end();
         RSI != RSE; ++RSI) {
      for (SmallVectorImpl<unsigned>::const_iterator
             RI = RSI->begin(), RE = RSI->end(); RI != RE; ++RI) {
        if (hasAliasedVariants(SchedModels.getSchedRead(*RI), SchedModels))
          return true;
      }
    }
  }
  return false;
}

// Populate IntersectingVariants with any variants or aliased sequences of the
// given SchedRW whose processor indices and predicates are not mutually
// exclusive with the given transition.
void PredTransitions::getIntersectingVariants(
  const CodeGenSchedRW &SchedRW, unsigned TransIdx,
  std::vector<TransVariant> &IntersectingVariants) {

  bool GenericRW = false;

  std::vector<TransVariant> Variants;
  if (SchedRW.HasVariants) {
    unsigned VarProcIdx = 0;
    if (SchedRW.TheDef->getValueInit("SchedModel")->isComplete()) {
      Record *ModelDef = SchedRW.TheDef->getValueAsDef("SchedModel");
      VarProcIdx = SchedModels.getProcModel(ModelDef).Index;
    }
    // Push each variant. Assign TransVecIdx later.
    const RecVec VarDefs = SchedRW.TheDef->getValueAsListOfDefs("Variants");
    for (RecIter RI = VarDefs.begin(), RE = VarDefs.end(); RI != RE; ++RI)
      Variants.push_back(TransVariant(*RI, SchedRW.Index, VarProcIdx, 0));
    if (VarProcIdx == 0)
      GenericRW = true;
  }
  for (RecIter AI = SchedRW.Aliases.begin(), AE = SchedRW.Aliases.end();
       AI != AE; ++AI) {
    // If either the SchedAlias itself or the SchedReadWrite that it aliases
    // to is defined within a processor model, constrain all variants to
    // that processor.
    unsigned AliasProcIdx = 0;
    if ((*AI)->getValueInit("SchedModel")->isComplete()) {
      Record *ModelDef = (*AI)->getValueAsDef("SchedModel");
      AliasProcIdx = SchedModels.getProcModel(ModelDef).Index;
    }
    const CodeGenSchedRW &AliasRW =
      SchedModels.getSchedRW((*AI)->getValueAsDef("AliasRW"));

    if (AliasRW.HasVariants) {
      const RecVec VarDefs = AliasRW.TheDef->getValueAsListOfDefs("Variants");
      for (RecIter RI = VarDefs.begin(), RE = VarDefs.end(); RI != RE; ++RI)
        Variants.push_back(TransVariant(*RI, AliasRW.Index, AliasProcIdx, 0));
    }
    if (AliasRW.IsSequence) {
      Variants.push_back(
        TransVariant(AliasRW.TheDef, SchedRW.Index, AliasProcIdx, 0));
    }
    if (AliasProcIdx == 0)
      GenericRW = true;
  }
  for (unsigned VIdx = 0, VEnd = Variants.size(); VIdx != VEnd; ++VIdx) {
    TransVariant &Variant = Variants[VIdx];
    // Don't expand variants if the processor models don't intersect.
    // A zero processor index means any processor.
    SmallVectorImpl<unsigned> &ProcIndices = TransVec[TransIdx].ProcIndices;
    if (ProcIndices[0] && Variants[VIdx].ProcIdx) {
      unsigned Cnt = std::count(ProcIndices.begin(), ProcIndices.end(),
                                Variant.ProcIdx);
      if (!Cnt)
        continue;
      if (Cnt > 1) {
        const CodeGenProcModel &PM =
          *(SchedModels.procModelBegin() + Variant.ProcIdx);
        PrintFatalError(Variant.VarOrSeqDef->getLoc(),
                        "Multiple variants defined for processor " +
                        PM.ModelName +
                        " Ensure only one SchedAlias exists per RW.");
      }
    }
    if (Variant.VarOrSeqDef->isSubClassOf("SchedVar")) {
      Record *PredDef = Variant.VarOrSeqDef->getValueAsDef("Predicate");
      if (mutuallyExclusive(PredDef, TransVec[TransIdx].PredTerm))
        continue;
    }
    if (IntersectingVariants.empty()) {
      // The first variant builds on the existing transition.
      Variant.TransVecIdx = TransIdx;
      IntersectingVariants.push_back(Variant);
    }
    else {
      // Push another copy of the current transition for more variants.
      Variant.TransVecIdx = TransVec.size();
      IntersectingVariants.push_back(Variant);
      TransVec.push_back(TransVec[TransIdx]);
    }
  }
  if (GenericRW && IntersectingVariants.empty()) {
    PrintFatalError(SchedRW.TheDef->getLoc(), "No variant of this type has "
                    "a matching predicate on any processor");
  }
}

// Push the Reads/Writes selected by this variant onto the PredTransition
// specified by VInfo.
void PredTransitions::
pushVariant(const TransVariant &VInfo, bool IsRead) {

  PredTransition &Trans = TransVec[VInfo.TransVecIdx];

  // If this operand transition is reached through a processor-specific alias,
  // then the whole transition is specific to this processor.
  if (VInfo.ProcIdx != 0)
    Trans.ProcIndices.assign(1, VInfo.ProcIdx);

  IdxVec SelectedRWs;
  if (VInfo.VarOrSeqDef->isSubClassOf("SchedVar")) {
    Record *PredDef = VInfo.VarOrSeqDef->getValueAsDef("Predicate");
    Trans.PredTerm.push_back(PredCheck(IsRead, VInfo.RWIdx,PredDef));
    RecVec SelectedDefs = VInfo.VarOrSeqDef->getValueAsListOfDefs("Selected");
    SchedModels.findRWs(SelectedDefs, SelectedRWs, IsRead);
  }
  else {
    assert(VInfo.VarOrSeqDef->isSubClassOf("WriteSequence") &&
           "variant must be a SchedVariant or aliased WriteSequence");
    SelectedRWs.push_back(SchedModels.getSchedRWIdx(VInfo.VarOrSeqDef, IsRead));
  }

  const CodeGenSchedRW &SchedRW = SchedModels.getSchedRW(VInfo.RWIdx, IsRead);

  SmallVectorImpl<SmallVector<unsigned,4> > &RWSequences = IsRead
    ? Trans.ReadSequences : Trans.WriteSequences;
  if (SchedRW.IsVariadic) {
    unsigned OperIdx = RWSequences.size()-1;
    // Make N-1 copies of this transition's last sequence.
    for (unsigned i = 1, e = SelectedRWs.size(); i != e; ++i) {
      // Create a temporary copy the vector could reallocate.
      RWSequences.reserve(RWSequences.size() + 1);
      RWSequences.push_back(RWSequences[OperIdx]);
    }
    // Push each of the N elements of the SelectedRWs onto a copy of the last
    // sequence (split the current operand into N operands).
    // Note that write sequences should be expanded within this loop--the entire
    // sequence belongs to a single operand.
    for (IdxIter RWI = SelectedRWs.begin(), RWE = SelectedRWs.end();
         RWI != RWE; ++RWI, ++OperIdx) {
      IdxVec ExpandedRWs;
      if (IsRead)
        ExpandedRWs.push_back(*RWI);
      else
        SchedModels.expandRWSequence(*RWI, ExpandedRWs, IsRead);
      RWSequences[OperIdx].insert(RWSequences[OperIdx].end(),
                                  ExpandedRWs.begin(), ExpandedRWs.end());
    }
    assert(OperIdx == RWSequences.size() && "missed a sequence");
  }
  else {
    // Push this transition's expanded sequence onto this transition's last
    // sequence (add to the current operand's sequence).
    SmallVectorImpl<unsigned> &Seq = RWSequences.back();
    IdxVec ExpandedRWs;
    for (IdxIter RWI = SelectedRWs.begin(), RWE = SelectedRWs.end();
         RWI != RWE; ++RWI) {
      if (IsRead)
        ExpandedRWs.push_back(*RWI);
      else
        SchedModels.expandRWSequence(*RWI, ExpandedRWs, IsRead);
    }
    Seq.insert(Seq.end(), ExpandedRWs.begin(), ExpandedRWs.end());
  }
}

// RWSeq is a sequence of all Reads or all Writes for the next read or write
// operand. StartIdx is an index into TransVec where partial results
// starts. RWSeq must be applied to all transitions between StartIdx and the end
// of TransVec.
void PredTransitions::substituteVariantOperand(
  const SmallVectorImpl<unsigned> &RWSeq, bool IsRead, unsigned StartIdx) {

  // Visit each original RW within the current sequence.
  for (SmallVectorImpl<unsigned>::const_iterator
         RWI = RWSeq.begin(), RWE = RWSeq.end(); RWI != RWE; ++RWI) {
    const CodeGenSchedRW &SchedRW = SchedModels.getSchedRW(*RWI, IsRead);
    // Push this RW on all partial PredTransitions or distribute variants.
    // New PredTransitions may be pushed within this loop which should not be
    // revisited (TransEnd must be loop invariant).
    for (unsigned TransIdx = StartIdx, TransEnd = TransVec.size();
         TransIdx != TransEnd; ++TransIdx) {
      // In the common case, push RW onto the current operand's sequence.
      if (!hasAliasedVariants(SchedRW, SchedModels)) {
        if (IsRead)
          TransVec[TransIdx].ReadSequences.back().push_back(*RWI);
        else
          TransVec[TransIdx].WriteSequences.back().push_back(*RWI);
        continue;
      }
      // Distribute this partial PredTransition across intersecting variants.
      // This will push a copies of TransVec[TransIdx] on the back of TransVec.
      std::vector<TransVariant> IntersectingVariants;
      getIntersectingVariants(SchedRW, TransIdx, IntersectingVariants);
      // Now expand each variant on top of its copy of the transition.
      for (std::vector<TransVariant>::const_iterator
             IVI = IntersectingVariants.begin(),
             IVE = IntersectingVariants.end();
           IVI != IVE; ++IVI) {
        pushVariant(*IVI, IsRead);
      }
    }
  }
}

// For each variant of a Read/Write in Trans, substitute the sequence of
// Read/Writes guarded by the variant. This is exponential in the number of
// variant Read/Writes, but in practice detection of mutually exclusive
// predicates should result in linear growth in the total number variants.
//
// This is one step in a breadth-first search of nested variants.
void PredTransitions::substituteVariants(const PredTransition &Trans) {
  // Build up a set of partial results starting at the back of
  // PredTransitions. Remember the first new transition.
  unsigned StartIdx = TransVec.size();
  TransVec.resize(TransVec.size() + 1);
  TransVec.back().PredTerm = Trans.PredTerm;
  TransVec.back().ProcIndices = Trans.ProcIndices;

  // Visit each original write sequence.
  for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
         WSI = Trans.WriteSequences.begin(), WSE = Trans.WriteSequences.end();
       WSI != WSE; ++WSI) {
    // Push a new (empty) write sequence onto all partial Transitions.
    for (std::vector<PredTransition>::iterator I =
           TransVec.begin() + StartIdx, E = TransVec.end(); I != E; ++I) {
      I->WriteSequences.resize(I->WriteSequences.size() + 1);
    }
    substituteVariantOperand(*WSI, /*IsRead=*/false, StartIdx);
  }
  // Visit each original read sequence.
  for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
         RSI = Trans.ReadSequences.begin(), RSE = Trans.ReadSequences.end();
       RSI != RSE; ++RSI) {
    // Push a new (empty) read sequence onto all partial Transitions.
    for (std::vector<PredTransition>::iterator I =
           TransVec.begin() + StartIdx, E = TransVec.end(); I != E; ++I) {
      I->ReadSequences.resize(I->ReadSequences.size() + 1);
    }
    substituteVariantOperand(*RSI, /*IsRead=*/true, StartIdx);
  }
}

// Create a new SchedClass for each variant found by inferFromRW. Pass
static void inferFromTransitions(ArrayRef<PredTransition> LastTransitions,
                                 unsigned FromClassIdx,
                                 CodeGenSchedModels &SchedModels) {
  // For each PredTransition, create a new CodeGenSchedTransition, which usually
  // requires creating a new SchedClass.
  for (ArrayRef<PredTransition>::iterator
         I = LastTransitions.begin(), E = LastTransitions.end(); I != E; ++I) {
    IdxVec OperWritesVariant;
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           WSI = I->WriteSequences.begin(), WSE = I->WriteSequences.end();
         WSI != WSE; ++WSI) {
      // Create a new write representing the expanded sequence.
      OperWritesVariant.push_back(
        SchedModels.findOrInsertRW(*WSI, /*IsRead=*/false));
    }
    IdxVec OperReadsVariant;
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           RSI = I->ReadSequences.begin(), RSE = I->ReadSequences.end();
         RSI != RSE; ++RSI) {
      // Create a new read representing the expanded sequence.
      OperReadsVariant.push_back(
        SchedModels.findOrInsertRW(*RSI, /*IsRead=*/true));
    }
    IdxVec ProcIndices(I->ProcIndices.begin(), I->ProcIndices.end());
    CodeGenSchedTransition SCTrans;
    SCTrans.ToClassIdx =
      SchedModels.addSchedClass(/*ItinClassDef=*/nullptr, OperWritesVariant,
                                OperReadsVariant, ProcIndices);
    SCTrans.ProcIndices = ProcIndices;
    // The final PredTerm is unique set of predicates guarding the transition.
    RecVec Preds;
    for (SmallVectorImpl<PredCheck>::const_iterator
           PI = I->PredTerm.begin(), PE = I->PredTerm.end(); PI != PE; ++PI) {
      Preds.push_back(PI->Predicate);
    }
    RecIter PredsEnd = std::unique(Preds.begin(), Preds.end());
    Preds.resize(PredsEnd - Preds.begin());
    SCTrans.PredTerm = Preds;
    SchedModels.getSchedClass(FromClassIdx).Transitions.push_back(SCTrans);
  }
}

// Create new SchedClasses for the given ReadWrite list. If any of the
// ReadWrites refers to a SchedVariant, create a new SchedClass for each variant
// of the ReadWrite list, following Aliases if necessary.
void CodeGenSchedModels::inferFromRW(ArrayRef<unsigned> OperWrites,
                                     ArrayRef<unsigned> OperReads,
                                     unsigned FromClassIdx,
                                     ArrayRef<unsigned> ProcIndices) {
  DEBUG(dbgs() << "INFER RW proc("; dumpIdxVec(ProcIndices); dbgs() << ") ");

  // Create a seed transition with an empty PredTerm and the expanded sequences
  // of SchedWrites for the current SchedClass.
  std::vector<PredTransition> LastTransitions;
  LastTransitions.resize(1);
  LastTransitions.back().ProcIndices.append(ProcIndices.begin(),
                                            ProcIndices.end());

  for (unsigned WriteIdx : OperWrites) {
    IdxVec WriteSeq;
    expandRWSequence(WriteIdx, WriteSeq, /*IsRead=*/false);
    unsigned Idx = LastTransitions[0].WriteSequences.size();
    LastTransitions[0].WriteSequences.resize(Idx + 1);
    SmallVectorImpl<unsigned> &Seq = LastTransitions[0].WriteSequences[Idx];
    for (IdxIter WI = WriteSeq.begin(), WE = WriteSeq.end(); WI != WE; ++WI)
      Seq.push_back(*WI);
    DEBUG(dbgs() << "("; dumpIdxVec(Seq); dbgs() << ") ");
  }
  DEBUG(dbgs() << " Reads: ");
  for (unsigned ReadIdx : OperReads) {
    IdxVec ReadSeq;
    expandRWSequence(ReadIdx, ReadSeq, /*IsRead=*/true);
    unsigned Idx = LastTransitions[0].ReadSequences.size();
    LastTransitions[0].ReadSequences.resize(Idx + 1);
    SmallVectorImpl<unsigned> &Seq = LastTransitions[0].ReadSequences[Idx];
    for (IdxIter RI = ReadSeq.begin(), RE = ReadSeq.end(); RI != RE; ++RI)
      Seq.push_back(*RI);
    DEBUG(dbgs() << "("; dumpIdxVec(Seq); dbgs() << ") ");
  }
  DEBUG(dbgs() << '\n');

  // Collect all PredTransitions for individual operands.
  // Iterate until no variant writes remain.
  while (hasVariant(LastTransitions, *this)) {
    PredTransitions Transitions(*this);
    for (std::vector<PredTransition>::const_iterator
           I = LastTransitions.begin(), E = LastTransitions.end();
         I != E; ++I) {
      Transitions.substituteVariants(*I);
    }
    DEBUG(Transitions.dump());
    LastTransitions.swap(Transitions.TransVec);
  }
  // If the first transition has no variants, nothing to do.
  if (LastTransitions[0].PredTerm.empty())
    return;

  // WARNING: We are about to mutate the SchedClasses vector. Do not refer to
  // OperWrites, OperReads, or ProcIndices after calling inferFromTransitions.
  inferFromTransitions(LastTransitions, FromClassIdx, *this);
}

// Check if any processor resource group contains all resource records in
// SubUnits.
bool CodeGenSchedModels::hasSuperGroup(RecVec &SubUnits, CodeGenProcModel &PM) {
  for (unsigned i = 0, e = PM.ProcResourceDefs.size(); i < e; ++i) {
    if (!PM.ProcResourceDefs[i]->isSubClassOf("ProcResGroup"))
      continue;
    RecVec SuperUnits =
      PM.ProcResourceDefs[i]->getValueAsListOfDefs("Resources");
    RecIter RI = SubUnits.begin(), RE = SubUnits.end();
    for ( ; RI != RE; ++RI) {
      if (std::find(SuperUnits.begin(), SuperUnits.end(), *RI)
          == SuperUnits.end()) {
        break;
      }
    }
    if (RI == RE)
      return true;
  }
  return false;
}

// Verify that overlapping groups have a common supergroup.
void CodeGenSchedModels::verifyProcResourceGroups(CodeGenProcModel &PM) {
  for (unsigned i = 0, e = PM.ProcResourceDefs.size(); i < e; ++i) {
    if (!PM.ProcResourceDefs[i]->isSubClassOf("ProcResGroup"))
      continue;
    RecVec CheckUnits =
      PM.ProcResourceDefs[i]->getValueAsListOfDefs("Resources");
    for (unsigned j = i+1; j < e; ++j) {
      if (!PM.ProcResourceDefs[j]->isSubClassOf("ProcResGroup"))
        continue;
      RecVec OtherUnits =
        PM.ProcResourceDefs[j]->getValueAsListOfDefs("Resources");
      if (std::find_first_of(CheckUnits.begin(), CheckUnits.end(),
                             OtherUnits.begin(), OtherUnits.end())
          != CheckUnits.end()) {
        // CheckUnits and OtherUnits overlap
        OtherUnits.insert(OtherUnits.end(), CheckUnits.begin(),
                          CheckUnits.end());
        if (!hasSuperGroup(OtherUnits, PM)) {
          PrintFatalError((PM.ProcResourceDefs[i])->getLoc(),
                          "proc resource group overlaps with "
                          + PM.ProcResourceDefs[j]->getName()
                          + " but no supergroup contains both.");
        }
      }
    }
  }
}

// Collect and sort WriteRes, ReadAdvance, and ProcResources.
void CodeGenSchedModels::collectProcResources() {
  // Add any subtarget-specific SchedReadWrites that are directly associated
  // with processor resources. Refer to the parent SchedClass's ProcIndices to
  // determine which processors they apply to.
  for (SchedClassIter SCI = schedClassBegin(), SCE = schedClassEnd();
       SCI != SCE; ++SCI) {
    if (SCI->ItinClassDef)
      collectItinProcResources(SCI->ItinClassDef);
    else {
      // This class may have a default ReadWrite list which can be overriden by
      // InstRW definitions.
      if (!SCI->InstRWs.empty()) {
        for (RecIter RWI = SCI->InstRWs.begin(), RWE = SCI->InstRWs.end();
             RWI != RWE; ++RWI) {
          Record *RWModelDef = (*RWI)->getValueAsDef("SchedModel");
          IdxVec ProcIndices(1, getProcModel(RWModelDef).Index);
          IdxVec Writes, Reads;
          findRWs((*RWI)->getValueAsListOfDefs("OperandReadWrites"),
                  Writes, Reads);
          collectRWResources(Writes, Reads, ProcIndices);
        }
      }
      collectRWResources(SCI->Writes, SCI->Reads, SCI->ProcIndices);
    }
  }
  // Add resources separately defined by each subtarget.
  RecVec WRDefs = Records.getAllDerivedDefinitions("WriteRes");
  for (RecIter WRI = WRDefs.begin(), WRE = WRDefs.end(); WRI != WRE; ++WRI) {
    Record *ModelDef = (*WRI)->getValueAsDef("SchedModel");
    addWriteRes(*WRI, getProcModel(ModelDef).Index);
  }
  RecVec SWRDefs = Records.getAllDerivedDefinitions("SchedWriteRes");
  for (RecIter WRI = SWRDefs.begin(), WRE = SWRDefs.end(); WRI != WRE; ++WRI) {
    Record *ModelDef = (*WRI)->getValueAsDef("SchedModel");
    addWriteRes(*WRI, getProcModel(ModelDef).Index);
  }
  RecVec RADefs = Records.getAllDerivedDefinitions("ReadAdvance");
  for (RecIter RAI = RADefs.begin(), RAE = RADefs.end(); RAI != RAE; ++RAI) {
    Record *ModelDef = (*RAI)->getValueAsDef("SchedModel");
    addReadAdvance(*RAI, getProcModel(ModelDef).Index);
  }
  RecVec SRADefs = Records.getAllDerivedDefinitions("SchedReadAdvance");
  for (RecIter RAI = SRADefs.begin(), RAE = SRADefs.end(); RAI != RAE; ++RAI) {
    if ((*RAI)->getValueInit("SchedModel")->isComplete()) {
      Record *ModelDef = (*RAI)->getValueAsDef("SchedModel");
      addReadAdvance(*RAI, getProcModel(ModelDef).Index);
    }
  }
  // Add ProcResGroups that are defined within this processor model, which may
  // not be directly referenced but may directly specify a buffer size.
  RecVec ProcResGroups = Records.getAllDerivedDefinitions("ProcResGroup");
  for (RecIter RI = ProcResGroups.begin(), RE = ProcResGroups.end();
       RI != RE; ++RI) {
    if (!(*RI)->getValueInit("SchedModel")->isComplete())
      continue;
    CodeGenProcModel &PM = getProcModel((*RI)->getValueAsDef("SchedModel"));
    RecIter I = std::find(PM.ProcResourceDefs.begin(),
                          PM.ProcResourceDefs.end(), *RI);
    if (I == PM.ProcResourceDefs.end())
      PM.ProcResourceDefs.push_back(*RI);
  }
  // Finalize each ProcModel by sorting the record arrays.
  for (CodeGenProcModel &PM : ProcModels) {
    std::sort(PM.WriteResDefs.begin(), PM.WriteResDefs.end(),
              LessRecord());
    std::sort(PM.ReadAdvanceDefs.begin(), PM.ReadAdvanceDefs.end(),
              LessRecord());
    std::sort(PM.ProcResourceDefs.begin(), PM.ProcResourceDefs.end(),
              LessRecord());
    DEBUG(
      PM.dump();
      dbgs() << "WriteResDefs: ";
      for (RecIter RI = PM.WriteResDefs.begin(),
             RE = PM.WriteResDefs.end(); RI != RE; ++RI) {
        if ((*RI)->isSubClassOf("WriteRes"))
          dbgs() << (*RI)->getValueAsDef("WriteType")->getName() << " ";
        else
          dbgs() << (*RI)->getName() << " ";
      }
      dbgs() << "\nReadAdvanceDefs: ";
      for (RecIter RI = PM.ReadAdvanceDefs.begin(),
             RE = PM.ReadAdvanceDefs.end(); RI != RE; ++RI) {
        if ((*RI)->isSubClassOf("ReadAdvance"))
          dbgs() << (*RI)->getValueAsDef("ReadType")->getName() << " ";
        else
          dbgs() << (*RI)->getName() << " ";
      }
      dbgs() << "\nProcResourceDefs: ";
      for (RecIter RI = PM.ProcResourceDefs.begin(),
             RE = PM.ProcResourceDefs.end(); RI != RE; ++RI) {
        dbgs() << (*RI)->getName() << " ";
      }
      dbgs() << '\n');
    verifyProcResourceGroups(PM);
  }
}

// Collect itinerary class resources for each processor.
void CodeGenSchedModels::collectItinProcResources(Record *ItinClassDef) {
  for (unsigned PIdx = 0, PEnd = ProcModels.size(); PIdx != PEnd; ++PIdx) {
    const CodeGenProcModel &PM = ProcModels[PIdx];
    // For all ItinRW entries.
    bool HasMatch = false;
    for (RecIter II = PM.ItinRWDefs.begin(), IE = PM.ItinRWDefs.end();
         II != IE; ++II) {
      RecVec Matched = (*II)->getValueAsListOfDefs("MatchedItinClasses");
      if (!std::count(Matched.begin(), Matched.end(), ItinClassDef))
        continue;
      if (HasMatch)
        PrintFatalError((*II)->getLoc(), "Duplicate itinerary class "
                        + ItinClassDef->getName()
                        + " in ItinResources for " + PM.ModelName);
      HasMatch = true;
      IdxVec Writes, Reads;
      findRWs((*II)->getValueAsListOfDefs("OperandReadWrites"), Writes, Reads);
      IdxVec ProcIndices(1, PIdx);
      collectRWResources(Writes, Reads, ProcIndices);
    }
  }
}

void CodeGenSchedModels::collectRWResources(unsigned RWIdx, bool IsRead,
                                            ArrayRef<unsigned> ProcIndices) {
  const CodeGenSchedRW &SchedRW = getSchedRW(RWIdx, IsRead);
  if (SchedRW.TheDef) {
    if (!IsRead && SchedRW.TheDef->isSubClassOf("SchedWriteRes")) {
      for (unsigned Idx : ProcIndices)
        addWriteRes(SchedRW.TheDef, Idx);
    }
    else if (IsRead && SchedRW.TheDef->isSubClassOf("SchedReadAdvance")) {
      for (unsigned Idx : ProcIndices)
        addReadAdvance(SchedRW.TheDef, Idx);
    }
  }
  for (RecIter AI = SchedRW.Aliases.begin(), AE = SchedRW.Aliases.end();
       AI != AE; ++AI) {
    IdxVec AliasProcIndices;
    if ((*AI)->getValueInit("SchedModel")->isComplete()) {
      AliasProcIndices.push_back(
        getProcModel((*AI)->getValueAsDef("SchedModel")).Index);
    }
    else
      AliasProcIndices = ProcIndices;
    const CodeGenSchedRW &AliasRW = getSchedRW((*AI)->getValueAsDef("AliasRW"));
    assert(AliasRW.IsRead == IsRead && "cannot alias reads to writes");

    IdxVec ExpandedRWs;
    expandRWSequence(AliasRW.Index, ExpandedRWs, IsRead);
    for (IdxIter SI = ExpandedRWs.begin(), SE = ExpandedRWs.end();
         SI != SE; ++SI) {
      collectRWResources(*SI, IsRead, AliasProcIndices);
    }
  }
}

// Collect resources for a set of read/write types and processor indices.
void CodeGenSchedModels::collectRWResources(ArrayRef<unsigned> Writes,
                                            ArrayRef<unsigned> Reads,
                                            ArrayRef<unsigned> ProcIndices) {

  for (unsigned Idx : Writes)
    collectRWResources(Idx, /*IsRead=*/false, ProcIndices);

  for (unsigned Idx : Reads)
    collectRWResources(Idx, /*IsRead=*/true, ProcIndices);
}


// Find the processor's resource units for this kind of resource.
Record *CodeGenSchedModels::findProcResUnits(Record *ProcResKind,
                                             const CodeGenProcModel &PM) const {
  if (ProcResKind->isSubClassOf("ProcResourceUnits"))
    return ProcResKind;

  Record *ProcUnitDef = nullptr;
  RecVec ProcResourceDefs =
    Records.getAllDerivedDefinitions("ProcResourceUnits");

  for (RecIter RI = ProcResourceDefs.begin(), RE = ProcResourceDefs.end();
       RI != RE; ++RI) {

    if ((*RI)->getValueAsDef("Kind") == ProcResKind
        && (*RI)->getValueAsDef("SchedModel") == PM.ModelDef) {
      if (ProcUnitDef) {
        PrintFatalError((*RI)->getLoc(),
                        "Multiple ProcessorResourceUnits associated with "
                        + ProcResKind->getName());
      }
      ProcUnitDef = *RI;
    }
  }
  RecVec ProcResGroups = Records.getAllDerivedDefinitions("ProcResGroup");
  for (RecIter RI = ProcResGroups.begin(), RE = ProcResGroups.end();
       RI != RE; ++RI) {

    if (*RI == ProcResKind
        && (*RI)->getValueAsDef("SchedModel") == PM.ModelDef) {
      if (ProcUnitDef) {
        PrintFatalError((*RI)->getLoc(),
                        "Multiple ProcessorResourceUnits associated with "
                        + ProcResKind->getName());
      }
      ProcUnitDef = *RI;
    }
  }
  if (!ProcUnitDef) {
    PrintFatalError(ProcResKind->getLoc(),
                    "No ProcessorResources associated with "
                    + ProcResKind->getName());
  }
  return ProcUnitDef;
}

// Iteratively add a resource and its super resources.
void CodeGenSchedModels::addProcResource(Record *ProcResKind,
                                         CodeGenProcModel &PM) {
  for (;;) {
    Record *ProcResUnits = findProcResUnits(ProcResKind, PM);

    // See if this ProcResource is already associated with this processor.
    RecIter I = std::find(PM.ProcResourceDefs.begin(),
                          PM.ProcResourceDefs.end(), ProcResUnits);
    if (I != PM.ProcResourceDefs.end())
      return;

    PM.ProcResourceDefs.push_back(ProcResUnits);
    if (ProcResUnits->isSubClassOf("ProcResGroup"))
      return;

    if (!ProcResUnits->getValueInit("Super")->isComplete())
      return;

    ProcResKind = ProcResUnits->getValueAsDef("Super");
  }
}

// Add resources for a SchedWrite to this processor if they don't exist.
void CodeGenSchedModels::addWriteRes(Record *ProcWriteResDef, unsigned PIdx) {
  assert(PIdx && "don't add resources to an invalid Processor model");

  RecVec &WRDefs = ProcModels[PIdx].WriteResDefs;
  RecIter WRI = std::find(WRDefs.begin(), WRDefs.end(), ProcWriteResDef);
  if (WRI != WRDefs.end())
    return;
  WRDefs.push_back(ProcWriteResDef);

  // Visit ProcResourceKinds referenced by the newly discovered WriteRes.
  RecVec ProcResDefs = ProcWriteResDef->getValueAsListOfDefs("ProcResources");
  for (RecIter WritePRI = ProcResDefs.begin(), WritePRE = ProcResDefs.end();
       WritePRI != WritePRE; ++WritePRI) {
    addProcResource(*WritePRI, ProcModels[PIdx]);
  }
}

// Add resources for a ReadAdvance to this processor if they don't exist.
void CodeGenSchedModels::addReadAdvance(Record *ProcReadAdvanceDef,
                                        unsigned PIdx) {
  RecVec &RADefs = ProcModels[PIdx].ReadAdvanceDefs;
  RecIter I = std::find(RADefs.begin(), RADefs.end(), ProcReadAdvanceDef);
  if (I != RADefs.end())
    return;
  RADefs.push_back(ProcReadAdvanceDef);
}

unsigned CodeGenProcModel::getProcResourceIdx(Record *PRDef) const {
  RecIter PRPos = std::find(ProcResourceDefs.begin(), ProcResourceDefs.end(),
                            PRDef);
  if (PRPos == ProcResourceDefs.end())
    PrintFatalError(PRDef->getLoc(), "ProcResource def is not included in "
                    "the ProcResources list for " + ModelName);
  // Idx=0 is reserved for invalid.
  return 1 + (PRPos - ProcResourceDefs.begin());
}

#ifndef NDEBUG
void CodeGenProcModel::dump() const {
  dbgs() << Index << ": " << ModelName << " "
         << (ModelDef ? ModelDef->getName() : "inferred") << " "
         << (ItinsDef ? ItinsDef->getName() : "no itinerary") << '\n';
}

void CodeGenSchedRW::dump() const {
  dbgs() << Name << (IsVariadic ? " (V) " : " ");
  if (IsSequence) {
    dbgs() << "(";
    dumpIdxVec(Sequence);
    dbgs() << ")";
  }
}

void CodeGenSchedClass::dump(const CodeGenSchedModels* SchedModels) const {
  dbgs() << "SCHEDCLASS " << Index << ":" << Name << '\n'
         << "  Writes: ";
  for (unsigned i = 0, N = Writes.size(); i < N; ++i) {
    SchedModels->getSchedWrite(Writes[i]).dump();
    if (i < N-1) {
      dbgs() << '\n';
      dbgs().indent(10);
    }
  }
  dbgs() << "\n  Reads: ";
  for (unsigned i = 0, N = Reads.size(); i < N; ++i) {
    SchedModels->getSchedRead(Reads[i]).dump();
    if (i < N-1) {
      dbgs() << '\n';
      dbgs().indent(10);
    }
  }
  dbgs() << "\n  ProcIdx: "; dumpIdxVec(ProcIndices); dbgs() << '\n';
  if (!Transitions.empty()) {
    dbgs() << "\n Transitions for Proc ";
    for (std::vector<CodeGenSchedTransition>::const_iterator
           TI = Transitions.begin(), TE = Transitions.end(); TI != TE; ++TI) {
      dumpIdxVec(TI->ProcIndices);
    }
  }
}

void PredTransitions::dump() const {
  dbgs() << "Expanded Variants:\n";
  for (std::vector<PredTransition>::const_iterator
         TI = TransVec.begin(), TE = TransVec.end(); TI != TE; ++TI) {
    dbgs() << "{";
    for (SmallVectorImpl<PredCheck>::const_iterator
           PCI = TI->PredTerm.begin(), PCE = TI->PredTerm.end();
         PCI != PCE; ++PCI) {
      if (PCI != TI->PredTerm.begin())
        dbgs() << ", ";
      dbgs() << SchedModels.getSchedRW(PCI->RWIdx, PCI->IsRead).Name
             << ":" << PCI->Predicate->getName();
    }
    dbgs() << "},\n  => {";
    for (SmallVectorImpl<SmallVector<unsigned,4> >::const_iterator
           WSI = TI->WriteSequences.begin(), WSE = TI->WriteSequences.end();
         WSI != WSE; ++WSI) {
      dbgs() << "(";
      for (SmallVectorImpl<unsigned>::const_iterator
             WI = WSI->begin(), WE = WSI->end(); WI != WE; ++WI) {
        if (WI != WSI->begin())
          dbgs() << ", ";
        dbgs() << SchedModels.getSchedWrite(*WI).Name;
      }
      dbgs() << "),";
    }
    dbgs() << "}\n";
  }
}
#endif // NDEBUG
