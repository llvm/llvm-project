//===- CodeGenRegisters.cpp - Register and RegisterClass Info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines structures to encapsulate information gleaned from the
// target register and register class definitions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenRegisters.h"
#include "CodeGenTarget.h"
#include "llvm/TableGen/Error.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//                             CodeGenSubRegIndex
//===----------------------------------------------------------------------===//

CodeGenSubRegIndex::CodeGenSubRegIndex(Record *R, unsigned Enum)
  : TheDef(R), EnumValue(Enum), LaneMask(0) {
  Name = R->getName();
  if (R->getValue("Namespace"))
    Namespace = R->getValueAsString("Namespace");
}

CodeGenSubRegIndex::CodeGenSubRegIndex(StringRef N, StringRef Nspace,
                                       unsigned Enum)
  : TheDef(0), Name(N), Namespace(Nspace), EnumValue(Enum), LaneMask(0) {
}

std::string CodeGenSubRegIndex::getQualifiedName() const {
  std::string N = getNamespace();
  if (!N.empty())
    N += "::";
  N += getName();
  return N;
}

void CodeGenSubRegIndex::updateComponents(CodeGenRegBank &RegBank) {
  if (!TheDef)
    return;

  std::vector<Record*> Comps = TheDef->getValueAsListOfDefs("ComposedOf");
  if (!Comps.empty()) {
    if (Comps.size() != 2)
      PrintFatalError(TheDef->getLoc(),
                      "ComposedOf must have exactly two entries");
    CodeGenSubRegIndex *A = RegBank.getSubRegIdx(Comps[0]);
    CodeGenSubRegIndex *B = RegBank.getSubRegIdx(Comps[1]);
    CodeGenSubRegIndex *X = A->addComposite(B, this);
    if (X)
      PrintFatalError(TheDef->getLoc(), "Ambiguous ComposedOf entries");
  }

  std::vector<Record*> Parts =
    TheDef->getValueAsListOfDefs("CoveringSubRegIndices");
  if (!Parts.empty()) {
    if (Parts.size() < 2)
      PrintFatalError(TheDef->getLoc(),
                    "CoveredBySubRegs must have two or more entries");
    SmallVector<CodeGenSubRegIndex*, 8> IdxParts;
    for (unsigned i = 0, e = Parts.size(); i != e; ++i)
      IdxParts.push_back(RegBank.getSubRegIdx(Parts[i]));
    RegBank.addConcatSubRegIndex(IdxParts, this);
  }
}

unsigned CodeGenSubRegIndex::computeLaneMask() {
  // Already computed?
  if (LaneMask)
    return LaneMask;

  // Recursion guard, shouldn't be required.
  LaneMask = ~0u;

  // The lane mask is simply the union of all sub-indices.
  unsigned M = 0;
  for (CompMap::iterator I = Composed.begin(), E = Composed.end(); I != E; ++I)
    M |= I->second->computeLaneMask();
  assert(M && "Missing lane mask, sub-register cycle?");
  LaneMask = M;
  return LaneMask;
}

//===----------------------------------------------------------------------===//
//                              CodeGenRegister
//===----------------------------------------------------------------------===//

CodeGenRegister::CodeGenRegister(Record *R, unsigned Enum)
  : TheDef(R),
    EnumValue(Enum),
    CostPerUse(R->getValueAsInt("CostPerUse")),
    CoveredBySubRegs(R->getValueAsBit("CoveredBySubRegs")),
    NumNativeRegUnits(0),
    SubRegsComplete(false),
    SuperRegsComplete(false),
    TopoSig(~0u)
{}

void CodeGenRegister::buildObjectGraph(CodeGenRegBank &RegBank) {
  std::vector<Record*> SRIs = TheDef->getValueAsListOfDefs("SubRegIndices");
  std::vector<Record*> SRs = TheDef->getValueAsListOfDefs("SubRegs");

  if (SRIs.size() != SRs.size())
    PrintFatalError(TheDef->getLoc(),
                    "SubRegs and SubRegIndices must have the same size");

  for (unsigned i = 0, e = SRIs.size(); i != e; ++i) {
    ExplicitSubRegIndices.push_back(RegBank.getSubRegIdx(SRIs[i]));
    ExplicitSubRegs.push_back(RegBank.getReg(SRs[i]));
  }

  // Also compute leading super-registers. Each register has a list of
  // covered-by-subregs super-registers where it appears as the first explicit
  // sub-register.
  //
  // This is used by computeSecondarySubRegs() to find candidates.
  if (CoveredBySubRegs && !ExplicitSubRegs.empty())
    ExplicitSubRegs.front()->LeadingSuperRegs.push_back(this);

  // Add ad hoc alias links. This is a symmetric relationship between two
  // registers, so build a symmetric graph by adding links in both ends.
  std::vector<Record*> Aliases = TheDef->getValueAsListOfDefs("Aliases");
  for (unsigned i = 0, e = Aliases.size(); i != e; ++i) {
    CodeGenRegister *Reg = RegBank.getReg(Aliases[i]);
    ExplicitAliases.push_back(Reg);
    Reg->ExplicitAliases.push_back(this);
  }
}

const std::string &CodeGenRegister::getName() const {
  return TheDef->getName();
}

namespace {
// Iterate over all register units in a set of registers.
class RegUnitIterator {
  CodeGenRegister::Set::const_iterator RegI, RegE;
  CodeGenRegister::RegUnitList::const_iterator UnitI, UnitE;

public:
  RegUnitIterator(const CodeGenRegister::Set &Regs):
    RegI(Regs.begin()), RegE(Regs.end()), UnitI(), UnitE() {

    if (RegI != RegE) {
      UnitI = (*RegI)->getRegUnits().begin();
      UnitE = (*RegI)->getRegUnits().end();
      advance();
    }
  }

  bool isValid() const { return UnitI != UnitE; }

  unsigned operator* () const { assert(isValid()); return *UnitI; }

  const CodeGenRegister *getReg() const { assert(isValid()); return *RegI; }

  /// Preincrement.  Move to the next unit.
  void operator++() {
    assert(isValid() && "Cannot advance beyond the last operand");
    ++UnitI;
    advance();
  }

protected:
  void advance() {
    while (UnitI == UnitE) {
      if (++RegI == RegE)
        break;
      UnitI = (*RegI)->getRegUnits().begin();
      UnitE = (*RegI)->getRegUnits().end();
    }
  }
};
} // namespace

// Merge two RegUnitLists maintaining the order and removing duplicates.
// Overwrites MergedRU in the process.
static void mergeRegUnits(CodeGenRegister::RegUnitList &MergedRU,
                          const CodeGenRegister::RegUnitList &RRU) {
  CodeGenRegister::RegUnitList LRU = MergedRU;
  MergedRU.clear();
  std::set_union(LRU.begin(), LRU.end(), RRU.begin(), RRU.end(),
                 std::back_inserter(MergedRU));
}

// Return true of this unit appears in RegUnits.
static bool hasRegUnit(CodeGenRegister::RegUnitList &RegUnits, unsigned Unit) {
  return std::count(RegUnits.begin(), RegUnits.end(), Unit);
}

// Inherit register units from subregisters.
// Return true if the RegUnits changed.
bool CodeGenRegister::inheritRegUnits(CodeGenRegBank &RegBank) {
  unsigned OldNumUnits = RegUnits.size();
  for (SubRegMap::const_iterator I = SubRegs.begin(), E = SubRegs.end();
       I != E; ++I) {
    CodeGenRegister *SR = I->second;
    // Merge the subregister's units into this register's RegUnits.
    mergeRegUnits(RegUnits, SR->RegUnits);
  }
  return OldNumUnits != RegUnits.size();
}

const CodeGenRegister::SubRegMap &
CodeGenRegister::computeSubRegs(CodeGenRegBank &RegBank) {
  // Only compute this map once.
  if (SubRegsComplete)
    return SubRegs;
  SubRegsComplete = true;

  // First insert the explicit subregs and make sure they are fully indexed.
  for (unsigned i = 0, e = ExplicitSubRegs.size(); i != e; ++i) {
    CodeGenRegister *SR = ExplicitSubRegs[i];
    CodeGenSubRegIndex *Idx = ExplicitSubRegIndices[i];
    if (!SubRegs.insert(std::make_pair(Idx, SR)).second)
      PrintFatalError(TheDef->getLoc(), "SubRegIndex " + Idx->getName() +
                      " appears twice in Register " + getName());
    // Map explicit sub-registers first, so the names take precedence.
    // The inherited sub-registers are mapped below.
    SubReg2Idx.insert(std::make_pair(SR, Idx));
  }

  // Keep track of inherited subregs and how they can be reached.
  SmallPtrSet<CodeGenRegister*, 8> Orphans;

  // Clone inherited subregs and place duplicate entries in Orphans.
  // Here the order is important - earlier subregs take precedence.
  for (unsigned i = 0, e = ExplicitSubRegs.size(); i != e; ++i) {
    CodeGenRegister *SR = ExplicitSubRegs[i];
    const SubRegMap &Map = SR->computeSubRegs(RegBank);

    for (SubRegMap::const_iterator SI = Map.begin(), SE = Map.end(); SI != SE;
         ++SI) {
      if (!SubRegs.insert(*SI).second)
        Orphans.insert(SI->second);
    }
  }

  // Expand any composed subreg indices.
  // If dsub_2 has ComposedOf = [qsub_1, dsub_0], and this register has a
  // qsub_1 subreg, add a dsub_2 subreg.  Keep growing Indices and process
  // expanded subreg indices recursively.
  SmallVector<CodeGenSubRegIndex*, 8> Indices = ExplicitSubRegIndices;
  for (unsigned i = 0; i != Indices.size(); ++i) {
    CodeGenSubRegIndex *Idx = Indices[i];
    const CodeGenSubRegIndex::CompMap &Comps = Idx->getComposites();
    CodeGenRegister *SR = SubRegs[Idx];
    const SubRegMap &Map = SR->computeSubRegs(RegBank);

    // Look at the possible compositions of Idx.
    // They may not all be supported by SR.
    for (CodeGenSubRegIndex::CompMap::const_iterator I = Comps.begin(),
           E = Comps.end(); I != E; ++I) {
      SubRegMap::const_iterator SRI = Map.find(I->first);
      if (SRI == Map.end())
        continue; // Idx + I->first doesn't exist in SR.
      // Add I->second as a name for the subreg SRI->second, assuming it is
      // orphaned, and the name isn't already used for something else.
      if (SubRegs.count(I->second) || !Orphans.erase(SRI->second))
        continue;
      // We found a new name for the orphaned sub-register.
      SubRegs.insert(std::make_pair(I->second, SRI->second));
      Indices.push_back(I->second);
    }
  }

  // Now Orphans contains the inherited subregisters without a direct index.
  // Create inferred indexes for all missing entries.
  // Work backwards in the Indices vector in order to compose subregs bottom-up.
  // Consider this subreg sequence:
  //
  //   qsub_1 -> dsub_0 -> ssub_0
  //
  // The qsub_1 -> dsub_0 composition becomes dsub_2, so the ssub_0 register
  // can be reached in two different ways:
  //
  //   qsub_1 -> ssub_0
  //   dsub_2 -> ssub_0
  //
  // We pick the latter composition because another register may have [dsub_0,
  // dsub_1, dsub_2] subregs without necessarily having a qsub_1 subreg.  The
  // dsub_2 -> ssub_0 composition can be shared.
  while (!Indices.empty() && !Orphans.empty()) {
    CodeGenSubRegIndex *Idx = Indices.pop_back_val();
    CodeGenRegister *SR = SubRegs[Idx];
    const SubRegMap &Map = SR->computeSubRegs(RegBank);
    for (SubRegMap::const_iterator SI = Map.begin(), SE = Map.end(); SI != SE;
         ++SI)
      if (Orphans.erase(SI->second))
        SubRegs[RegBank.getCompositeSubRegIndex(Idx, SI->first)] = SI->second;
  }

  // Compute the inverse SubReg -> Idx map.
  for (SubRegMap::const_iterator SI = SubRegs.begin(), SE = SubRegs.end();
       SI != SE; ++SI) {
    if (SI->second == this) {
      ArrayRef<SMLoc> Loc;
      if (TheDef)
        Loc = TheDef->getLoc();
      PrintFatalError(Loc, "Register " + getName() +
                      " has itself as a sub-register");
    }
    // Ensure that every sub-register has a unique name.
    DenseMap<const CodeGenRegister*, CodeGenSubRegIndex*>::iterator Ins =
      SubReg2Idx.insert(std::make_pair(SI->second, SI->first)).first;
    if (Ins->second == SI->first)
      continue;
    // Trouble: Two different names for SI->second.
    ArrayRef<SMLoc> Loc;
    if (TheDef)
      Loc = TheDef->getLoc();
    PrintFatalError(Loc, "Sub-register can't have two names: " +
                  SI->second->getName() + " available as " +
                  SI->first->getName() + " and " + Ins->second->getName());
  }

  // Derive possible names for sub-register concatenations from any explicit
  // sub-registers. By doing this before computeSecondarySubRegs(), we ensure
  // that getConcatSubRegIndex() won't invent any concatenated indices that the
  // user already specified.
  for (unsigned i = 0, e = ExplicitSubRegs.size(); i != e; ++i) {
    CodeGenRegister *SR = ExplicitSubRegs[i];
    if (!SR->CoveredBySubRegs || SR->ExplicitSubRegs.size() <= 1)
      continue;

    // SR is composed of multiple sub-regs. Find their names in this register.
    SmallVector<CodeGenSubRegIndex*, 8> Parts;
    for (unsigned j = 0, e = SR->ExplicitSubRegs.size(); j != e; ++j)
      Parts.push_back(getSubRegIndex(SR->ExplicitSubRegs[j]));

    // Offer this as an existing spelling for the concatenation of Parts.
    RegBank.addConcatSubRegIndex(Parts, ExplicitSubRegIndices[i]);
  }

  // Initialize RegUnitList. Because getSubRegs is called recursively, this
  // processes the register hierarchy in postorder.
  //
  // Inherit all sub-register units. It is good enough to look at the explicit
  // sub-registers, the other registers won't contribute any more units.
  for (unsigned i = 0, e = ExplicitSubRegs.size(); i != e; ++i) {
    CodeGenRegister *SR = ExplicitSubRegs[i];
    // Explicit sub-registers are usually disjoint, so this is a good way of
    // computing the union. We may pick up a few duplicates that will be
    // eliminated below.
    unsigned N = RegUnits.size();
    RegUnits.append(SR->RegUnits.begin(), SR->RegUnits.end());
    std::inplace_merge(RegUnits.begin(), RegUnits.begin() + N, RegUnits.end());
  }
  RegUnits.erase(std::unique(RegUnits.begin(), RegUnits.end()), RegUnits.end());

  // Absent any ad hoc aliasing, we create one register unit per leaf register.
  // These units correspond to the maximal cliques in the register overlap
  // graph which is optimal.
  //
  // When there is ad hoc aliasing, we simply create one unit per edge in the
  // undirected ad hoc aliasing graph. Technically, we could do better by
  // identifying maximal cliques in the ad hoc graph, but cliques larger than 2
  // are extremely rare anyway (I've never seen one), so we don't bother with
  // the added complexity.
  for (unsigned i = 0, e = ExplicitAliases.size(); i != e; ++i) {
    CodeGenRegister *AR = ExplicitAliases[i];
    // Only visit each edge once.
    if (AR->SubRegsComplete)
      continue;
    // Create a RegUnit representing this alias edge, and add it to both
    // registers.
    unsigned Unit = RegBank.newRegUnit(this, AR);
    RegUnits.push_back(Unit);
    AR->RegUnits.push_back(Unit);
  }

  // Finally, create units for leaf registers without ad hoc aliases. Note that
  // a leaf register with ad hoc aliases doesn't get its own unit - it isn't
  // necessary. This means the aliasing leaf registers can share a single unit.
  if (RegUnits.empty())
    RegUnits.push_back(RegBank.newRegUnit(this));

  // We have now computed the native register units. More may be adopted later
  // for balancing purposes.
  NumNativeRegUnits = RegUnits.size();

  return SubRegs;
}

// In a register that is covered by its sub-registers, try to find redundant
// sub-registers. For example:
//
//   QQ0 = {Q0, Q1}
//   Q0 = {D0, D1}
//   Q1 = {D2, D3}
//
// We can infer that D1_D2 is also a sub-register, even if it wasn't named in
// the register definition.
//
// The explicitly specified registers form a tree. This function discovers
// sub-register relationships that would force a DAG.
//
void CodeGenRegister::computeSecondarySubRegs(CodeGenRegBank &RegBank) {
  // Collect new sub-registers first, add them later.
  SmallVector<SubRegMap::value_type, 8> NewSubRegs;

  // Look at the leading super-registers of each sub-register. Those are the
  // candidates for new sub-registers, assuming they are fully contained in
  // this register.
  for (SubRegMap::iterator I = SubRegs.begin(), E = SubRegs.end(); I != E; ++I){
    const CodeGenRegister *SubReg = I->second;
    const CodeGenRegister::SuperRegList &Leads = SubReg->LeadingSuperRegs;
    for (unsigned i = 0, e = Leads.size(); i != e; ++i) {
      CodeGenRegister *Cand = const_cast<CodeGenRegister*>(Leads[i]);
      // Already got this sub-register?
      if (Cand == this || getSubRegIndex(Cand))
        continue;
      // Check if each component of Cand is already a sub-register.
      // We know that the first component is I->second, and is present with the
      // name I->first.
      SmallVector<CodeGenSubRegIndex*, 8> Parts(1, I->first);
      assert(!Cand->ExplicitSubRegs.empty() &&
             "Super-register has no sub-registers");
      for (unsigned j = 1, e = Cand->ExplicitSubRegs.size(); j != e; ++j) {
        if (CodeGenSubRegIndex *Idx = getSubRegIndex(Cand->ExplicitSubRegs[j]))
          Parts.push_back(Idx);
        else {
          // Sub-register doesn't exist.
          Parts.clear();
          break;
        }
      }
      // If some Cand sub-register is not part of this register, or if Cand only
      // has one sub-register, there is nothing to do.
      if (Parts.size() <= 1)
        continue;

      // Each part of Cand is a sub-register of this. Make the full Cand also
      // a sub-register with a concatenated sub-register index.
      CodeGenSubRegIndex *Concat= RegBank.getConcatSubRegIndex(Parts);
      NewSubRegs.push_back(std::make_pair(Concat, Cand));
    }
  }

  // Now add all the new sub-registers.
  for (unsigned i = 0, e = NewSubRegs.size(); i != e; ++i) {
    // Don't add Cand if another sub-register is already using the index.
    if (!SubRegs.insert(NewSubRegs[i]).second)
      continue;

    CodeGenSubRegIndex *NewIdx = NewSubRegs[i].first;
    CodeGenRegister *NewSubReg = NewSubRegs[i].second;
    SubReg2Idx.insert(std::make_pair(NewSubReg, NewIdx));
  }

  // Create sub-register index composition maps for the synthesized indices.
  for (unsigned i = 0, e = NewSubRegs.size(); i != e; ++i) {
    CodeGenSubRegIndex *NewIdx = NewSubRegs[i].first;
    CodeGenRegister *NewSubReg = NewSubRegs[i].second;
    for (SubRegMap::const_iterator SI = NewSubReg->SubRegs.begin(),
           SE = NewSubReg->SubRegs.end(); SI != SE; ++SI) {
      CodeGenSubRegIndex *SubIdx = getSubRegIndex(SI->second);
      if (!SubIdx)
        PrintFatalError(TheDef->getLoc(), "No SubRegIndex for " +
                        SI->second->getName() + " in " + getName());
      NewIdx->addComposite(SI->first, SubIdx);
    }
  }
}

void CodeGenRegister::computeSuperRegs(CodeGenRegBank &RegBank) {
  // Only visit each register once.
  if (SuperRegsComplete)
    return;
  SuperRegsComplete = true;

  // Make sure all sub-registers have been visited first, so the super-reg
  // lists will be topologically ordered.
  for (SubRegMap::const_iterator I = SubRegs.begin(), E = SubRegs.end();
       I != E; ++I)
    I->second->computeSuperRegs(RegBank);

  // Now add this as a super-register on all sub-registers.
  // Also compute the TopoSigId in post-order.
  TopoSigId Id;
  for (SubRegMap::const_iterator I = SubRegs.begin(), E = SubRegs.end();
       I != E; ++I) {
    // Topological signature computed from SubIdx, TopoId(SubReg).
    // Loops and idempotent indices have TopoSig = ~0u.
    Id.push_back(I->first->EnumValue);
    Id.push_back(I->second->TopoSig);

    // Don't add duplicate entries.
    if (!I->second->SuperRegs.empty() && I->second->SuperRegs.back() == this)
      continue;
    I->second->SuperRegs.push_back(this);
  }
  TopoSig = RegBank.getTopoSig(Id);
}

void
CodeGenRegister::addSubRegsPreOrder(SetVector<const CodeGenRegister*> &OSet,
                                    CodeGenRegBank &RegBank) const {
  assert(SubRegsComplete && "Must precompute sub-registers");
  for (unsigned i = 0, e = ExplicitSubRegs.size(); i != e; ++i) {
    CodeGenRegister *SR = ExplicitSubRegs[i];
    if (OSet.insert(SR))
      SR->addSubRegsPreOrder(OSet, RegBank);
  }
  // Add any secondary sub-registers that weren't part of the explicit tree.
  for (SubRegMap::const_iterator I = SubRegs.begin(), E = SubRegs.end();
       I != E; ++I)
    OSet.insert(I->second);
}

// Compute overlapping registers.
//
// The standard set is all super-registers and all sub-registers, but the
// target description can add arbitrary overlapping registers via the 'Aliases'
// field. This complicates things, but we can compute overlapping sets using
// the following rules:
//
// 1. The relation overlap(A, B) is reflexive and symmetric but not transitive.
//
// 2. overlap(A, B) implies overlap(A, S) for all S in supers(B).
//
// Alternatively:
//
//    overlap(A, B) iff there exists:
//    A' in { A, subregs(A) } and B' in { B, subregs(B) } such that:
//    A' = B' or A' in aliases(B') or B' in aliases(A').
//
// Here subregs(A) is the full flattened sub-register set returned by
// A.getSubRegs() while aliases(A) is simply the special 'Aliases' field in the
// description of register A.
//
// This also implies that registers with a common sub-register are considered
// overlapping. This can happen when forming register pairs:
//
//    P0 = (R0, R1)
//    P1 = (R1, R2)
//    P2 = (R2, R3)
//
// In this case, we will infer an overlap between P0 and P1 because of the
// shared sub-register R1. There is no overlap between P0 and P2.
//
void CodeGenRegister::computeOverlaps(CodeGenRegister::Set &Overlaps,
                                      const CodeGenRegBank &RegBank) const {
  assert(!RegUnits.empty() && "Compute register units before overlaps.");

  // Register units are assigned such that the overlapping registers are the
  // super-registers of the root registers of the register units.
  for (unsigned rui = 0, rue = RegUnits.size(); rui != rue; ++rui) {
    const RegUnit &RU = RegBank.getRegUnit(RegUnits[rui]);
    ArrayRef<const CodeGenRegister*> Roots = RU.getRoots();
    for (unsigned ri = 0, re = Roots.size(); ri != re; ++ri) {
      const CodeGenRegister *Root = Roots[ri];
      Overlaps.insert(Root);
      ArrayRef<const CodeGenRegister*> Supers = Root->getSuperRegs();
      Overlaps.insert(Supers.begin(), Supers.end());
    }
  }
}

// Get the sum of this register's unit weights.
unsigned CodeGenRegister::getWeight(const CodeGenRegBank &RegBank) const {
  unsigned Weight = 0;
  for (RegUnitList::const_iterator I = RegUnits.begin(), E = RegUnits.end();
       I != E; ++I) {
    Weight += RegBank.getRegUnit(*I).Weight;
  }
  return Weight;
}

//===----------------------------------------------------------------------===//
//                               RegisterTuples
//===----------------------------------------------------------------------===//

// A RegisterTuples def is used to generate pseudo-registers from lists of
// sub-registers. We provide a SetTheory expander class that returns the new
// registers.
namespace {
struct TupleExpander : SetTheory::Expander {
  void expand(SetTheory &ST, Record *Def, SetTheory::RecSet &Elts) {
    std::vector<Record*> Indices = Def->getValueAsListOfDefs("SubRegIndices");
    unsigned Dim = Indices.size();
    ListInit *SubRegs = Def->getValueAsListInit("SubRegs");
    if (Dim != SubRegs->getSize())
      PrintFatalError(Def->getLoc(), "SubRegIndices and SubRegs size mismatch");
    if (Dim < 2)
      PrintFatalError(Def->getLoc(),
                      "Tuples must have at least 2 sub-registers");

    // Evaluate the sub-register lists to be zipped.
    unsigned Length = ~0u;
    SmallVector<SetTheory::RecSet, 4> Lists(Dim);
    for (unsigned i = 0; i != Dim; ++i) {
      ST.evaluate(SubRegs->getElement(i), Lists[i], Def->getLoc());
      Length = std::min(Length, unsigned(Lists[i].size()));
    }

    if (Length == 0)
      return;

    // Precompute some types.
    Record *RegisterCl = Def->getRecords().getClass("Register");
    RecTy *RegisterRecTy = RecordRecTy::get(RegisterCl);
    StringInit *BlankName = StringInit::get("");

    // Zip them up.
    for (unsigned n = 0; n != Length; ++n) {
      std::string Name;
      Record *Proto = Lists[0][n];
      std::vector<Init*> Tuple;
      unsigned CostPerUse = 0;
      for (unsigned i = 0; i != Dim; ++i) {
        Record *Reg = Lists[i][n];
        if (i) Name += '_';
        Name += Reg->getName();
        Tuple.push_back(DefInit::get(Reg));
        CostPerUse = std::max(CostPerUse,
                              unsigned(Reg->getValueAsInt("CostPerUse")));
      }

      // Create a new Record representing the synthesized register. This record
      // is only for consumption by CodeGenRegister, it is not added to the
      // RecordKeeper.
      Record *NewReg = new Record(Name, Def->getLoc(), Def->getRecords());
      Elts.insert(NewReg);

      // Copy Proto super-classes.
      for (unsigned i = 0, e = Proto->getSuperClasses().size(); i != e; ++i)
        NewReg->addSuperClass(Proto->getSuperClasses()[i]);

      // Copy Proto fields.
      for (unsigned i = 0, e = Proto->getValues().size(); i != e; ++i) {
        RecordVal RV = Proto->getValues()[i];

        // Skip existing fields, like NAME.
        if (NewReg->getValue(RV.getNameInit()))
          continue;

        StringRef Field = RV.getName();

        // Replace the sub-register list with Tuple.
        if (Field == "SubRegs")
          RV.setValue(ListInit::get(Tuple, RegisterRecTy));

        // Provide a blank AsmName. MC hacks are required anyway.
        if (Field == "AsmName")
          RV.setValue(BlankName);

        // CostPerUse is aggregated from all Tuple members.
        if (Field == "CostPerUse")
          RV.setValue(IntInit::get(CostPerUse));

        // Composite registers are always covered by sub-registers.
        if (Field == "CoveredBySubRegs")
          RV.setValue(BitInit::get(true));

        // Copy fields from the RegisterTuples def.
        if (Field == "SubRegIndices" ||
            Field == "CompositeIndices") {
          NewReg->addValue(*Def->getValue(Field));
          continue;
        }

        // Some fields get their default uninitialized value.
        if (Field == "DwarfNumbers" ||
            Field == "DwarfAlias" ||
            Field == "Aliases") {
          if (const RecordVal *DefRV = RegisterCl->getValue(Field))
            NewReg->addValue(*DefRV);
          continue;
        }

        // Everything else is copied from Proto.
        NewReg->addValue(RV);
      }
    }
  }
};
}

//===----------------------------------------------------------------------===//
//                            CodeGenRegisterClass
//===----------------------------------------------------------------------===//

CodeGenRegisterClass::CodeGenRegisterClass(CodeGenRegBank &RegBank, Record *R)
  : TheDef(R),
    Name(R->getName()),
    TopoSigs(RegBank.getNumTopoSigs()),
    EnumValue(-1) {
  // Rename anonymous register classes.
  if (R->getName().size() > 9 && R->getName()[9] == '.') {
    static unsigned AnonCounter = 0;
    R->setName("AnonRegClass_"+utostr(AnonCounter++));
  }

  std::vector<Record*> TypeList = R->getValueAsListOfDefs("RegTypes");
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    Record *Type = TypeList[i];
    if (!Type->isSubClassOf("ValueType"))
      PrintFatalError("RegTypes list member '" + Type->getName() +
        "' does not derive from the ValueType class!");
    VTs.push_back(getValueType(Type));
  }
  assert(!VTs.empty() && "RegisterClass must contain at least one ValueType!");

  // Allocation order 0 is the full set. AltOrders provides others.
  const SetTheory::RecVec *Elements = RegBank.getSets().expand(R);
  ListInit *AltOrders = R->getValueAsListInit("AltOrders");
  Orders.resize(1 + AltOrders->size());

  // Default allocation order always contains all registers.
  for (unsigned i = 0, e = Elements->size(); i != e; ++i) {
    Orders[0].push_back((*Elements)[i]);
    const CodeGenRegister *Reg = RegBank.getReg((*Elements)[i]);
    Members.insert(Reg);
    TopoSigs.set(Reg->getTopoSig());
  }

  // Alternative allocation orders may be subsets.
  SetTheory::RecSet Order;
  for (unsigned i = 0, e = AltOrders->size(); i != e; ++i) {
    RegBank.getSets().evaluate(AltOrders->getElement(i), Order, R->getLoc());
    Orders[1 + i].append(Order.begin(), Order.end());
    // Verify that all altorder members are regclass members.
    while (!Order.empty()) {
      CodeGenRegister *Reg = RegBank.getReg(Order.back());
      Order.pop_back();
      if (!contains(Reg))
        PrintFatalError(R->getLoc(), " AltOrder register " + Reg->getName() +
                      " is not a class member");
    }
  }

  // Allow targets to override the size in bits of the RegisterClass.
  unsigned Size = R->getValueAsInt("Size");

  Namespace = R->getValueAsString("Namespace");
  SpillSize = Size ? Size : EVT(VTs[0]).getSizeInBits();
  SpillAlignment = R->getValueAsInt("Alignment");
  CopyCost = R->getValueAsInt("CopyCost");
  Allocatable = R->getValueAsBit("isAllocatable");
  AltOrderSelect = R->getValueAsString("AltOrderSelect");
}

// Create an inferred register class that was missing from the .td files.
// Most properties will be inherited from the closest super-class after the
// class structure has been computed.
CodeGenRegisterClass::CodeGenRegisterClass(CodeGenRegBank &RegBank,
                                           StringRef Name, Key Props)
  : Members(*Props.Members),
    TheDef(0),
    Name(Name),
    TopoSigs(RegBank.getNumTopoSigs()),
    EnumValue(-1),
    SpillSize(Props.SpillSize),
    SpillAlignment(Props.SpillAlignment),
    CopyCost(0),
    Allocatable(true) {
  for (CodeGenRegister::Set::iterator I = Members.begin(), E = Members.end();
       I != E; ++I)
    TopoSigs.set((*I)->getTopoSig());
}

// Compute inherited propertied for a synthesized register class.
void CodeGenRegisterClass::inheritProperties(CodeGenRegBank &RegBank) {
  assert(!getDef() && "Only synthesized classes can inherit properties");
  assert(!SuperClasses.empty() && "Synthesized class without super class");

  // The last super-class is the smallest one.
  CodeGenRegisterClass &Super = *SuperClasses.back();

  // Most properties are copied directly.
  // Exceptions are members, size, and alignment
  Namespace = Super.Namespace;
  VTs = Super.VTs;
  CopyCost = Super.CopyCost;
  Allocatable = Super.Allocatable;
  AltOrderSelect = Super.AltOrderSelect;

  // Copy all allocation orders, filter out foreign registers from the larger
  // super-class.
  Orders.resize(Super.Orders.size());
  for (unsigned i = 0, ie = Super.Orders.size(); i != ie; ++i)
    for (unsigned j = 0, je = Super.Orders[i].size(); j != je; ++j)
      if (contains(RegBank.getReg(Super.Orders[i][j])))
        Orders[i].push_back(Super.Orders[i][j]);
}

bool CodeGenRegisterClass::contains(const CodeGenRegister *Reg) const {
  return Members.count(Reg);
}

namespace llvm {
  raw_ostream &operator<<(raw_ostream &OS, const CodeGenRegisterClass::Key &K) {
    OS << "{ S=" << K.SpillSize << ", A=" << K.SpillAlignment;
    for (CodeGenRegister::Set::const_iterator I = K.Members->begin(),
         E = K.Members->end(); I != E; ++I)
      OS << ", " << (*I)->getName();
    return OS << " }";
  }
}

// This is a simple lexicographical order that can be used to search for sets.
// It is not the same as the topological order provided by TopoOrderRC.
bool CodeGenRegisterClass::Key::
operator<(const CodeGenRegisterClass::Key &B) const {
  assert(Members && B.Members);
  if (*Members != *B.Members)
    return *Members < *B.Members;
  if (SpillSize != B.SpillSize)
    return SpillSize < B.SpillSize;
  return SpillAlignment < B.SpillAlignment;
}

// Returns true if RC is a strict subclass.
// RC is a sub-class of this class if it is a valid replacement for any
// instruction operand where a register of this classis required. It must
// satisfy these conditions:
//
// 1. All RC registers are also in this.
// 2. The RC spill size must not be smaller than our spill size.
// 3. RC spill alignment must be compatible with ours.
//
static bool testSubClass(const CodeGenRegisterClass *A,
                         const CodeGenRegisterClass *B) {
  return A->SpillAlignment && B->SpillAlignment % A->SpillAlignment == 0 &&
    A->SpillSize <= B->SpillSize &&
    std::includes(A->getMembers().begin(), A->getMembers().end(),
                  B->getMembers().begin(), B->getMembers().end(),
                  CodeGenRegister::Less());
}

/// Sorting predicate for register classes.  This provides a topological
/// ordering that arranges all register classes before their sub-classes.
///
/// Register classes with the same registers, spill size, and alignment form a
/// clique.  They will be ordered alphabetically.
///
static int TopoOrderRC(const void *PA, const void *PB) {
  const CodeGenRegisterClass *A = *(const CodeGenRegisterClass* const*)PA;
  const CodeGenRegisterClass *B = *(const CodeGenRegisterClass* const*)PB;
  if (A == B)
    return 0;

  // Order by ascending spill size.
  if (A->SpillSize < B->SpillSize)
    return -1;
  if (A->SpillSize > B->SpillSize)
    return 1;

  // Order by ascending spill alignment.
  if (A->SpillAlignment < B->SpillAlignment)
    return -1;
  if (A->SpillAlignment > B->SpillAlignment)
    return 1;

  // Order by descending set size.  Note that the classes' allocation order may
  // not have been computed yet.  The Members set is always vaild.
  if (A->getMembers().size() > B->getMembers().size())
    return -1;
  if (A->getMembers().size() < B->getMembers().size())
    return 1;

  // Finally order by name as a tie breaker.
  return StringRef(A->getName()).compare(B->getName());
}

std::string CodeGenRegisterClass::getQualifiedName() const {
  if (Namespace.empty())
    return getName();
  else
    return Namespace + "::" + getName();
}

// Compute sub-classes of all register classes.
// Assume the classes are ordered topologically.
void CodeGenRegisterClass::computeSubClasses(CodeGenRegBank &RegBank) {
  ArrayRef<CodeGenRegisterClass*> RegClasses = RegBank.getRegClasses();

  // Visit backwards so sub-classes are seen first.
  for (unsigned rci = RegClasses.size(); rci; --rci) {
    CodeGenRegisterClass &RC = *RegClasses[rci - 1];
    RC.SubClasses.resize(RegClasses.size());
    RC.SubClasses.set(RC.EnumValue);

    // Normally, all subclasses have IDs >= rci, unless RC is part of a clique.
    for (unsigned s = rci; s != RegClasses.size(); ++s) {
      if (RC.SubClasses.test(s))
        continue;
      CodeGenRegisterClass *SubRC = RegClasses[s];
      if (!testSubClass(&RC, SubRC))
        continue;
      // SubRC is a sub-class. Grap all its sub-classes so we won't have to
      // check them again.
      RC.SubClasses |= SubRC->SubClasses;
    }

    // Sweep up missed clique members.  They will be immediately preceding RC.
    for (unsigned s = rci - 1; s && testSubClass(&RC, RegClasses[s - 1]); --s)
      RC.SubClasses.set(s - 1);
  }

  // Compute the SuperClasses lists from the SubClasses vectors.
  for (unsigned rci = 0; rci != RegClasses.size(); ++rci) {
    const BitVector &SC = RegClasses[rci]->getSubClasses();
    for (int s = SC.find_first(); s >= 0; s = SC.find_next(s)) {
      if (unsigned(s) == rci)
        continue;
      RegClasses[s]->SuperClasses.push_back(RegClasses[rci]);
    }
  }

  // With the class hierarchy in place, let synthesized register classes inherit
  // properties from their closest super-class. The iteration order here can
  // propagate properties down multiple levels.
  for (unsigned rci = 0; rci != RegClasses.size(); ++rci)
    if (!RegClasses[rci]->getDef())
      RegClasses[rci]->inheritProperties(RegBank);
}

void
CodeGenRegisterClass::getSuperRegClasses(CodeGenSubRegIndex *SubIdx,
                                         BitVector &Out) const {
  DenseMap<CodeGenSubRegIndex*,
           SmallPtrSet<CodeGenRegisterClass*, 8> >::const_iterator
    FindI = SuperRegClasses.find(SubIdx);
  if (FindI == SuperRegClasses.end())
    return;
  for (SmallPtrSet<CodeGenRegisterClass*, 8>::const_iterator I =
       FindI->second.begin(), E = FindI->second.end(); I != E; ++I)
    Out.set((*I)->EnumValue);
}

// Populate a unique sorted list of units from a register set.
void CodeGenRegisterClass::buildRegUnitSet(
  std::vector<unsigned> &RegUnits) const {
  std::vector<unsigned> TmpUnits;
  for (RegUnitIterator UnitI(Members); UnitI.isValid(); ++UnitI)
    TmpUnits.push_back(*UnitI);
  std::sort(TmpUnits.begin(), TmpUnits.end());
  std::unique_copy(TmpUnits.begin(), TmpUnits.end(),
                   std::back_inserter(RegUnits));
}

//===----------------------------------------------------------------------===//
//                               CodeGenRegBank
//===----------------------------------------------------------------------===//

CodeGenRegBank::CodeGenRegBank(RecordKeeper &Records) {
  // Configure register Sets to understand register classes and tuples.
  Sets.addFieldExpander("RegisterClass", "MemberList");
  Sets.addFieldExpander("CalleeSavedRegs", "SaveList");
  Sets.addExpander("RegisterTuples", new TupleExpander());

  // Read in the user-defined (named) sub-register indices.
  // More indices will be synthesized later.
  std::vector<Record*> SRIs = Records.getAllDerivedDefinitions("SubRegIndex");
  std::sort(SRIs.begin(), SRIs.end(), LessRecord());
  for (unsigned i = 0, e = SRIs.size(); i != e; ++i)
    getSubRegIdx(SRIs[i]);
  // Build composite maps from ComposedOf fields.
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i)
    SubRegIndices[i]->updateComponents(*this);

  // Read in the register definitions.
  std::vector<Record*> Regs = Records.getAllDerivedDefinitions("Register");
  std::sort(Regs.begin(), Regs.end(), LessRecord());
  Registers.reserve(Regs.size());
  // Assign the enumeration values.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i)
    getReg(Regs[i]);

  // Expand tuples and number the new registers.
  std::vector<Record*> Tups =
    Records.getAllDerivedDefinitions("RegisterTuples");
  for (unsigned i = 0, e = Tups.size(); i != e; ++i) {
    const std::vector<Record*> *TupRegs = Sets.expand(Tups[i]);
    for (unsigned j = 0, je = TupRegs->size(); j != je; ++j)
      getReg((*TupRegs)[j]);
  }

  // Now all the registers are known. Build the object graph of explicit
  // register-register references.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    Registers[i]->buildObjectGraph(*this);

  // Compute register name map.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    RegistersByName.GetOrCreateValue(
                       Registers[i]->TheDef->getValueAsString("AsmName"),
                       Registers[i]);

  // Precompute all sub-register maps.
  // This will create Composite entries for all inferred sub-register indices.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    Registers[i]->computeSubRegs(*this);

  // Infer even more sub-registers by combining leading super-registers.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    if (Registers[i]->CoveredBySubRegs)
      Registers[i]->computeSecondarySubRegs(*this);

  // After the sub-register graph is complete, compute the topologically
  // ordered SuperRegs list.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i)
    Registers[i]->computeSuperRegs(*this);

  // Native register units are associated with a leaf register. They've all been
  // discovered now.
  NumNativeRegUnits = RegUnits.size();

  // Read in register class definitions.
  std::vector<Record*> RCs = Records.getAllDerivedDefinitions("RegisterClass");
  if (RCs.empty())
    PrintFatalError(std::string("No 'RegisterClass' subclasses defined!"));

  // Allocate user-defined register classes.
  RegClasses.reserve(RCs.size());
  for (unsigned i = 0, e = RCs.size(); i != e; ++i)
    addToMaps(new CodeGenRegisterClass(*this, RCs[i]));

  // Infer missing classes to create a full algebra.
  computeInferredRegisterClasses();

  // Order register classes topologically and assign enum values.
  array_pod_sort(RegClasses.begin(), RegClasses.end(), TopoOrderRC);
  for (unsigned i = 0, e = RegClasses.size(); i != e; ++i)
    RegClasses[i]->EnumValue = i;
  CodeGenRegisterClass::computeSubClasses(*this);
}

// Create a synthetic CodeGenSubRegIndex without a corresponding Record.
CodeGenSubRegIndex*
CodeGenRegBank::createSubRegIndex(StringRef Name, StringRef Namespace) {
  CodeGenSubRegIndex *Idx = new CodeGenSubRegIndex(Name, Namespace,
                                                   SubRegIndices.size() + 1);
  SubRegIndices.push_back(Idx);
  return Idx;
}

CodeGenSubRegIndex *CodeGenRegBank::getSubRegIdx(Record *Def) {
  CodeGenSubRegIndex *&Idx = Def2SubRegIdx[Def];
  if (Idx)
    return Idx;
  Idx = new CodeGenSubRegIndex(Def, SubRegIndices.size() + 1);
  SubRegIndices.push_back(Idx);
  return Idx;
}

CodeGenRegister *CodeGenRegBank::getReg(Record *Def) {
  CodeGenRegister *&Reg = Def2Reg[Def];
  if (Reg)
    return Reg;
  Reg = new CodeGenRegister(Def, Registers.size() + 1);
  Registers.push_back(Reg);
  return Reg;
}

void CodeGenRegBank::addToMaps(CodeGenRegisterClass *RC) {
  RegClasses.push_back(RC);

  if (Record *Def = RC->getDef())
    Def2RC.insert(std::make_pair(Def, RC));

  // Duplicate classes are rejected by insert().
  // That's OK, we only care about the properties handled by CGRC::Key.
  CodeGenRegisterClass::Key K(*RC);
  Key2RC.insert(std::make_pair(K, RC));
}

// Create a synthetic sub-class if it is missing.
CodeGenRegisterClass*
CodeGenRegBank::getOrCreateSubClass(const CodeGenRegisterClass *RC,
                                    const CodeGenRegister::Set *Members,
                                    StringRef Name) {
  // Synthetic sub-class has the same size and alignment as RC.
  CodeGenRegisterClass::Key K(Members, RC->SpillSize, RC->SpillAlignment);
  RCKeyMap::const_iterator FoundI = Key2RC.find(K);
  if (FoundI != Key2RC.end())
    return FoundI->second;

  // Sub-class doesn't exist, create a new one.
  CodeGenRegisterClass *NewRC = new CodeGenRegisterClass(*this, Name, K);
  addToMaps(NewRC);
  return NewRC;
}

CodeGenRegisterClass *CodeGenRegBank::getRegClass(Record *Def) {
  if (CodeGenRegisterClass *RC = Def2RC[Def])
    return RC;

  PrintFatalError(Def->getLoc(), "Not a known RegisterClass!");
}

CodeGenSubRegIndex*
CodeGenRegBank::getCompositeSubRegIndex(CodeGenSubRegIndex *A,
                                        CodeGenSubRegIndex *B) {
  // Look for an existing entry.
  CodeGenSubRegIndex *Comp = A->compose(B);
  if (Comp)
    return Comp;

  // None exists, synthesize one.
  std::string Name = A->getName() + "_then_" + B->getName();
  Comp = createSubRegIndex(Name, A->getNamespace());
  A->addComposite(B, Comp);
  return Comp;
}

CodeGenSubRegIndex *CodeGenRegBank::
getConcatSubRegIndex(const SmallVector<CodeGenSubRegIndex*, 8> &Parts) {
  assert(Parts.size() > 1 && "Need two parts to concatenate");

  // Look for an existing entry.
  CodeGenSubRegIndex *&Idx = ConcatIdx[Parts];
  if (Idx)
    return Idx;

  // None exists, synthesize one.
  std::string Name = Parts.front()->getName();
  for (unsigned i = 1, e = Parts.size(); i != e; ++i) {
    Name += '_';
    Name += Parts[i]->getName();
  }
  return Idx = createSubRegIndex(Name, Parts.front()->getNamespace());
}

void CodeGenRegBank::computeComposites() {
  // Keep track of TopoSigs visited. We only need to visit each TopoSig once,
  // and many registers will share TopoSigs on regular architectures.
  BitVector TopoSigs(getNumTopoSigs());

  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    CodeGenRegister *Reg1 = Registers[i];

    // Skip identical subreg structures already processed.
    if (TopoSigs.test(Reg1->getTopoSig()))
      continue;
    TopoSigs.set(Reg1->getTopoSig());

    const CodeGenRegister::SubRegMap &SRM1 = Reg1->getSubRegs();
    for (CodeGenRegister::SubRegMap::const_iterator i1 = SRM1.begin(),
         e1 = SRM1.end(); i1 != e1; ++i1) {
      CodeGenSubRegIndex *Idx1 = i1->first;
      CodeGenRegister *Reg2 = i1->second;
      // Ignore identity compositions.
      if (Reg1 == Reg2)
        continue;
      const CodeGenRegister::SubRegMap &SRM2 = Reg2->getSubRegs();
      // Try composing Idx1 with another SubRegIndex.
      for (CodeGenRegister::SubRegMap::const_iterator i2 = SRM2.begin(),
           e2 = SRM2.end(); i2 != e2; ++i2) {
        CodeGenSubRegIndex *Idx2 = i2->first;
        CodeGenRegister *Reg3 = i2->second;
        // Ignore identity compositions.
        if (Reg2 == Reg3)
          continue;
        // OK Reg1:IdxPair == Reg3. Find the index with Reg:Idx == Reg3.
        CodeGenSubRegIndex *Idx3 = Reg1->getSubRegIndex(Reg3);
        assert(Idx3 && "Sub-register doesn't have an index");

        // Conflicting composition? Emit a warning but allow it.
        if (CodeGenSubRegIndex *Prev = Idx1->addComposite(Idx2, Idx3))
          PrintWarning(Twine("SubRegIndex ") + Idx1->getQualifiedName() +
                       " and " + Idx2->getQualifiedName() +
                       " compose ambiguously as " + Prev->getQualifiedName() +
                       " or " + Idx3->getQualifiedName());
      }
    }
  }
}

// Compute lane masks. This is similar to register units, but at the
// sub-register index level. Each bit in the lane mask is like a register unit
// class, and two lane masks will have a bit in common if two sub-register
// indices overlap in some register.
//
// Conservatively share a lane mask bit if two sub-register indices overlap in
// some registers, but not in others. That shouldn't happen a lot.
void CodeGenRegBank::computeSubRegIndexLaneMasks() {
  // First assign individual bits to all the leaf indices.
  unsigned Bit = 0;
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i) {
    CodeGenSubRegIndex *Idx = SubRegIndices[i];
    if (Idx->getComposites().empty()) {
      Idx->LaneMask = 1u << Bit;
      // Share bit 31 in the unlikely case there are more than 32 leafs.
      if (Bit < 31) ++Bit;
    } else {
      Idx->LaneMask = 0;
    }
  }

  // FIXME: What if ad-hoc aliasing introduces overlaps that aren't represented
  // by the sub-register graph? This doesn't occur in any known targets.

  // Inherit lanes from composites.
  for (unsigned i = 0, e = SubRegIndices.size(); i != e; ++i)
    SubRegIndices[i]->computeLaneMask();
}

namespace {
// UberRegSet is a helper class for computeRegUnitWeights. Each UberRegSet is
// the transitive closure of the union of overlapping register
// classes. Together, the UberRegSets form a partition of the registers. If we
// consider overlapping register classes to be connected, then each UberRegSet
// is a set of connected components.
//
// An UberRegSet will likely be a horizontal slice of register names of
// the same width. Nontrivial subregisters should then be in a separate
// UberRegSet. But this property isn't required for valid computation of
// register unit weights.
//
// A Weight field caches the max per-register unit weight in each UberRegSet.
//
// A set of SingularDeterminants flags single units of some register in this set
// for which the unit weight equals the set weight. These units should not have
// their weight increased.
struct UberRegSet {
  CodeGenRegister::Set Regs;
  unsigned Weight;
  CodeGenRegister::RegUnitList SingularDeterminants;

  UberRegSet(): Weight(0) {}
};
} // namespace

// Partition registers into UberRegSets, where each set is the transitive
// closure of the union of overlapping register classes.
//
// UberRegSets[0] is a special non-allocatable set.
static void computeUberSets(std::vector<UberRegSet> &UberSets,
                            std::vector<UberRegSet*> &RegSets,
                            CodeGenRegBank &RegBank) {

  const std::vector<CodeGenRegister*> &Registers = RegBank.getRegisters();

  // The Register EnumValue is one greater than its index into Registers.
  assert(Registers.size() == Registers[Registers.size()-1]->EnumValue &&
         "register enum value mismatch");

  // For simplicitly make the SetID the same as EnumValue.
  IntEqClasses UberSetIDs(Registers.size()+1);
  std::set<unsigned> AllocatableRegs;
  for (unsigned i = 0, e = RegBank.getRegClasses().size(); i != e; ++i) {

    CodeGenRegisterClass *RegClass = RegBank.getRegClasses()[i];
    if (!RegClass->Allocatable)
      continue;

    const CodeGenRegister::Set &Regs = RegClass->getMembers();
    if (Regs.empty())
      continue;

    unsigned USetID = UberSetIDs.findLeader((*Regs.begin())->EnumValue);
    assert(USetID && "register number 0 is invalid");

    AllocatableRegs.insert((*Regs.begin())->EnumValue);
    for (CodeGenRegister::Set::const_iterator I = llvm::next(Regs.begin()),
           E = Regs.end(); I != E; ++I) {
      AllocatableRegs.insert((*I)->EnumValue);
      UberSetIDs.join(USetID, (*I)->EnumValue);
    }
  }
  // Combine non-allocatable regs.
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    unsigned RegNum = Registers[i]->EnumValue;
    if (AllocatableRegs.count(RegNum))
      continue;

    UberSetIDs.join(0, RegNum);
  }
  UberSetIDs.compress();

  // Make the first UberSet a special unallocatable set.
  unsigned ZeroID = UberSetIDs[0];

  // Insert Registers into the UberSets formed by union-find.
  // Do not resize after this.
  UberSets.resize(UberSetIDs.getNumClasses());
  for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
    const CodeGenRegister *Reg = Registers[i];
    unsigned USetID = UberSetIDs[Reg->EnumValue];
    if (!USetID)
      USetID = ZeroID;
    else if (USetID == ZeroID)
      USetID = 0;

    UberRegSet *USet = &UberSets[USetID];
    USet->Regs.insert(Reg);
    RegSets[i] = USet;
  }
}

// Recompute each UberSet weight after changing unit weights.
static void computeUberWeights(std::vector<UberRegSet> &UberSets,
                               CodeGenRegBank &RegBank) {
  // Skip the first unallocatable set.
  for (std::vector<UberRegSet>::iterator I = llvm::next(UberSets.begin()),
         E = UberSets.end(); I != E; ++I) {

    // Initialize all unit weights in this set, and remember the max units/reg.
    const CodeGenRegister *Reg = 0;
    unsigned MaxWeight = 0, Weight = 0;
    for (RegUnitIterator UnitI(I->Regs); UnitI.isValid(); ++UnitI) {
      if (Reg != UnitI.getReg()) {
        if (Weight > MaxWeight)
          MaxWeight = Weight;
        Reg = UnitI.getReg();
        Weight = 0;
      }
      unsigned UWeight = RegBank.getRegUnit(*UnitI).Weight;
      if (!UWeight) {
        UWeight = 1;
        RegBank.increaseRegUnitWeight(*UnitI, UWeight);
      }
      Weight += UWeight;
    }
    if (Weight > MaxWeight)
      MaxWeight = Weight;

    // Update the set weight.
    I->Weight = MaxWeight;

    // Find singular determinants.
    for (CodeGenRegister::Set::iterator RegI = I->Regs.begin(),
           RegE = I->Regs.end(); RegI != RegE; ++RegI) {
      if ((*RegI)->getRegUnits().size() == 1
          && (*RegI)->getWeight(RegBank) == I->Weight)
        mergeRegUnits(I->SingularDeterminants, (*RegI)->getRegUnits());
    }
  }
}

// normalizeWeight is a computeRegUnitWeights helper that adjusts the weight of
// a register and its subregisters so that they have the same weight as their
// UberSet. Self-recursion processes the subregister tree in postorder so
// subregisters are normalized first.
//
// Side effects:
// - creates new adopted register units
// - causes superregisters to inherit adopted units
// - increases the weight of "singular" units
// - induces recomputation of UberWeights.
static bool normalizeWeight(CodeGenRegister *Reg,
                            std::vector<UberRegSet> &UberSets,
                            std::vector<UberRegSet*> &RegSets,
                            std::set<unsigned> &NormalRegs,
                            CodeGenRegister::RegUnitList &NormalUnits,
                            CodeGenRegBank &RegBank) {
  bool Changed = false;
  if (!NormalRegs.insert(Reg->EnumValue).second)
    return Changed;

  const CodeGenRegister::SubRegMap &SRM = Reg->getSubRegs();
  for (CodeGenRegister::SubRegMap::const_iterator SRI = SRM.begin(),
         SRE = SRM.end(); SRI != SRE; ++SRI) {
    if (SRI->second == Reg)
      continue; // self-cycles happen

    Changed |= normalizeWeight(SRI->second, UberSets, RegSets,
                               NormalRegs, NormalUnits, RegBank);
  }
  // Postorder register normalization.

  // Inherit register units newly adopted by subregisters.
  if (Reg->inheritRegUnits(RegBank))
    computeUberWeights(UberSets, RegBank);

  // Check if this register is too skinny for its UberRegSet.
  UberRegSet *UberSet = RegSets[RegBank.getRegIndex(Reg)];

  unsigned RegWeight = Reg->getWeight(RegBank);
  if (UberSet->Weight > RegWeight) {
    // A register unit's weight can be adjusted only if it is the singular unit
    // for this register, has not been used to normalize a subregister's set,
    // and has not already been used to singularly determine this UberRegSet.
    unsigned AdjustUnit = Reg->getRegUnits().front();
    if (Reg->getRegUnits().size() != 1
        || hasRegUnit(NormalUnits, AdjustUnit)
        || hasRegUnit(UberSet->SingularDeterminants, AdjustUnit)) {
      // We don't have an adjustable unit, so adopt a new one.
      AdjustUnit = RegBank.newRegUnit(UberSet->Weight - RegWeight);
      Reg->adoptRegUnit(AdjustUnit);
      // Adopting a unit does not immediately require recomputing set weights.
    }
    else {
      // Adjust the existing single unit.
      RegBank.increaseRegUnitWeight(AdjustUnit, UberSet->Weight - RegWeight);
      // The unit may be shared among sets and registers within this set.
      computeUberWeights(UberSets, RegBank);
    }
    Changed = true;
  }

  // Mark these units normalized so superregisters can't change their weights.
  mergeRegUnits(NormalUnits, Reg->getRegUnits());

  return Changed;
}

// Compute a weight for each register unit created during getSubRegs.
//
// The goal is that two registers in the same class will have the same weight,
// where each register's weight is defined as sum of its units' weights.
void CodeGenRegBank::computeRegUnitWeights() {
  std::vector<UberRegSet> UberSets;
  std::vector<UberRegSet*> RegSets(Registers.size());
  computeUberSets(UberSets, RegSets, *this);
  // UberSets and RegSets are now immutable.

  computeUberWeights(UberSets, *this);

  // Iterate over each Register, normalizing the unit weights until reaching
  // a fix point.
  unsigned NumIters = 0;
  for (bool Changed = true; Changed; ++NumIters) {
    assert(NumIters <= NumNativeRegUnits && "Runaway register unit weights");
    Changed = false;
    for (unsigned i = 0, e = Registers.size(); i != e; ++i) {
      CodeGenRegister::RegUnitList NormalUnits;
      std::set<unsigned> NormalRegs;
      Changed |= normalizeWeight(Registers[i], UberSets, RegSets,
                                 NormalRegs, NormalUnits, *this);
    }
  }
}

// Find a set in UniqueSets with the same elements as Set.
// Return an iterator into UniqueSets.
static std::vector<RegUnitSet>::const_iterator
findRegUnitSet(const std::vector<RegUnitSet> &UniqueSets,
               const RegUnitSet &Set) {
  std::vector<RegUnitSet>::const_iterator
    I = UniqueSets.begin(), E = UniqueSets.end();
  for(;I != E; ++I) {
    if (I->Units == Set.Units)
      break;
  }
  return I;
}

// Return true if the RUSubSet is a subset of RUSuperSet.
static bool isRegUnitSubSet(const std::vector<unsigned> &RUSubSet,
                            const std::vector<unsigned> &RUSuperSet) {
  return std::includes(RUSuperSet.begin(), RUSuperSet.end(),
                       RUSubSet.begin(), RUSubSet.end());
}

// Iteratively prune unit sets.
void CodeGenRegBank::pruneUnitSets() {
  assert(RegClassUnitSets.empty() && "this invalidates RegClassUnitSets");

  // Form an equivalence class of UnitSets with no significant difference.
  std::vector<unsigned> SuperSetIDs;
  for (unsigned SubIdx = 0, EndIdx = RegUnitSets.size();
       SubIdx != EndIdx; ++SubIdx) {
    const RegUnitSet &SubSet = RegUnitSets[SubIdx];
    unsigned SuperIdx = 0;
    for (; SuperIdx != EndIdx; ++SuperIdx) {
      if (SuperIdx == SubIdx)
        continue;

      const RegUnitSet &SuperSet = RegUnitSets[SuperIdx];
      if (isRegUnitSubSet(SubSet.Units, SuperSet.Units)
          && (SubSet.Units.size() + 3 > SuperSet.Units.size())) {
        break;
      }
    }
    if (SuperIdx == EndIdx)
      SuperSetIDs.push_back(SubIdx);
  }
  // Populate PrunedUnitSets with each equivalence class's superset.
  std::vector<RegUnitSet> PrunedUnitSets(SuperSetIDs.size());
  for (unsigned i = 0, e = SuperSetIDs.size(); i != e; ++i) {
    unsigned SuperIdx = SuperSetIDs[i];
    PrunedUnitSets[i].Name = RegUnitSets[SuperIdx].Name;
    PrunedUnitSets[i].Units.swap(RegUnitSets[SuperIdx].Units);
  }
  RegUnitSets.swap(PrunedUnitSets);
}

// Create a RegUnitSet for each RegClass that contains all units in the class
// including adopted units that are necessary to model register pressure. Then
// iteratively compute RegUnitSets such that the union of any two overlapping
// RegUnitSets is repreresented.
//
// RegisterInfoEmitter will map each RegClass to its RegUnitClass and any
// RegUnitSet that is a superset of that RegUnitClass.
void CodeGenRegBank::computeRegUnitSets() {

  // Compute a unique RegUnitSet for each RegClass.
  const ArrayRef<CodeGenRegisterClass*> &RegClasses = getRegClasses();
  unsigned NumRegClasses = RegClasses.size();
  for (unsigned RCIdx = 0, RCEnd = NumRegClasses; RCIdx != RCEnd; ++RCIdx) {
    if (!RegClasses[RCIdx]->Allocatable)
      continue;

    // Speculatively grow the RegUnitSets to hold the new set.
    RegUnitSets.resize(RegUnitSets.size() + 1);
    RegUnitSets.back().Name = RegClasses[RCIdx]->getName();

    // Compute a sorted list of units in this class.
    RegClasses[RCIdx]->buildRegUnitSet(RegUnitSets.back().Units);

    // Find an existing RegUnitSet.
    std::vector<RegUnitSet>::const_iterator SetI =
      findRegUnitSet(RegUnitSets, RegUnitSets.back());
    if (SetI != llvm::prior(RegUnitSets.end()))
      RegUnitSets.pop_back();
  }

  // Iteratively prune unit sets.
  pruneUnitSets();

  // Iterate over all unit sets, including new ones added by this loop.
  unsigned NumRegUnitSubSets = RegUnitSets.size();
  for (unsigned Idx = 0, EndIdx = RegUnitSets.size(); Idx != EndIdx; ++Idx) {
    // In theory, this is combinatorial. In practice, it needs to be bounded
    // by a small number of sets for regpressure to be efficient.
    // If the assert is hit, we need to implement pruning.
    assert(Idx < (2*NumRegUnitSubSets) && "runaway unit set inference");

    // Compare new sets with all original classes.
    for (unsigned SearchIdx = (Idx >= NumRegUnitSubSets) ? 0 : Idx+1;
         SearchIdx != EndIdx; ++SearchIdx) {
      std::set<unsigned> Intersection;
      std::set_intersection(RegUnitSets[Idx].Units.begin(),
                            RegUnitSets[Idx].Units.end(),
                            RegUnitSets[SearchIdx].Units.begin(),
                            RegUnitSets[SearchIdx].Units.end(),
                            std::inserter(Intersection, Intersection.begin()));
      if (Intersection.empty())
        continue;

      // Speculatively grow the RegUnitSets to hold the new set.
      RegUnitSets.resize(RegUnitSets.size() + 1);
      RegUnitSets.back().Name =
        RegUnitSets[Idx].Name + "+" + RegUnitSets[SearchIdx].Name;

      std::set_union(RegUnitSets[Idx].Units.begin(),
                     RegUnitSets[Idx].Units.end(),
                     RegUnitSets[SearchIdx].Units.begin(),
                     RegUnitSets[SearchIdx].Units.end(),
                     std::inserter(RegUnitSets.back().Units,
                                   RegUnitSets.back().Units.begin()));

      // Find an existing RegUnitSet, or add the union to the unique sets.
      std::vector<RegUnitSet>::const_iterator SetI =
        findRegUnitSet(RegUnitSets, RegUnitSets.back());
      if (SetI != llvm::prior(RegUnitSets.end()))
        RegUnitSets.pop_back();
    }
  }

  // Iteratively prune unit sets after inferring supersets.
  pruneUnitSets();

  // For each register class, list the UnitSets that are supersets.
  RegClassUnitSets.resize(NumRegClasses);
  for (unsigned RCIdx = 0, RCEnd = NumRegClasses; RCIdx != RCEnd; ++RCIdx) {
    if (!RegClasses[RCIdx]->Allocatable)
      continue;

    // Recompute the sorted list of units in this class.
    std::vector<unsigned> RegUnits;
    RegClasses[RCIdx]->buildRegUnitSet(RegUnits);

    // Don't increase pressure for unallocatable regclasses.
    if (RegUnits.empty())
      continue;

    // Find all supersets.
    for (unsigned USIdx = 0, USEnd = RegUnitSets.size();
         USIdx != USEnd; ++USIdx) {
      if (isRegUnitSubSet(RegUnits, RegUnitSets[USIdx].Units))
        RegClassUnitSets[RCIdx].push_back(USIdx);
    }
    assert(!RegClassUnitSets[RCIdx].empty() && "missing unit set for regclass");
  }
}

void CodeGenRegBank::computeDerivedInfo() {
  computeComposites();
  computeSubRegIndexLaneMasks();

  // Compute a weight for each register unit created during getSubRegs.
  // This may create adopted register units (with unit # >= NumNativeRegUnits).
  computeRegUnitWeights();

  // Compute a unique set of RegUnitSets. One for each RegClass and inferred
  // supersets for the union of overlapping sets.
  computeRegUnitSets();
}

//
// Synthesize missing register class intersections.
//
// Make sure that sub-classes of RC exists such that getCommonSubClass(RC, X)
// returns a maximal register class for all X.
//
void CodeGenRegBank::inferCommonSubClass(CodeGenRegisterClass *RC) {
  for (unsigned rci = 0, rce = RegClasses.size(); rci != rce; ++rci) {
    CodeGenRegisterClass *RC1 = RC;
    CodeGenRegisterClass *RC2 = RegClasses[rci];
    if (RC1 == RC2)
      continue;

    // Compute the set intersection of RC1 and RC2.
    const CodeGenRegister::Set &Memb1 = RC1->getMembers();
    const CodeGenRegister::Set &Memb2 = RC2->getMembers();
    CodeGenRegister::Set Intersection;
    std::set_intersection(Memb1.begin(), Memb1.end(),
                          Memb2.begin(), Memb2.end(),
                          std::inserter(Intersection, Intersection.begin()),
                          CodeGenRegister::Less());

    // Skip disjoint class pairs.
    if (Intersection.empty())
      continue;

    // If RC1 and RC2 have different spill sizes or alignments, use the
    // larger size for sub-classing.  If they are equal, prefer RC1.
    if (RC2->SpillSize > RC1->SpillSize ||
        (RC2->SpillSize == RC1->SpillSize &&
         RC2->SpillAlignment > RC1->SpillAlignment))
      std::swap(RC1, RC2);

    getOrCreateSubClass(RC1, &Intersection,
                        RC1->getName() + "_and_" + RC2->getName());
  }
}

//
// Synthesize missing sub-classes for getSubClassWithSubReg().
//
// Make sure that the set of registers in RC with a given SubIdx sub-register
// form a register class.  Update RC->SubClassWithSubReg.
//
void CodeGenRegBank::inferSubClassWithSubReg(CodeGenRegisterClass *RC) {
  // Map SubRegIndex to set of registers in RC supporting that SubRegIndex.
  typedef std::map<CodeGenSubRegIndex*, CodeGenRegister::Set,
                   CodeGenSubRegIndex::Less> SubReg2SetMap;

  // Compute the set of registers supporting each SubRegIndex.
  SubReg2SetMap SRSets;
  for (CodeGenRegister::Set::const_iterator RI = RC->getMembers().begin(),
       RE = RC->getMembers().end(); RI != RE; ++RI) {
    const CodeGenRegister::SubRegMap &SRM = (*RI)->getSubRegs();
    for (CodeGenRegister::SubRegMap::const_iterator I = SRM.begin(),
         E = SRM.end(); I != E; ++I)
      SRSets[I->first].insert(*RI);
  }

  // Find matching classes for all SRSets entries.  Iterate in SubRegIndex
  // numerical order to visit synthetic indices last.
  for (unsigned sri = 0, sre = SubRegIndices.size(); sri != sre; ++sri) {
    CodeGenSubRegIndex *SubIdx = SubRegIndices[sri];
    SubReg2SetMap::const_iterator I = SRSets.find(SubIdx);
    // Unsupported SubRegIndex. Skip it.
    if (I == SRSets.end())
      continue;
    // In most cases, all RC registers support the SubRegIndex.
    if (I->second.size() == RC->getMembers().size()) {
      RC->setSubClassWithSubReg(SubIdx, RC);
      continue;
    }
    // This is a real subset.  See if we have a matching class.
    CodeGenRegisterClass *SubRC =
      getOrCreateSubClass(RC, &I->second,
                          RC->getName() + "_with_" + I->first->getName());
    RC->setSubClassWithSubReg(SubIdx, SubRC);
  }
}

//
// Synthesize missing sub-classes of RC for getMatchingSuperRegClass().
//
// Create sub-classes of RC such that getMatchingSuperRegClass(RC, SubIdx, X)
// has a maximal result for any SubIdx and any X >= FirstSubRegRC.
//

void CodeGenRegBank::inferMatchingSuperRegClass(CodeGenRegisterClass *RC,
                                                unsigned FirstSubRegRC) {
  SmallVector<std::pair<const CodeGenRegister*,
                        const CodeGenRegister*>, 16> SSPairs;
  BitVector TopoSigs(getNumTopoSigs());

  // Iterate in SubRegIndex numerical order to visit synthetic indices last.
  for (unsigned sri = 0, sre = SubRegIndices.size(); sri != sre; ++sri) {
    CodeGenSubRegIndex *SubIdx = SubRegIndices[sri];
    // Skip indexes that aren't fully supported by RC's registers. This was
    // computed by inferSubClassWithSubReg() above which should have been
    // called first.
    if (RC->getSubClassWithSubReg(SubIdx) != RC)
      continue;

    // Build list of (Super, Sub) pairs for this SubIdx.
    SSPairs.clear();
    TopoSigs.reset();
    for (CodeGenRegister::Set::const_iterator RI = RC->getMembers().begin(),
         RE = RC->getMembers().end(); RI != RE; ++RI) {
      const CodeGenRegister *Super = *RI;
      const CodeGenRegister *Sub = Super->getSubRegs().find(SubIdx)->second;
      assert(Sub && "Missing sub-register");
      SSPairs.push_back(std::make_pair(Super, Sub));
      TopoSigs.set(Sub->getTopoSig());
    }

    // Iterate over sub-register class candidates.  Ignore classes created by
    // this loop. They will never be useful.
    for (unsigned rci = FirstSubRegRC, rce = RegClasses.size(); rci != rce;
         ++rci) {
      CodeGenRegisterClass *SubRC = RegClasses[rci];
      // Topological shortcut: SubRC members have the wrong shape.
      if (!TopoSigs.anyCommon(SubRC->getTopoSigs()))
        continue;
      // Compute the subset of RC that maps into SubRC.
      CodeGenRegister::Set SubSet;
      for (unsigned i = 0, e = SSPairs.size(); i != e; ++i)
        if (SubRC->contains(SSPairs[i].second))
          SubSet.insert(SSPairs[i].first);
      if (SubSet.empty())
        continue;
      // RC injects completely into SubRC.
      if (SubSet.size() == SSPairs.size()) {
        SubRC->addSuperRegClass(SubIdx, RC);
        continue;
      }
      // Only a subset of RC maps into SubRC. Make sure it is represented by a
      // class.
      getOrCreateSubClass(RC, &SubSet, RC->getName() +
                          "_with_" + SubIdx->getName() +
                          "_in_" + SubRC->getName());
    }
  }
}


//
// Infer missing register classes.
//
void CodeGenRegBank::computeInferredRegisterClasses() {
  // When this function is called, the register classes have not been sorted
  // and assigned EnumValues yet.  That means getSubClasses(),
  // getSuperClasses(), and hasSubClass() functions are defunct.
  unsigned FirstNewRC = RegClasses.size();

  // Visit all register classes, including the ones being added by the loop.
  for (unsigned rci = 0; rci != RegClasses.size(); ++rci) {
    CodeGenRegisterClass *RC = RegClasses[rci];

    // Synthesize answers for getSubClassWithSubReg().
    inferSubClassWithSubReg(RC);

    // Synthesize answers for getCommonSubClass().
    inferCommonSubClass(RC);

    // Synthesize answers for getMatchingSuperRegClass().
    inferMatchingSuperRegClass(RC);

    // New register classes are created while this loop is running, and we need
    // to visit all of them.  I  particular, inferMatchingSuperRegClass needs
    // to match old super-register classes with sub-register classes created
    // after inferMatchingSuperRegClass was called.  At this point,
    // inferMatchingSuperRegClass has checked SuperRC = [0..rci] with SubRC =
    // [0..FirstNewRC).  We need to cover SubRC = [FirstNewRC..rci].
    if (rci + 1 == FirstNewRC) {
      unsigned NextNewRC = RegClasses.size();
      for (unsigned rci2 = 0; rci2 != FirstNewRC; ++rci2)
        inferMatchingSuperRegClass(RegClasses[rci2], FirstNewRC);
      FirstNewRC = NextNewRC;
    }
  }
}

/// getRegisterClassForRegister - Find the register class that contains the
/// specified physical register.  If the register is not in a register class,
/// return null. If the register is in multiple classes, and the classes have a
/// superset-subset relationship and the same set of types, return the
/// superclass.  Otherwise return null.
const CodeGenRegisterClass*
CodeGenRegBank::getRegClassForRegister(Record *R) {
  const CodeGenRegister *Reg = getReg(R);
  ArrayRef<CodeGenRegisterClass*> RCs = getRegClasses();
  const CodeGenRegisterClass *FoundRC = 0;
  for (unsigned i = 0, e = RCs.size(); i != e; ++i) {
    const CodeGenRegisterClass &RC = *RCs[i];
    if (!RC.contains(Reg))
      continue;

    // If this is the first class that contains the register,
    // make a note of it and go on to the next class.
    if (!FoundRC) {
      FoundRC = &RC;
      continue;
    }

    // If a register's classes have different types, return null.
    if (RC.getValueTypes() != FoundRC->getValueTypes())
      return 0;

    // Check to see if the previously found class that contains
    // the register is a subclass of the current class. If so,
    // prefer the superclass.
    if (RC.hasSubClass(FoundRC)) {
      FoundRC = &RC;
      continue;
    }

    // Check to see if the previously found class that contains
    // the register is a superclass of the current class. If so,
    // prefer the superclass.
    if (FoundRC->hasSubClass(&RC))
      continue;

    // Multiple classes, and neither is a superclass of the other.
    // Return null.
    return 0;
  }
  return FoundRC;
}

BitVector CodeGenRegBank::computeCoveredRegisters(ArrayRef<Record*> Regs) {
  SetVector<const CodeGenRegister*> Set;

  // First add Regs with all sub-registers.
  for (unsigned i = 0, e = Regs.size(); i != e; ++i) {
    CodeGenRegister *Reg = getReg(Regs[i]);
    if (Set.insert(Reg))
      // Reg is new, add all sub-registers.
      // The pre-ordering is not important here.
      Reg->addSubRegsPreOrder(Set, *this);
  }

  // Second, find all super-registers that are completely covered by the set.
  for (unsigned i = 0; i != Set.size(); ++i) {
    const CodeGenRegister::SuperRegList &SR = Set[i]->getSuperRegs();
    for (unsigned j = 0, e = SR.size(); j != e; ++j) {
      const CodeGenRegister *Super = SR[j];
      if (!Super->CoveredBySubRegs || Set.count(Super))
        continue;
      // This new super-register is covered by its sub-registers.
      bool AllSubsInSet = true;
      const CodeGenRegister::SubRegMap &SRM = Super->getSubRegs();
      for (CodeGenRegister::SubRegMap::const_iterator I = SRM.begin(),
             E = SRM.end(); I != E; ++I)
        if (!Set.count(I->second)) {
          AllSubsInSet = false;
          break;
        }
      // All sub-registers in Set, add Super as well.
      // We will visit Super later to recheck its super-registers.
      if (AllSubsInSet)
        Set.insert(Super);
    }
  }

  // Convert to BitVector.
  BitVector BV(Registers.size() + 1);
  for (unsigned i = 0, e = Set.size(); i != e; ++i)
    BV.set(Set[i]->EnumValue);
  return BV;
}
