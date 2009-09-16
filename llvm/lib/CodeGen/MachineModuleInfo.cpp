//===-- llvm/CodeGen/MachineModuleInfo.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"

#include "llvm/Constants.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Streams.h"
using namespace llvm;
using namespace llvm::dwarf;

// Handle the Pass registration stuff necessary to use TargetData's.
static RegisterPass<MachineModuleInfo>
X("machinemoduleinfo", "Module Information");
char MachineModuleInfo::ID = 0;

//===----------------------------------------------------------------------===//
  
MachineModuleInfo::MachineModuleInfo()
: ImmutablePass(&ID)
, LabelIDList()
, FrameMoves()
, LandingPads()
, Personalities()
, CallsEHReturn(0)
, CallsUnwindInit(0)
, DbgInfoAvailable(false)
{
  // Always emit some info, by default "no personality" info.
  Personalities.push_back(NULL);
}
MachineModuleInfo::~MachineModuleInfo() {

}

/// doInitialization - Initialize the state for a new module.
///
bool MachineModuleInfo::doInitialization() {
  return false;
}

/// doFinalization - Tear down the state after completion of a module.
///
bool MachineModuleInfo::doFinalization() {
  return false;
}

/// BeginFunction - Begin gathering function meta information.
///
void MachineModuleInfo::BeginFunction(MachineFunction *MF) {
  // Coming soon.
}

/// EndFunction - Discard function meta information.
///
void MachineModuleInfo::EndFunction() {
  // Clean up frame info.
  FrameMoves.clear();
  
  // Clean up exception info.
  LandingPads.clear();
  TypeInfos.clear();
  FilterIds.clear();
  FilterEnds.clear();
  CallsEHReturn = 0;
  CallsUnwindInit = 0;
}

/// AnalyzeModule - Scan the module for global debug information.
///
void MachineModuleInfo::AnalyzeModule(Module &M) {
  // Insert functions in the llvm.used array (but not llvm.compiler.used) into
  // UsedFunctions.
  GlobalVariable *GV = M.getGlobalVariable("llvm.used");
  if (!GV || !GV->hasInitializer()) return;

  // Should be an array of 'i8*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (InitList == 0) return;

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (Function *F =
          dyn_cast<Function>(InitList->getOperand(i)->stripPointerCasts()))
      UsedFunctions.insert(F);
}

//===-EH-------------------------------------------------------------------===//

/// getOrCreateLandingPadInfo - Find or create an LandingPadInfo for the
/// specified MachineBasicBlock.
LandingPadInfo &MachineModuleInfo::getOrCreateLandingPadInfo
    (MachineBasicBlock *LandingPad) {
  unsigned N = LandingPads.size();
  for (unsigned i = 0; i < N; ++i) {
    LandingPadInfo &LP = LandingPads[i];
    if (LP.LandingPadBlock == LandingPad)
      return LP;
  }
  
  LandingPads.push_back(LandingPadInfo(LandingPad));
  return LandingPads[N];
}

/// addInvoke - Provide the begin and end labels of an invoke style call and
/// associate it with a try landing pad block.
void MachineModuleInfo::addInvoke(MachineBasicBlock *LandingPad,
                                  unsigned BeginLabel, unsigned EndLabel) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.BeginLabels.push_back(BeginLabel);
  LP.EndLabels.push_back(EndLabel);
}

/// addLandingPad - Provide the label of a try LandingPad block.
///
unsigned MachineModuleInfo::addLandingPad(MachineBasicBlock *LandingPad) {
  unsigned LandingPadLabel = NextLabelID();
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.LandingPadLabel = LandingPadLabel;  
  return LandingPadLabel;
}

/// addPersonality - Provide the personality function for the exception
/// information.
void MachineModuleInfo::addPersonality(MachineBasicBlock *LandingPad,
                                       Function *Personality) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.Personality = Personality;

  for (unsigned i = 0; i < Personalities.size(); ++i)
    if (Personalities[i] == Personality)
      return;
  
  // If this is the first personality we're adding go
  // ahead and add it at the beginning.
  if (Personalities[0] == NULL)
    Personalities[0] = Personality;
  else
    Personalities.push_back(Personality);
}

/// addCatchTypeInfo - Provide the catch typeinfo for a landing pad.
///
void MachineModuleInfo::addCatchTypeInfo(MachineBasicBlock *LandingPad,
                                        std::vector<GlobalVariable *> &TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  for (unsigned N = TyInfo.size(); N; --N)
    LP.TypeIds.push_back(getTypeIDFor(TyInfo[N - 1]));
}

/// addFilterTypeInfo - Provide the filter typeinfo for a landing pad.
///
void MachineModuleInfo::addFilterTypeInfo(MachineBasicBlock *LandingPad,
                                        std::vector<GlobalVariable *> &TyInfo) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  std::vector<unsigned> IdsInFilter(TyInfo.size());
  for (unsigned I = 0, E = TyInfo.size(); I != E; ++I)
    IdsInFilter[I] = getTypeIDFor(TyInfo[I]);
  LP.TypeIds.push_back(getFilterIDFor(IdsInFilter));
}

/// addCleanup - Add a cleanup action for a landing pad.
///
void MachineModuleInfo::addCleanup(MachineBasicBlock *LandingPad) {
  LandingPadInfo &LP = getOrCreateLandingPadInfo(LandingPad);
  LP.TypeIds.push_back(0);
}

/// TidyLandingPads - Remap landing pad labels and remove any deleted landing
/// pads.
void MachineModuleInfo::TidyLandingPads() {
  for (unsigned i = 0; i != LandingPads.size(); ) {
    LandingPadInfo &LandingPad = LandingPads[i];
    LandingPad.LandingPadLabel = MappedLabel(LandingPad.LandingPadLabel);

    // Special case: we *should* emit LPs with null LP MBB. This indicates
    // "nounwind" case.
    if (!LandingPad.LandingPadLabel && LandingPad.LandingPadBlock) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    for (unsigned j=0; j != LandingPads[i].BeginLabels.size(); ) {
      unsigned BeginLabel = MappedLabel(LandingPad.BeginLabels[j]);
      unsigned EndLabel = MappedLabel(LandingPad.EndLabels[j]);

      if (!BeginLabel || !EndLabel) {
        LandingPad.BeginLabels.erase(LandingPad.BeginLabels.begin() + j);
        LandingPad.EndLabels.erase(LandingPad.EndLabels.begin() + j);
        continue;
      }

      LandingPad.BeginLabels[j] = BeginLabel;
      LandingPad.EndLabels[j] = EndLabel;
      ++j;
    }

    // Remove landing pads with no try-ranges.
    if (LandingPads[i].BeginLabels.empty()) {
      LandingPads.erase(LandingPads.begin() + i);
      continue;
    }

    // If there is no landing pad, ensure that the list of typeids is empty.
    // If the only typeid is a cleanup, this is the same as having no typeids.
    if (!LandingPad.LandingPadBlock ||
        (LandingPad.TypeIds.size() == 1 && !LandingPad.TypeIds[0]))
      LandingPad.TypeIds.clear();

    ++i;
  }
}

/// getTypeIDFor - Return the type id for the specified typeinfo.  This is 
/// function wide.
unsigned MachineModuleInfo::getTypeIDFor(GlobalVariable *TI) {
  for (unsigned i = 0, N = TypeInfos.size(); i != N; ++i)
    if (TypeInfos[i] == TI) return i + 1;

  TypeInfos.push_back(TI);
  return TypeInfos.size();
}

/// getFilterIDFor - Return the filter id for the specified typeinfos.  This is
/// function wide.
int MachineModuleInfo::getFilterIDFor(std::vector<unsigned> &TyIds) {
  // If the new filter coincides with the tail of an existing filter, then
  // re-use the existing filter.  Folding filters more than this requires
  // re-ordering filters and/or their elements - probably not worth it.
  for (std::vector<unsigned>::iterator I = FilterEnds.begin(),
       E = FilterEnds.end(); I != E; ++I) {
    unsigned i = *I, j = TyIds.size();

    while (i && j)
      if (FilterIds[--i] != TyIds[--j])
        goto try_next;

    if (!j)
      // The new filter coincides with range [i, end) of the existing filter.
      return -(1 + i);

try_next:;
  }

  // Add the new filter.
  int FilterID = -(1 + FilterIds.size());
  FilterIds.reserve(FilterIds.size() + TyIds.size() + 1);
  for (unsigned I = 0, N = TyIds.size(); I != N; ++I)
    FilterIds.push_back(TyIds[I]);
  FilterEnds.push_back(FilterIds.size());
  FilterIds.push_back(0); // terminator
  return FilterID;
}

/// getPersonality - Return the personality function for the current function.
Function *MachineModuleInfo::getPersonality() const {
  // FIXME: Until PR1414 will be fixed, we're using 1 personality function per
  // function
  return !LandingPads.empty() ? LandingPads[0].Personality : NULL;
}

/// getPersonalityIndex - Return unique index for current personality
/// function. NULL/first personality function should always get zero index.
unsigned MachineModuleInfo::getPersonalityIndex() const {
  const Function* Personality = NULL;
  
  // Scan landing pads. If there is at least one non-NULL personality - use it.
  for (unsigned i = 0; i != LandingPads.size(); ++i)
    if (LandingPads[i].Personality) {
      Personality = LandingPads[i].Personality;
      break;
    }
  
  for (unsigned i = 0; i < Personalities.size(); ++i) {
    if (Personalities[i] == Personality)
      return i;
  }

  // This will happen if the current personality function is
  // in the zero index.
  return 0;
}

//===----------------------------------------------------------------------===//
/// DebugLabelFolding pass - This pass prunes out redundant labels.  This allows
/// a info consumer to determine if the range of two labels is empty, by seeing
/// if the labels map to the same reduced label.

namespace llvm {

struct DebugLabelFolder : public MachineFunctionPass {
  static char ID;
  DebugLabelFolder() : MachineFunctionPass(&ID) {}

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.setPreservesCFG();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  virtual bool runOnMachineFunction(MachineFunction &MF);
  virtual const char *getPassName() const { return "Label Folder"; }
};

char DebugLabelFolder::ID = 0;

bool DebugLabelFolder::runOnMachineFunction(MachineFunction &MF) {
  // Get machine module info.
  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  if (!MMI) return false;
  
  // Track if change is made.
  bool MadeChange = false;
  // No prior label to begin.
  unsigned PriorLabel = 0;
  
  // Iterate through basic blocks.
  for (MachineFunction::iterator BB = MF.begin(), E = MF.end();
       BB != E; ++BB) {
    // Iterate through instructions.
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      // Is it a label.
      if (I->isDebugLabel() && !MMI->isDbgLabelUsed(I->getOperand(0).getImm())){
        // The label ID # is always operand #0, an immediate.
        unsigned NextLabel = I->getOperand(0).getImm();
        
        // If there was an immediate prior label.
        if (PriorLabel) {
          // Remap the current label to prior label.
          MMI->RemapLabel(NextLabel, PriorLabel);
          // Delete the current label.
          I = BB->erase(I);
          // Indicate a change has been made.
          MadeChange = true;
          continue;
        } else {
          // Start a new round.
          PriorLabel = NextLabel;
        }
       } else {
        // No consecutive labels.
        PriorLabel = 0;
      }
      
      ++I;
    }
  }
  
  return MadeChange;
}

FunctionPass *createDebugLabelFoldingPass() { return new DebugLabelFolder(); }

}

