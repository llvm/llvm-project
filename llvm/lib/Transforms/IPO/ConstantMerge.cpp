//===- ConstantMerge.cpp - Merge duplicate global constants ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface to a pass that merges duplicate global
// constants together into a single constant that is shared.  This is useful
// because some passes (ie TraceValues) insert a lot of string constants into
// the program, regardless of whether or not an existing string is available.
//
// Algorithm: ConstantMerge is designed to build up a map of available constants
// and eliminate duplicates when it is initialized.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "constmerge"
#include "llvm/Transforms/IPO.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"
#include <map>
using namespace llvm;

STATISTIC(NumMerged, "Number of global constants merged");

namespace {
  struct VISIBILITY_HIDDEN ConstantMerge : public ModulePass {
    static char ID; // Pass identification, replacement for typeid
    ConstantMerge() : ModulePass(&ID) {}

    // run - For this pass, process all of the globals in the module,
    // eliminating duplicate constants.
    //
    bool runOnModule(Module &M);
  };
}

char ConstantMerge::ID = 0;
static RegisterPass<ConstantMerge>
X("constmerge", "Merge Duplicate Global Constants");

ModulePass *llvm::createConstantMergePass() { return new ConstantMerge(); }

bool ConstantMerge::runOnModule(Module &M) {
  // Map unique constant/section pairs to globals.  We don't want to merge
  // globals in different sections.
  std::map<std::pair<Constant*, std::string>, GlobalVariable*> CMap;

  // Replacements - This vector contains a list of replacements to perform.
  std::vector<std::pair<GlobalVariable*, GlobalVariable*> > Replacements;

  bool MadeChange = false;

  // Iterate constant merging while we are still making progress.  Merging two
  // constants together may allow us to merge other constants together if the
  // second level constants have initializers which point to the globals that
  // were just merged.
  while (1) {
    // First pass: identify all globals that can be merged together, filling in
    // the Replacements vector.  We cannot do the replacement in this pass
    // because doing so may cause initializers of other globals to be rewritten,
    // invalidating the Constant* pointers in CMap.
    //
    for (Module::global_iterator GVI = M.global_begin(), E = M.global_end();
         GVI != E; ) {
      GlobalVariable *GV = GVI++;
      
      // If this GV is dead, remove it.
      GV->removeDeadConstantUsers();
      if (GV->use_empty() && GV->hasLocalLinkage()) {
        GV->eraseFromParent();
        continue;
      }
      
      // Only process constants with initializers.
      if (GV->isConstant() && GV->hasInitializer()) {
        Constant *Init = GV->getInitializer();

        // Check to see if the initializer is already known.
        GlobalVariable *&Slot = CMap[std::make_pair(Init, GV->getSection())];

        if (Slot == 0) {    // Nope, add it to the map.
          Slot = GV;
        } else if (GV->hasLocalLinkage()) {    // Yup, this is a duplicate!
          // Make all uses of the duplicate constant use the canonical version.
          Replacements.push_back(std::make_pair(GV, Slot));
        }
      }
    }

    if (Replacements.empty())
      return MadeChange;
    CMap.clear();

    // Now that we have figured out which replacements must be made, do them all
    // now.  This avoid invalidating the pointers in CMap, which are unneeded
    // now.
    for (unsigned i = 0, e = Replacements.size(); i != e; ++i) {
      // Eliminate any uses of the dead global...
      Replacements[i].first->replaceAllUsesWith(Replacements[i].second);

      // Delete the global value from the module...
      M.getGlobalList().erase(Replacements[i].first);
    }

    NumMerged += Replacements.size();
    Replacements.clear();
  }
}
