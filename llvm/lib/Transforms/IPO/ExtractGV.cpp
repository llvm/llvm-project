//===-- ExtractGV.cpp - Global Value extraction pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass extracts global values
//
//===----------------------------------------------------------------------===//

#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Constants.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
using namespace llvm;

namespace {
  /// @brief A pass to extract specific functions and their dependencies.
  class VISIBILITY_HIDDEN GVExtractorPass : public ModulePass {
    std::vector<GlobalValue*> Named;
    bool deleteStuff;
    bool reLink;
  public:
    static char ID; // Pass identification, replacement for typeid

    /// FunctionExtractorPass - If deleteFn is true, this pass deletes as the
    /// specified function. Otherwise, it deletes as much of the module as
    /// possible, except for the function specified.
    ///
    explicit GVExtractorPass(std::vector<GlobalValue*>& GVs, bool deleteS = true,
                             bool relinkCallees = false)
      : ModulePass(&ID), Named(GVs), deleteStuff(deleteS),
        reLink(relinkCallees) {}

    bool runOnModule(Module &M) {
      if (Named.size() == 0) {
        return false;  // Nothing to extract
      }
      
      if (deleteStuff)
        return deleteGV();
      M.setModuleInlineAsm("");
      return isolateGV(M);
    }

    bool deleteGV() {
      for (std::vector<GlobalValue*>::iterator GI = Named.begin(), 
             GE = Named.end(); GI != GE; ++GI) {
        if (Function* NamedFunc = dyn_cast<Function>(*GI)) {
         // If we're in relinking mode, set linkage of all internal callees to
         // external. This will allow us extract function, and then - link
         // everything together
         if (reLink) {
           for (Function::iterator B = NamedFunc->begin(), BE = NamedFunc->end();
                B != BE; ++B) {
             for (BasicBlock::iterator I = B->begin(), E = B->end();
                  I != E; ++I) {
               if (CallInst* callInst = dyn_cast<CallInst>(&*I)) {
                 Function* Callee = callInst->getCalledFunction();
                 if (Callee && Callee->hasLocalLinkage())
                   Callee->setLinkage(GlobalValue::ExternalLinkage);
               }
             }
           }
         }
         
         NamedFunc->setLinkage(GlobalValue::ExternalLinkage);
         NamedFunc->deleteBody();
         assert(NamedFunc->isDeclaration() && "This didn't make the function external!");
       } else {
          if (!(*GI)->isDeclaration()) {
            cast<GlobalVariable>(*GI)->setInitializer(0);  //clear the initializer
            (*GI)->setLinkage(GlobalValue::ExternalLinkage);
          }
        }
      }
      return true;
    }

    bool isolateGV(Module &M) {
      // Mark all globals internal
      // FIXME: what should we do with private linkage?
      for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E; ++I)
        if (!I->isDeclaration()) {
          I->setLinkage(GlobalValue::InternalLinkage);
        }
      for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
        if (!I->isDeclaration()) {
          I->setLinkage(GlobalValue::InternalLinkage);
        }

      // Make sure our result is globally accessible...
      // by putting them in the used array
      {
        std::vector<Constant *> AUGs;
        const Type *SBP= PointerType::getUnqual(Type::Int8Ty);
        for (std::vector<GlobalValue*>::iterator GI = Named.begin(), 
               GE = Named.end(); GI != GE; ++GI) {
          (*GI)->setLinkage(GlobalValue::ExternalLinkage);
          AUGs.push_back(ConstantExpr::getBitCast(*GI, SBP));
        }
        ArrayType *AT = ArrayType::get(SBP, AUGs.size());
        Constant *Init = ConstantArray::get(AT, AUGs);
        GlobalValue *gv = new GlobalVariable(AT, false, 
                                             GlobalValue::AppendingLinkage, 
                                             Init, "llvm.used", &M);
        gv->setSection("llvm.metadata");
      }

      // All of the functions may be used by global variables or the named
      // globals.  Loop through them and create a new, external functions that
      // can be "used", instead of ones with bodies.
      std::vector<Function*> NewFunctions;

      Function *Last = --M.end();  // Figure out where the last real fn is.

      for (Module::iterator I = M.begin(); ; ++I) {
        if (std::find(Named.begin(), Named.end(), &*I) == Named.end()) {
          Function *New = Function::Create(I->getFunctionType(),
                                           GlobalValue::ExternalLinkage);
          New->copyAttributesFrom(I);

          // If it's not the named function, delete the body of the function
          I->dropAllReferences();

          M.getFunctionList().push_back(New);
          NewFunctions.push_back(New);
          New->takeName(I);
        }

        if (&*I == Last) break;  // Stop after processing the last function
      }

      // Now that we have replacements all set up, loop through the module,
      // deleting the old functions, replacing them with the newly created
      // functions.
      if (!NewFunctions.empty()) {
        unsigned FuncNum = 0;
        Module::iterator I = M.begin();
        do {
          if (std::find(Named.begin(), Named.end(), &*I) == Named.end()) {
            // Make everything that uses the old function use the new dummy fn
            I->replaceAllUsesWith(NewFunctions[FuncNum++]);

            Function *Old = I;
            ++I;  // Move the iterator to the new function

            // Delete the old function!
            M.getFunctionList().erase(Old);

          } else {
            ++I;  // Skip the function we are extracting
          }
        } while (&*I != NewFunctions[0]);
      }

      return true;
    }
  };

  char GVExtractorPass::ID = 0;
}

ModulePass *llvm::createGVExtractionPass(std::vector<GlobalValue*>& GVs, 
                                         bool deleteFn, bool relinkCallees) {
  return new GVExtractorPass(GVs, deleteFn, relinkCallees);
}
