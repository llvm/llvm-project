//===- llvm/PassManager.h - Container for Passes ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManager class.  This class is used to hold,
// maintain, and optimize execution of Passes.  The PassManager class ensures
// that analysis results are available before a pass runs, and that Pass's are
// destroyed when the PassManager is destroyed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSMANAGER_H
#define LLVM_PASSMANAGER_H

#include "llvm/Pass.h"

namespace llvm {

class Pass;
class ModulePass;
class Module;
class ModuleProvider;

class PassManagerImpl;
class FunctionPassManagerImpl;

/// PassManagerBase - An abstract interface to allow code to add passes to
/// a pass manager without having to hard-code what kind of pass manager
/// it is.
class PassManagerBase {
public:
  virtual ~PassManagerBase();

  /// add - Add a pass to the queue of passes to run.  This passes ownership of
  /// the Pass to the PassManager.  When the PassManager is destroyed, the pass
  /// will be destroyed as well, so there is no need to delete the pass.  This
  /// implies that all passes MUST be allocated with 'new'.
  virtual void add(Pass *P) = 0;
};

/// PassManager manages ModulePassManagers
class PassManager : public PassManagerBase {
public:

  PassManager();
  ~PassManager();

  /// add - Add a pass to the queue of passes to run.  This passes ownership of
  /// the Pass to the PassManager.  When the PassManager is destroyed, the pass
  /// will be destroyed as well, so there is no need to delete the pass.  This
  /// implies that all passes MUST be allocated with 'new'.
  void add(Pass *P);
 
  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool run(Module &M);

private:

  /// PassManagerImpl_New is the actual class. PassManager is just the 
  /// wraper to publish simple pass manager interface
  PassManagerImpl *PM;
};

/// FunctionPassManager manages FunctionPasses and BasicBlockPassManagers.
class FunctionPassManager : public PassManagerBase {
public:
  /// FunctionPassManager ctor - This initializes the pass manager.  It needs,
  /// but does not take ownership of, the specified module provider.
  explicit FunctionPassManager(ModuleProvider *P);
  ~FunctionPassManager();
 
  /// add - Add a pass to the queue of passes to run.  This passes
  /// ownership of the Pass to the PassManager.  When the
  /// PassManager_X is destroyed, the pass will be destroyed as well, so
  /// there is no need to delete the pass. (TODO delete passes.)
  /// This implies that all passes MUST be allocated with 'new'.
  void add(Pass *P);

  /// run - Execute all of the passes scheduled for execution.  Keep
  /// track of whether any of the passes modifies the function, and if
  /// so, return true.
  ///
  bool run(Function &F);
  
  /// doInitialization - Run all of the initializers for the function passes.
  ///
  bool doInitialization();
  
  /// doFinalization - Run all of the finalizers for the function passes.
  ///
  bool doFinalization();
  
  /// getModuleProvider - Return the module provider that this passmanager is
  /// currently using.  This is the module provider that it uses when a function
  /// is optimized that is non-resident in the module.
  ModuleProvider *getModuleProvider() const { return MP; }
  void setModuleProvider(ModuleProvider *NewMP) { MP = NewMP; }

private:
  FunctionPassManagerImpl *FPM;
  ModuleProvider *MP;
};

} // End llvm namespace

#endif
