//===-- llvm/ModuleProvider.h - Interface for module providers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an abstract interface for loading a module from some
// place.  This interface allows incremental or random access loading of
// functions from the file.  This is useful for applications like JIT compilers
// or interprocedural optimizers that do not need the entire program in memory
// at the same time.
//
//===----------------------------------------------------------------------===//

#ifndef MODULEPROVIDER_H
#define MODULEPROVIDER_H

#include <string>

namespace llvm {

class Function;
class Module;

class ModuleProvider {
protected:
  Module *TheModule;
  ModuleProvider();

public:
  virtual ~ModuleProvider();

  /// getModule - returns the module this provider is encapsulating.
  ///
  Module* getModule() { return TheModule; }

  /// materializeFunction - make sure the given function is fully read.  If the
  /// module is corrupt, this returns true and fills in the optional string
  /// with information about the problem.  If successful, this returns false.
  ///
  virtual bool materializeFunction(Function *F, std::string *ErrInfo = 0) = 0;

  /// dematerializeFunction - If the given function is read in, and if the
  /// module provider supports it, release the memory for the function, and set
  /// it up to be materialized lazily.  If the provider doesn't support this
  /// capability, this method is a noop.
  ///
  virtual void dematerializeFunction(Function *) {}
  
  /// materializeModule - make sure the entire Module has been completely read.
  /// On error, return null and fill in the error string if specified.
  ///
  virtual Module* materializeModule(std::string *ErrInfo = 0) = 0;

  /// releaseModule - no longer delete the Module* when provider is destroyed.
  /// On error, return null and fill in the error string if specified.
  ///
  virtual Module* releaseModule(std::string *ErrInfo = 0) {
    // Since we're losing control of this Module, we must hand it back complete
    if (!materializeModule(ErrInfo))
      return 0;
    Module *tempM = TheModule;
    TheModule = 0;
    return tempM;
  }
};


/// ExistingModuleProvider - Allow conversion from a fully materialized Module
/// into a ModuleProvider, allowing code that expects a ModuleProvider to work
/// if we just have a Module.  Note that the ModuleProvider takes ownership of
/// the Module specified.
struct ExistingModuleProvider : public ModuleProvider {
  explicit ExistingModuleProvider(Module *M) {
    TheModule = M;
  }
  bool materializeFunction(Function *, std::string * = 0) {
    return false;
  }
  Module* materializeModule(std::string * = 0) { return TheModule; }
};

} // End llvm namespace

#endif
