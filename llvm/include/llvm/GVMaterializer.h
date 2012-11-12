//===-- llvm/GVMaterializer.h - Interface for GV materializers --*- C++ -*-===//
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

#ifndef GVMATERIALIZER_H
#define GVMATERIALIZER_H

#include <string>

namespace llvm {

class Function;
class GlobalValue;
class Module;

class GVMaterializer {
protected:
  GVMaterializer() {}

public:
  virtual ~GVMaterializer();

  /// isMaterializable - True if GV can be materialized from whatever backing
  /// store this GVMaterializer uses and has not been materialized yet.
  virtual bool isMaterializable(const GlobalValue *GV) const = 0;

  /// isDematerializable - True if GV has been materialized and can be
  /// dematerialized back to whatever backing store this GVMaterializer uses.
  virtual bool isDematerializable(const GlobalValue *GV) const = 0;

  /// Materialize - make sure the given GlobalValue is fully read.  If the
  /// module is corrupt, this returns true and fills in the optional string with
  /// information about the problem.  If successful, this returns false.
  ///
  virtual bool Materialize(GlobalValue *GV, std::string *ErrInfo = 0) = 0;

  /// Dematerialize - If the given GlobalValue is read in, and if the
  /// GVMaterializer supports it, release the memory for the GV, and set it up
  /// to be materialized lazily.  If the Materializer doesn't support this
  /// capability, this method is a noop.
  ///
  virtual void Dematerialize(GlobalValue *) {}

  /// MaterializeModule - make sure the entire Module has been completely read.
  /// On error, this returns true and fills in the optional string with
  /// information about the problem.  If successful, this returns false.
  ///
  virtual bool MaterializeModule(Module *M, std::string *ErrInfo = 0) = 0;
};

} // End llvm namespace

#endif
