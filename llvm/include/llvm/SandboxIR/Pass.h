//===- Pass.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_PASS_H
#define LLVM_SANDBOXIR_PASS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class AAResults;
class ScalarEvolution;
class TargetTransformInfo;

namespace sandboxir {

class Function;
class Region;

class Analyses {
  AAResults *AA = nullptr;
  ScalarEvolution *SE = nullptr;
  TargetTransformInfo *TTI = nullptr;

  Analyses() = default;

public:
  Analyses(AAResults &AA, ScalarEvolution &SE, TargetTransformInfo &TTI)
      : AA(&AA), SE(&SE), TTI(&TTI) {}

public:
  AAResults &getAA() const { return *AA; }
  ScalarEvolution &getScalarEvolution() const { return *SE; }
  TargetTransformInfo &getTTI() const { return *TTI; }
  /// For use by unit tests.
  static Analyses emptyForTesting() { return Analyses(); }
};

/// The base class of a Sandbox IR Pass.
class Pass {
protected:
  /// The pass name. This is also used as a command-line flag and should not
  /// contain whitespaces.
  const std::string Name;

public:
  /// \p Name can't contain any spaces or start with '-'.
  Pass(StringRef Name) : Name(Name) {
    assert(!Name.contains(' ') &&
           "A pass name should not contain whitespaces!");
    assert(!Name.starts_with('-') && "A pass name should not start with '-'!");
  }
  virtual ~Pass() = default;
  /// \Returns the name of the pass.
  StringRef getName() const { return Name; }
#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS, const Pass &Pass) {
    Pass.print(OS);
    return OS;
  }
  virtual void print(raw_ostream &OS) const { OS << Name; }
  LLVM_ABI LLVM_DUMP_METHOD virtual void dump() const;
#endif
  /// Similar to print() but adds a newline. Used for testing.
  virtual void printPipeline(raw_ostream &OS) const { OS << Name << "\n"; }
};

/// A pass that runs on a sandbox::Function.
class FunctionPass : public Pass {
public:
  /// \p Name can't contain any spaces or start with '-'.
  FunctionPass(StringRef Name) : Pass(Name) {}
  /// \Returns true if it modifies \p F.
  virtual bool runOnFunction(Function &F, const Analyses &A) = 0;
};

/// A pass that runs on a sandbox::Region.
class RegionPass : public Pass {
public:
  /// \p Name can't contain any spaces or start with '-'.
  RegionPass(StringRef Name) : Pass(Name) {}
  /// \Returns true if it modifies \p R.
  virtual bool runOnRegion(Region &R, const Analyses &A) = 0;
};

class AuxPassArgsRegistry;
/// This represents an auxiliary pass argument. Its value defaults to false and
/// gets set by AuxPassArgsRegistry::parse(). It's value is cheap to access.
class AuxPassArg {
  unsigned ArgIdx = 0;
  AuxPassArgsRegistry *Registry = nullptr;

  /// Use AuxPassArgsRegistry::createArg() to create.
  AuxPassArg() = delete;
  AuxPassArg(unsigned ArgIdx, AuxPassArgsRegistry *Registry)
      : ArgIdx(ArgIdx), Registry(Registry) {}
  friend class AuxPassArgsRegistry; // For constructor.
  LLVM_ABI bool set(bool NewVal);

public:
  LLVM_ABI bool get() const;
  LLVM_ABI StringRef getFlagStr() const;
  operator bool() const { return get(); }
  bool operator=(bool NewVal) { return set(NewVal); }

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

/// A registry for the auxiliary pass argument arguments. This includes a
/// builder for the pass arguments.
class AuxPassArgsRegistry {
  struct Entry {
    Entry(StringRef FlagStr, bool Val) : FlagStr(FlagStr), Val(Val) {}
    StringRef FlagStr;
    bool Val;
  };
  SmallVector<Entry> Entries;
  friend class AuxPassArg; // For Entries

  /// Linear-time access to the entry for \p Flag. \returns nullptr if \p Flag
  /// is not registered.
  Entry *getEntry(StringRef Flag);

public:
  /// Builder for argument with \p Flag.
  LLVM_ABI AuxPassArg createArg(StringRef Flag);
  /// Parse the aux argument string \p ArgsStr and set the values.
  LLVM_ABI void parse(StringRef ArgsStr);

#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

} // namespace sandboxir
} // namespace llvm

#endif // LLVM_SANDBOXIR_PASS_H
