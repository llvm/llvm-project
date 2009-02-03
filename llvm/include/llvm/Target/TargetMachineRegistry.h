//===-- Target/TargetMachineRegistry.h - Target Registration ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes two classes: the TargetMachineRegistry class, which allows
// tools to inspect all of registered targets, and the RegisterTarget class,
// which TargetMachine implementations should use to register themselves with
// the system.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINEREGISTRY_H
#define LLVM_TARGET_TARGETMACHINEREGISTRY_H

#include "llvm/Module.h"
#include "llvm/Support/Registry.h"

namespace llvm {
  class Module;
  class TargetMachine;

  struct TargetMachineRegistryEntry {
    const char *Name;
    const char *ShortDesc;
    TargetMachine *(*CtorFn)(const Module &, const std::string &);
    unsigned (*ModuleMatchQualityFn)(const Module &M);
    unsigned (*JITMatchQualityFn)();

  public:
    TargetMachineRegistryEntry(const char *N, const char *SD,
                      TargetMachine *(*CF)(const Module &, const std::string &),
                               unsigned (*MMF)(const Module &M),
                               unsigned (*JMF)())
      : Name(N), ShortDesc(SD), CtorFn(CF), ModuleMatchQualityFn(MMF),
        JITMatchQualityFn(JMF) {}
  };

  template<>
  class RegistryTraits<TargetMachine> {
  public:
    typedef TargetMachineRegistryEntry entry;

    static const char *nameof(const entry &Entry) { return Entry.Name; }
    static const char *descof(const entry &Entry) { return Entry.ShortDesc; }
  };

  struct TargetMachineRegistry : public Registry<TargetMachine> {
    /// getClosestStaticTargetForModule - Given an LLVM module, pick the best
    /// target that is compatible with the module.  If no close target can be
    /// found, this returns null and sets the Error string to a reason.
    static const entry *getClosestStaticTargetForModule(const Module &M,
                                                        std::string &Error);

    /// getClosestTargetForJIT - Pick the best target that is compatible with
    /// the current host.  If no close target can be found, this returns null
    /// and sets the Error string to a reason.
    static const entry *getClosestTargetForJIT(std::string &Error);

  };

  //===--------------------------------------------------------------------===//
  /// RegisterTarget - This class is used to make targets automatically register
  /// themselves with the tool they are linked.  Targets should define an
  /// instance of this and implement the static methods described in the
  /// TargetMachine comments.
  /// The type 'TargetMachineImpl' should provide a constructor with two
  /// parameters:
  /// - const Module& M: the module that is being compiled:
  /// - const std::string& FS: target-specific string describing target
  ///   flavour.

  template<class TargetMachineImpl>
  struct RegisterTarget {
    RegisterTarget(const char *Name, const char *ShortDesc)
      : Entry(Name, ShortDesc, &Allocator,
              &TargetMachineImpl::getModuleMatchQuality,
              &TargetMachineImpl::getJITMatchQuality),
        Node(Entry)
    {}

  private:
    TargetMachineRegistry::entry Entry;
    TargetMachineRegistry::node Node;

    static TargetMachine *Allocator(const Module &M, const std::string &FS) {
      return new TargetMachineImpl(M, FS);
    }
  };

}

#endif
