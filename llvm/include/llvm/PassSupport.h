//===- llvm/PassSupport.h - Pass Support code -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines stuff that is used to define and "use" Passes.  This file
// is automatically #included by Pass.h, so:
//
//           NO .CPP FILES SHOULD INCLUDE THIS FILE DIRECTLY
//
// Instead, #include Pass.h.
//
// This file defines Pass registration code and classes used for it.
//
//===----------------------------------------------------------------------===//

#if !defined(LLVM_PASS_H) || defined(LLVM_PASSSUPPORT_H)
#error "Do not include <PassSupport.h>; include <Pass.h> instead"
#endif

#ifndef LLVM_PASSSUPPORT_H
#define LLVM_PASSSUPPORT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/PassInfo.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Threading.h"
#include <functional>

namespace llvm {

class Pass;

#define INITIALIZE_PASS(passName, arg, name, cfg, analysis)                    \
  static void *initialize##passName##PassOnce(PassRegistry &Registry) {        \
    PassInfo *PI = new PassInfo(                                               \
        name, arg, &passName::ID,                                              \
        PassInfo::NormalCtor_t(callDefaultCtor<passName>), cfg, analysis);     \
    Registry.registerPass(*PI, true);                                          \
    return PI;                                                                 \
  }                                                                            \
  static llvm::once_flag Initialize##passName##PassFlag;                       \
  void llvm::initialize##passName##Pass(PassRegistry &Registry) {              \
    llvm::call_once(Initialize##passName##PassFlag,                            \
                    initialize##passName##PassOnce, std::ref(Registry));       \
  }

#define INITIALIZE_PASS_BEGIN(passName, arg, name, cfg, analysis)              \
  static void *initialize##passName##PassOnce(PassRegistry &Registry) {

#define INITIALIZE_PASS_DEPENDENCY(depName) initialize##depName##Pass(Registry);

#define INITIALIZE_PASS_END(passName, arg, name, cfg, analysis)                \
  PassInfo *PI = new PassInfo(                                                 \
      name, arg, &passName::ID,                                                \
      PassInfo::NormalCtor_t(callDefaultCtor<passName>), cfg, analysis);       \
  Registry.registerPass(*PI, true);                                            \
  return PI;                                                                   \
  }                                                                            \
  static llvm::once_flag Initialize##passName##PassFlag;                       \
  void llvm::initialize##passName##Pass(PassRegistry &Registry) {              \
    llvm::call_once(Initialize##passName##PassFlag,                            \
                    initialize##passName##PassOnce, std::ref(Registry));       \
  }

#define INITIALIZE_PASS_WITH_OPTIONS(PassName, Arg, Name, Cfg, Analysis)       \
  INITIALIZE_PASS_BEGIN(PassName, Arg, Name, Cfg, Analysis)                    \
  PassName::registerOptions();                                                 \
  INITIALIZE_PASS_END(PassName, Arg, Name, Cfg, Analysis)

#define INITIALIZE_PASS_WITH_OPTIONS_BEGIN(PassName, Arg, Name, Cfg, Analysis) \
  INITIALIZE_PASS_BEGIN(PassName, Arg, Name, Cfg, Analysis)                    \
  PassName::registerOptions();

template <
    class PassName,
    std::enable_if_t<std::is_default_constructible<PassName>{}, bool> = true>
Pass *callDefaultCtor() {
  return new PassName();
}

template <
    class PassName,
    std::enable_if_t<!std::is_default_constructible<PassName>{}, bool> = true>
Pass *callDefaultCtor() {
  // Some codegen passes should only be testable via
  // `llc -{start|stop}-{before|after}=<passname>`, not via `opt -<passname>`.
  report_fatal_error("target-specific codegen-only pass");
}

//===---------------------------------------------------------------------------
/// RegisterPass<t> template - This template class is used to notify the system
/// that a Pass is available for use, and registers it into the internal
/// database maintained by the PassManager.  Unless this template is used, opt,
/// for example will not be able to see the pass and attempts to create the pass
/// will fail. This template is used in the follow manner (at global scope, in
/// your .cpp file):
///
/// static RegisterPass<YourPassClassName> tmp("passopt", "My Pass Name");
///
/// This statement will cause your pass to be created by calling the default
/// constructor exposed by the pass.
template <typename passName> struct RegisterPass : public PassInfo {
  // Register Pass using default constructor...
  RegisterPass(StringRef PassArg, StringRef Name, bool CFGOnly = false,
               bool is_analysis = false)
      : PassInfo(Name, PassArg, &passName::ID,
                 PassInfo::NormalCtor_t(callDefaultCtor<passName>), CFGOnly,
                 is_analysis) {
    PassRegistry::getPassRegistry()->registerPass(*this);
  }
};

//===---------------------------------------------------------------------------
/// PassRegistrationListener class - This class is meant to be derived from by
/// clients that are interested in which passes get registered and unregistered
/// at runtime (which can be because of the RegisterPass constructors being run
/// as the program starts up, or may be because a shared object just got
/// loaded).
struct PassRegistrationListener {
  PassRegistrationListener() = default;
  virtual ~PassRegistrationListener() = default;

  /// Callback functions - These functions are invoked whenever a pass is loaded
  /// or removed from the current executable.
  virtual void passRegistered(const PassInfo *) {}

  /// enumeratePasses - Iterate over the registered passes, calling the
  /// passEnumerate callback on each PassInfo object.
  void enumeratePasses();

  /// passEnumerate - Callback function invoked when someone calls
  /// enumeratePasses on this PassRegistrationListener object.
  virtual void passEnumerate(const PassInfo *) {}
};

} // end namespace llvm

#endif // LLVM_PASSSUPPORT_H
