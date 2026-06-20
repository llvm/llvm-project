//===--- PluginRegistry.cpp - LLVM Advisor -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Loads external plugins via LLVM's DynamicLibrary and resolves the stable C
// ABI symbols so plugins can register capabilities at runtime.
//
//===----------------------------------------------------------------------===//

#include "Capability/PluginRegistry.h"
#include "Capability/CapabilityRegistry.h"
#include "Capability/PluginRunner.h"
#include "Utils/Hashing.h"

using namespace llvm;
using namespace llvm::advisor;

Error PluginRegistry::load(StringRef Path) {
  if (Loaded.contains(Path))
    return Error::success();
  std::string Err;
  std::string PathStr = Path.str();
  sys::DynamicLibrary Library =
      sys::DynamicLibrary::getPermanentLibrary(PathStr.c_str(), &Err);
  if (!Library.isValid())
    return createStringError(inconvertibleErrorCode(), Twine(Err));

  Plugin P;
  P.Path = Path.str();
  P.Handle = Library;
  P.RegisterFn =
      Library.getAddressOfSymbol("llvm_advisor_register_capabilities");
  P.RunFn = Library.getAddressOfSymbol("llvm_advisor_run_capability");
  P.FreeFn = Library.getAddressOfSymbol("llvm_advisor_free_result");

  if (!P.RegisterFn || !P.RunFn) {
    return createStringError(
        inconvertibleErrorCode(),
        Twine("plugin ") + Path +
            " missing required symbols (register_capabilities=" +
            (P.RegisterFn ? "ok" : "missing") +
            ", run_capability=" + (P.RunFn ? "ok" : "missing") + ")");
  }

  Plugins.push_back(std::move(P));
  Loaded.insert(Path);
  return Error::success();
}

Error PluginRegistry::loadVerified(StringRef Path, StringRef BLAKE3) {
  Expected<std::string> Digest = hashFile(Path);
  if (!Digest)
    return Digest.takeError();
  if (!BLAKE3.empty() && *Digest != BLAKE3)
    return createStringError(inconvertibleErrorCode(),
                             Twine("plugin hash mismatch for ") + Path);
  return load(Path);
}

Error PluginRegistry::registerPlugins(CapabilityRegistry &Registry) {
  for (const Plugin &P : Plugins) {
    auto *RegisterFn =
        reinterpret_cast<decltype(&llvm_advisor_register_capabilities)>(
            P.RegisterFn);
    auto *RunFn =
        reinterpret_cast<decltype(&llvm_advisor_run_capability)>(P.RunFn);
    auto *FreeFn =
        reinterpret_cast<decltype(&llvm_advisor_free_result)>(P.FreeFn);

    AdvisorCapabilitySpec *Specs = nullptr;
    int Count = 0;
    RegisterFn(&Specs, &Count);

    if (!Specs || Count <= 0)
      continue;

    for (int I = 0; I < Count; ++I) {
      const AdvisorCapabilitySpec &CS = Specs[I];
      if (!CS.capability_id)
        continue;

      CapabilitySpec Spec;
      Spec.ID = CS.capability_id;
      Spec.Version = CS.version ? CS.version : "1";
      Spec.Description = CS.description ? CS.description : "";
      Spec.ExecutionMode = CS.execution_mode ? CS.execution_mode : "library";
      Spec.CostClass = CS.cost_class ? CS.cost_class : "moderate";
      Spec.Readiness = CS.readiness_level ? CS.readiness_level : "L1";
      Spec.Runner = "plugin";

      if (CS.depends_on) {
        for (const char **Ptr = CS.depends_on; *Ptr; ++Ptr)
          Spec.Dependencies.push_back(*Ptr);
      }
      if (CS.required_inputs) {
        for (const char **Ptr = CS.required_inputs; *Ptr; ++Ptr)
          Spec.RequiredInputs.push_back(*Ptr);
      }

      if (Error Err = Registry.addSpec(Spec))
        return Err;
      if (Error Err = Registry.addRunner(
              std::make_unique<PluginRunner>(Spec.ID, RunFn, FreeFn)))
        return Err;
    }
  }
  return Error::success();
}
