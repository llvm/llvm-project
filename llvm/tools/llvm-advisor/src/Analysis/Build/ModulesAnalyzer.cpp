//===--- ModulesAnalyzer.cpp - LLVM Advisor ------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Build/ModulesAnalyzer.h"

namespace llvm::advisor {

// Module-related flags we recognize.
static constexpr StringLiteral ModuleFlags[] = {
    "-fmodules",          "-fmodules-ts",
    "-fmodule-name=",     "-fmodules-cache-path=",
    "-fimplicit-modules", "-fno-implicit-modules",
    "-fmodule-map-file=", "-fprebuilt-module-path=",
    "-fmodule-file=",     "-fcxx-modules",
};

Expected<std::unique_ptr<CapabilityResult>>
ModulesAnalyzer::run(const CapabilityContext &Context) {
  bool HasModules = false;
  json::Array FoundFlags;

  for (const std::string &Arg : Context.Unit.Arguments) {
    StringRef S(Arg);
    for (StringLiteral Flag : ModuleFlags) {
      if (S == Flag || S.starts_with(Flag)) {
        HasModules = true;
        FoundFlags.push_back(Arg);
        break;
      }
    }
  }

  // Detect C++20 named module import (-std=c++20 + presence of .cppm files
  // or -fmodule-file).
  bool HasCxx20Modules = false;
  bool HasStdCxx20 = false;
  for (const std::string &Arg : Context.Unit.Arguments) {
    StringRef S(Arg);
    if (S == "-std=c++20" || S == "-std=c++2a" || S == "-std=c++23")
      HasStdCxx20 = true;
    if (S.starts_with("-fmodule-file=") || S.ends_with(".cppm") ||
        S.ends_with(".ixx"))
      HasCxx20Modules = true;
  }
  if (HasStdCxx20 && HasCxx20Modules)
    HasModules = true;

  json::Value Result = json::Object{
      {"has_modules", HasModules},
      {"has_cxx20_modules", HasCxx20Modules},
      {"flags", std::move(FoundFlags)},
  };
  return std::make_unique<JSONCapabilityResult>(std::move(Result));
}

} // namespace llvm::advisor
