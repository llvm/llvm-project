//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Plugins/PassPlugin.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

using namespace llvm;

Expected<PassPlugin> PassPlugin::load(StringRef Filename) {
  std::string ErrMsg;
  auto Library =
      sys::DynamicLibrary::getPermanentLibrary(Filename.str().c_str(), &ErrMsg);
  if (!Library.isValid())
    return make_error<StringError>(Twine("Could not load library '") +
                                       Filename + "': " + ErrMsg,
                                   inconvertibleErrorCode());

  PassPlugin P{Filename.str(), Library};

  // llvmGetPassPluginInfo should be resolved to the definition from the plugin
  // we are currently loading.
  intptr_t getDetailsFn =
      (intptr_t)Library.getAddressOfSymbol("llvmGetPassPluginInfo");

  if (!getDetailsFn)
    // If the symbol isn't found, this is probably a legacy plugin, which is an
    // error
    return make_error<StringError>(Twine("Plugin entry point not found in '") +
                                       Filename + "'. Is this a legacy plugin?",
                                   inconvertibleErrorCode());

  P.Info = reinterpret_cast<decltype(llvmGetPassPluginInfo) *>(getDetailsFn)();

  if (P.Info.APIVersion != LLVM_PLUGIN_API_VERSION)
    return make_error<StringError>(
        Twine("Wrong API version on plugin '") + Filename + "'. Got version " +
            Twine(P.Info.APIVersion) + ", supported version is " +
            Twine(LLVM_PLUGIN_API_VERSION) + ".",
        inconvertibleErrorCode());

  return P;
}

Error PassPlugin::passArguments(ArrayRef<PassPlugin> Plugins,
                                ArrayRef<std::string> Args) {
  // If two plugins have the same name, the last one receives the arguments.
  DenseMap<StringRef, unsigned> Index;
  for (auto [I, P] : enumerate(Plugins))
    Index[P.getPluginName()] = I;
  // The argument follows the first comma, so it is NUL-terminated.
  SmallVector<SmallVector<const char *, 0>, 0> PluginArgs(Plugins.size());
  for (const std::string &Arg : Args) {
    auto [Name, Rest] = StringRef(Arg).split(',');
    if (!Rest.data())
      return createStringError("expected <plugin>,<arg> in -plugin-arg=" + Arg);
    auto It = Index.find(Name);
    if (It == Index.end())
      return createStringError("no pass plugin named '" + Name +
                               "' is loaded, in -plugin-arg=" + Arg);
    PluginArgs[It->second].push_back(Rest.data());
  }
  for (auto [P, PArgs] : zip_equal(Plugins, PluginArgs)) {
    if (PArgs.empty())
      continue;
    if (!P.Info.ParseArguments)
      return createStringError("pass plugin '" + P.getPluginName() +
                               "' does not accept arguments");
    if (Error E = P.Info.ParseArguments(PArgs))
      return E;
  }
  return Error::success();
}
