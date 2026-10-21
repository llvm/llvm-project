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
#include "llvm/Option/LibraryOptions.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

#define OPTIONS_STRUCT_DECL
#include "PluginsOptions.inc"
#define OPTIONS_STRUCT_DEFS
#include "PluginsOptions.inc"

static llvm::opt::RegisterLibraryOptions<llvm::PluginsOptions> Registration;

using namespace llvm;

Expected<PassPlugin> PassPlugin::load(StringRef Filename) {
  std::string Error;
  auto Library =
      sys::DynamicLibrary::getPermanentLibrary(Filename.str().c_str(), &Error);
  if (!Library.isValid())
    return make_error<StringError>(Twine("Could not load library '") +
                                       Filename + "': " + Error,
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

Error llvm::passPluginArguments(ArrayRef<PassPluginLibraryInfo> Infos,
                                const cl::LibraryOptionsContext *Opts) {
  DenseMap<StringRef, unsigned> Index;
  for (auto [I, Info] : enumerate(Infos))
    if (!Index.try_emplace(Info.PluginName, I).second)
      return createStringError("multiple pass plugins are named '" +
                               Twine(Info.PluginName) + "'");
  // Collect all plugin arguments and drop their prefixes. For example given
  // -plugin-arg=Polly,-polly-tile-sizes=256,16 we pass -polly-tile-sizes=256,16
  // to the plugin named Polly.
  SmallVector<SmallVector<const char *, 0>, 0> PluginArgs(Infos.size());
  const auto &PluginOpts = cl::LibraryOptionsContext::get<PluginsOptions>(Opts);
  for (StringRef Arg : PluginOpts.plugin_args) {
    auto [Name, Rest] = Arg.split(',');
    // Rest is null without a comma and empty for an empty argument.
    if (!Rest.data())
      return createStringError("expected <plugin>,<arg> in -plugin-arg=" + Arg);
    auto It = Index.find(Name);
    if (It == Index.end())
      return createStringError("no pass plugin named '" + Name +
                               "' is loaded, in -plugin-arg=" + Arg);
    // Arg is a whole value from argv, so its tail is null-terminated.
    assert(Rest.data()[Rest.size()] == '\0' && "Must be null-terminated");
    PluginArgs[It->second].push_back(Rest.data());
  }
  for (auto [Info, PArgs] : zip_equal(Infos, PluginArgs)) {
    if (PArgs.empty())
      continue;
    if (!Info.ParseArguments)
      return createStringError("pass plugin '" + Twine(Info.PluginName) +
                               "' does not accept arguments");
    if (Error E = Info.ParseArguments(PArgs))
      return E;
  }
  return Error::success();
}
