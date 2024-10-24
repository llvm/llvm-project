//===- CASRegistry.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/PluginCAS.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace llvm::cas;

static Expected<
    std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
createOnDiskCASImpl(const Twine &Path) {
  std::string CASPath = Path.str();
  // If path is empty, use default ondisk CAS path.
  if (CASPath.empty())
    CASPath = getDefaultOnDiskCASPath();

  auto UniDB = createOnDiskUnifiedCASDatabases(Path.str());
  if (!UniDB)
    return UniDB.takeError();

  return std::pair{std::move(UniDB->first), std::move(UniDB->second)};
}

static Expected<
    std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
createPluginCASImpl(const Twine &URL) {
  // Format used is
  //   plugin://${PATH_TO_PLUGIN}?${OPT1}=${VAL1}&${OPT2}=${VAL2}..
  // "ondisk-path" as option is treated specially, the rest of options are
  // passed to the plugin verbatim.
  SmallString<256> PathBuf;
  auto [PluginPath, Options] = URL.toStringRef(PathBuf).split('?');
  std::string OnDiskPath;
  SmallVector<std::pair<std::string, std::string>> PluginArgs;
  while (!Options.empty()) {
    StringRef Opt;
    std::tie(Opt, Options) = Options.split('&');
    auto [Name, Value] = Opt.split('=');
    if (Name == "ondisk-path") {
      OnDiskPath = Value;
    } else {
      PluginArgs.push_back({std::string(Name), std::string(Value)});
    }
  }

  if (OnDiskPath.empty())
    OnDiskPath = getDefaultOnDiskCASPath();

  return createPluginCASDatabases(PluginPath, OnDiskPath, PluginArgs);
}

static Expected<
    std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
createInMemoryCASImpl(const Twine &) {
  return std::pair{createInMemoryCAS(), createInMemoryActionCache()};
}

static ManagedStatic<StringMap<ObjectStoreCreateFuncTy *>> RegisteredScheme;

static StringMap<ObjectStoreCreateFuncTy *> &getRegisteredScheme() {
  if (!RegisteredScheme.isConstructed()) {
    RegisteredScheme->insert({"mem://", &createInMemoryCASImpl});
    RegisteredScheme->insert({"file://", &createOnDiskCASImpl});
    RegisteredScheme->insert({"plugin://", &createPluginCASImpl});
  }
  return *RegisteredScheme;
}

Expected<std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
cas::createCASFromIdentifier(StringRef Id) {
  for (auto &Scheme : getRegisteredScheme()) {
    if (Id.consume_front(Scheme.getKey()))
      return Scheme.getValue()(Id);
  }

  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           "Unknown CAS identifier is provided");
}

bool cas::isRegisteredCASIdentifier(StringRef Id) {
  for (auto &Scheme : getRegisteredScheme()) {
    if (Id.consume_front(Scheme.getKey()))
      return true;
  }
  return false;
}

void cas::registerCASURLScheme(StringRef Prefix,
                               ObjectStoreCreateFuncTy *Func) {
  getRegisteredScheme().insert({Prefix, Func});
}
