//===- CASConfiguration.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASConfiguration.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/BuiltinUnifiedCASDatabases.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/JSON.h"

using namespace llvm;
using namespace llvm::cas;

Error CASConfiguration::getResolvedCASPath(
    llvm::SmallVectorImpl<char> &Result) const {
  if (CASPath == "auto")
    return getDefaultOnDiskCASPath(Result);

  Result.assign(CASPath.begin(), CASPath.end());
  return Error::success();
}

Expected<std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
CASConfiguration::createDatabases() const {
  if (!PluginPath.empty())
    return createPluginCASDatabases(PluginPath, CASPath, PluginOptions);

  if (CASPath.empty()) {
    return std::pair(createInMemoryCAS(), createInMemoryActionCache());
  }

  SmallString<128> PathBuf;
  if (auto E = getResolvedCASPath(PathBuf))
    return std::move(E);

  std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>> DBs;
  return createOnDiskUnifiedCASDatabases(PathBuf);
}

void CASConfiguration::writeConfigurationFile(raw_ostream &OS) const {
  using namespace llvm::json;
  Object Root;
  Root["CASPath"] = CASPath;
  Root["PluginPath"] = PluginPath;

  Array PlugOpts;
  for (const auto &Opt : PluginOptions) {
    Object Entry;
    Entry[Opt.first] = Opt.second;
    PlugOpts.emplace_back(std::move(Entry));
  }
  Root["PluginOptions"] = std::move(PlugOpts);

  OS << formatv("{0:2}", Value(std::move(Root)));
}

Expected<CASConfiguration>
CASConfiguration::createFromConfig(StringRef Content) {
  auto Parsed = json::parse(Content);
  if (!Parsed)
    return Parsed.takeError();

  CASConfiguration Config;
  auto *Root = Parsed->getAsObject();
  if (!Root)
    return createStringError(
        "CASConfiguration file error: top level object missing");

  if (auto CASPath = Root->getString("CASPath"))
    Config.CASPath = *CASPath;

  if (auto PluginPath = Root->getString("PluginPath"))
    Config.PluginPath = *PluginPath;

  if (auto *Opts = Root->getArray("PluginOptions")) {
    for (auto &Opt : *Opts) {
      if (auto *Arg = Opt.getAsObject()) {
        for (auto &Entry : *Arg) {
          if (auto V = Entry.second.getAsString())
            Config.PluginOptions.emplace_back(Entry.first.str(), *V);
        }
      }
    }
  }

  return Config;
}

std::optional<std::pair<std::string, CASConfiguration>>
CASConfiguration::createFromSearchConfigFile(
    StringRef Path, IntrusiveRefCntPtr<vfs::FileSystem> VFS) {
  if (!VFS)
    VFS = vfs::getRealFileSystem();

  while (!Path.empty()) {
    SmallString<256> ConfigPath(Path);
    sys::path::append(ConfigPath, ".cas-config");
    auto File = VFS->openFileForRead(ConfigPath);
    if (!File || !*File) {
      Path = sys::path::parent_path(Path);
      continue;
    }

    auto Buffer = (*File)->getBuffer(ConfigPath);
    if (!Buffer || !*Buffer) {
      Path = sys::path::parent_path(Path);
      continue;
    }

    auto Config = createFromConfig((*Buffer)->getBuffer());
    if (!Config) {
      consumeError(Config.takeError());
      Path = sys::path::parent_path(Path);
      continue;
    }
    return std::pair{ConfigPath.str().str(), *Config};
  }
  return std::nullopt;
}
