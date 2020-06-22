//===- Reproducer.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Reproducer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::dsymutil;

static std::string createReproducerDir(std::error_code &EC) {
  SmallString<128> Root;
  if (const char *Path = getenv("DSYMUTIL_REPRODUCER_PATH")) {
    Root.assign(Path);
    EC = sys::fs::create_directory(Root);
  } else {
    EC = sys::fs::createUniqueDirectory("dsymutil", Root);
  }
  return EC ? "" : std::string(Root);
}

Reproducer::Reproducer() : VFS(vfs::getRealFileSystem()) {}
Reproducer::~Reproducer() = default;

ReproducerGenerate::ReproducerGenerate(std::error_code &EC)
    : Root(createReproducerDir(EC)), FC() {
  if (!Root.empty())
    FC = std::make_shared<FileCollector>(Root, Root);
  VFS = FileCollector::createCollectorVFS(vfs::getRealFileSystem(), FC);
}

ReproducerGenerate::~ReproducerGenerate() {
  if (!FC)
    return;
  FC->copyFiles(false);
  SmallString<128> Mapping(Root);
  sys::path::append(Mapping, "mapping.yaml");
  FC->writeMapping(Mapping.str());
  outs() << "reproducer written to " << Root << '\n';
}

ReproducerUse::~ReproducerUse() = default;

ReproducerUse::ReproducerUse(StringRef Root, std::error_code &EC) {
  SmallString<128> Mapping(Root);
  sys::path::append(Mapping, "mapping.yaml");
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
      vfs::getRealFileSystem()->getBufferForFile(Mapping.str());

  if (!Buffer) {
    EC = Buffer.getError();
    return;
  }

  VFS = llvm::vfs::getVFSFromYAML(std::move(Buffer.get()), nullptr, Mapping);
}

llvm::Expected<std::unique_ptr<Reproducer>>
Reproducer::createReproducer(ReproducerMode Mode, StringRef Root) {
  switch (Mode) {
  case ReproducerMode::Generate: {
    std::error_code EC;
    std::unique_ptr<Reproducer> Repro =
        std::make_unique<ReproducerGenerate>(EC);
    if (EC)
      return errorCodeToError(EC);
    return std::move(Repro);
  }
  case ReproducerMode::Use: {
    std::error_code EC;
    std::unique_ptr<Reproducer> Repro =
        std::make_unique<ReproducerUse>(Root, EC);
    if (EC)
      return errorCodeToError(EC);
    return std::move(Repro);
  }
  case ReproducerMode::Off:
    return std::make_unique<Reproducer>();
  }
  llvm_unreachable("All cases handled above.");
}
