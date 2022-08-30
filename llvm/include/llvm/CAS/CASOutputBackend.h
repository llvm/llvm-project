//===- llvm/CAS/CASOutputBackend.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASOUTPUTBACKEND_H
#define LLVM_CAS_CASOUTPUTBACKEND_H

#include "llvm/CAS/CASReference.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualOutputBackend.h"

namespace llvm {
namespace cas {
class ObjectStore;
class CASID;

/// Handle the cas
class CASOutputBackend : public vfs::OutputBackend {
  void anchor() override;

public:
  ObjectStore &getCAS() const { return CAS; }

  struct OutputFile {
    std::string Path;
    ObjectRef Object;
  };

  SmallVector<OutputFile> takeOutputs() { return std::move(Outputs); }

private:
  Expected<std::unique_ptr<vfs::OutputFileImpl>>
  createFileImpl(StringRef Path, Optional<vfs::OutputConfig> Config) override;

  /// Backend is fully thread-safe (so far). Just return a pointer to itself.
  IntrusiveRefCntPtr<vfs::OutputBackend> cloneImpl() const override {
    return IntrusiveRefCntPtr<CASOutputBackend>(
        const_cast<CASOutputBackend *>(this));
  }

public:
  CASOutputBackend(std::shared_ptr<ObjectStore> CAS);
  CASOutputBackend(ObjectStore &CAS);
  ~CASOutputBackend();

private:
  SmallVector<OutputFile> Outputs;
  ObjectStore &CAS;
  std::shared_ptr<ObjectStore> OwnedCAS;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_CASOUTPUTBACKEND_H
