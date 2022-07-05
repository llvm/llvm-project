//===- llvm/CAS/CASOutputBackend.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_CASOUTPUTBACKEND_H
#define LLVM_CAS_CASOUTPUTBACKEND_H

#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualOutputBackend.h"

namespace llvm {
namespace cas {
class CASDB;
class CASID;
class ObjectProxy;

/// Handle the cas
class CASOutputBackend : public vfs::OutputBackend {
  void anchor() override;

public:
  /// Create a top-level tree for all created files. This will contain all files
  Expected<ObjectProxy> getCASProxy();

  /// Add a CAS object to the path in the output backend.
  Error addObject(StringRef Path, ObjectRef Object);

private:
  Expected<std::unique_ptr<vfs::OutputFileImpl>>
  createFileImpl(StringRef Path, Optional<vfs::OutputConfig> Config) override;

  /// Backend is fully thread-safe (so far). Just return a pointer to itself.
  IntrusiveRefCntPtr<vfs::OutputBackend> cloneImpl() const override {
    return IntrusiveRefCntPtr<CASOutputBackend>(
        const_cast<CASOutputBackend *>(this));
  }

public:
  CASOutputBackend(std::shared_ptr<CASDB> CAS);
  CASOutputBackend(CASDB &CAS);
  ~CASOutputBackend();

private:
  struct PrivateImpl;
  std::unique_ptr<PrivateImpl> Impl;

  CASDB &CAS;
  std::shared_ptr<CASDB> OwnedCAS;
};

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_CASOUTPUTBACKEND_H
