//===- CASOutputBackend.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASOutputBackend.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/CAS/Utils.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

using namespace llvm;
using namespace llvm::cas;

void CASOutputBackend::anchor() {}

namespace {
class CASOutputFile final : public vfs::OutputFileImpl {
public:
  Error keep() override { return OnKeep(Path, Bytes); }
  Error discard() override { return Error::success(); }
  raw_pwrite_stream &getOS() override { return OS; }

  using OnKeepType = llvm::unique_function<Error(StringRef, StringRef)>;
  CASOutputFile(StringRef Path, OnKeepType OnKeep)
      : Path(Path.str()), OS(Bytes), OnKeep(std::move(OnKeep)) {}

private:
  std::string Path;
  SmallString<16> Bytes;
  raw_svector_ostream OS;
  OnKeepType OnKeep;
};
} // namespace

CASOutputBackend::CASOutputBackend(std::shared_ptr<ObjectStore> CAS)
    : CASOutputBackend(*CAS) {
  this->OwnedCAS = std::move(CAS);
}

CASOutputBackend::CASOutputBackend(ObjectStore &CAS) : CAS(CAS) {}

CASOutputBackend::~CASOutputBackend() = default;

Expected<std::unique_ptr<vfs::OutputFileImpl>>
CASOutputBackend::createFileImpl(StringRef ResolvedPath,
                                 Optional<vfs::OutputConfig> Config) {
  // FIXME: CASIDOutputBackend.createFile() should be called NOW (not inside
  // the OnKeep closure) so that if there are initialization errors (such as
  // output directory not existing) they're reported by createFileImpl().
  //
  // The opened file can be kept inside \a CASOutputFile and forwarded.
  return std::make_unique<CASOutputFile>(
      ResolvedPath, [&](StringRef Path, StringRef Bytes) -> Error {
        Optional<ObjectRef> BytesRef;
        if (Error E = CAS.storeFromString(None, Bytes).moveInto(BytesRef))
          return E;
        // FIXME: Should there be a lock taken before modifying Outputs?
        Outputs.push_back({std::string(Path), *BytesRef});
        return Error::success();
      });
}
