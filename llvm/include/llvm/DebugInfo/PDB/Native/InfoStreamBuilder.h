//===- InfoStreamBuilder.h - PDB Info Stream Creation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_INFOSTREAMBUILDER_H
#define LLVM_DEBUGINFO_PDB_NATIVE_INFOSTREAMBUILDER_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"

#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"

namespace llvm {
class WritableBinaryStreamRef;

namespace msf {
class MSFBuilder;
struct MSFLayout;
}
namespace pdb {
class NamedStreamMap;

class InfoStreamBuilder {
public:
  LLVM_ABI InfoStreamBuilder(msf::MSFBuilder &Msf,
                             NamedStreamMap &NamedStreams);
  InfoStreamBuilder(const InfoStreamBuilder &) = delete;
  InfoStreamBuilder &operator=(const InfoStreamBuilder &) = delete;

  LLVM_ABI void setVersion(PdbRaw_ImplVer V);
  LLVM_ABI void addFeature(PdbRaw_FeatureSig Sig);

  // If this is true, the PDB contents are hashed and this hash is used as
  // PDB GUID and as Signature. The age is always 1.
  LLVM_ABI void setHashPDBContentsToGUID(bool B);

  // These only have an effect if hashPDBContentsToGUID() is false.
  LLVM_ABI void setSignature(uint32_t S);
  LLVM_ABI void setAge(uint32_t A);
  LLVM_ABI void setGuid(codeview::GUID G);

  bool hashPDBContentsToGUID() const { return HashPDBContentsToGUID; }
  uint32_t getAge() const { return Age; }
  codeview::GUID getGuid() const { return Guid; }
  std::optional<uint32_t> getSignature() const { return Signature; }

  LLVM_ABI uint32_t finalize();

  LLVM_ABI Error finalizeMsfLayout();

  LLVM_ABI Error commit(const msf::MSFLayout &Layout,
                        WritableBinaryStreamRef Buffer) const;

private:
  msf::MSFBuilder &Msf;

  std::vector<PdbRaw_FeatureSig> Features;
  PdbRaw_ImplVer Ver;
  uint32_t Age;
  std::optional<uint32_t> Signature;
  codeview::GUID Guid;

  bool HashPDBContentsToGUID = false;

  NamedStreamMap &NamedStreams;
};
} // namespace pdb
}

#endif
