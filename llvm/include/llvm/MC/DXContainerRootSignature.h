//===- llvm/MC/DXContainerRootSignature.h - RootSignature -*- C++ -*- ========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/BinaryStreamWriter.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

namespace llvm {

class raw_ostream;

namespace mcdxbc {

class StreamOffsetHelper {
private:
  std::map<std::string, std::pair<uint32_t, uint32_t>> OffsetsMaping;
  BinaryStreamWriter &Stream;

public:
  explicit StreamOffsetHelper(BinaryStreamWriter &Stream) : Stream(Stream) {}

  Error addOffset(std::string Key);

  void addRewriteValue(std::string Key);

  Error rewrite();
};

struct RootSignatureDesc {
  dxbc::RootSignatureHeader Header;
  SmallVector<dxbc::RootParameter> Parameters;

  Error write(raw_ostream &OS) const;

  uint32_t getSizeInBytes() const {
    // Header Size + accounting for parameter offset + parameters size
    return 24 + (Parameters.size() * 4) + Parameters.size_in_bytes();
  }
};
} // namespace mcdxbc
} // namespace llvm
