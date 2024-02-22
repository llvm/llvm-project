//===- llvm/CASUtil/Utils.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CASOBJECTFORMATS_UTILS_H
#define LLVM_CASOBJECTFORMATS_UTILS_H

#include "llvm/Support/Error.h"

namespace llvm {

class raw_ostream;
class MemoryBufferRef;

namespace cas {
class ObjectStore;
class CASID;

Expected<CASID> readCASIDBuffer(cas::ObjectStore &CAS,
                                llvm::MemoryBufferRef Buffer);

void writeCASIDBuffer(const CASID &ID, llvm::raw_ostream &OS);

} // namespace cas

namespace casobjectformats {

namespace reader {
class CASObjectReader;
}

Error printCASObject(const reader::CASObjectReader &Reader, raw_ostream &OS,
                     bool omitCASID,
                     std::function<const char *(uint8_t)> GetEdgeName,
                     std::function<const char *(uint8_t)> GetScopeName,
                     std::function<const char *(uint8_t)> GetLinkageName);

} // end namespace casobjectformats
} // end namespace llvm

#endif // LLVM_CASOBJECTFORMATS_UTILS_H
