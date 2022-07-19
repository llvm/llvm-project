//===- llvm/CAS/Utils.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_UTILS_H
#define LLVM_CAS_UTILS_H

#include "llvm/Support/Error.h"

namespace llvm {

class MemoryBufferRef;

namespace cas {

class CASDB;
class CASID;

Expected<CASID> readCASIDBuffer(cas::CASDB &CAS, llvm::MemoryBufferRef Buffer);

void writeCASIDBuffer(const CASID &ID, llvm::raw_ostream &OS);

} // namespace cas
} // namespace llvm

#endif // LLVM_CAS_UTILS_H
