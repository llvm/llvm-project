//===- Dtlto.h - Distributed ThinLTO functions and classes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef LLVM_DTLTO_H
#define LLVM_DTLTO_H

#include "llvm/LTO/LTO.h"
#include "llvm/Support/MemoryBuffer.h"

namespace dtlto {

llvm::Expected<llvm::lto::InputFile*> addInput(llvm::lto::LTO *LtoObj,
                               std::unique_ptr<llvm::lto::InputFile> Input);

llvm::Error process(llvm::lto::LTO &LtoObj);
} // namespace dtlto

#endif // LLVM_DTLTO_H
