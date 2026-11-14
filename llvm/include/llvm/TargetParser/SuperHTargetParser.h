//===-- SuperHTargetParser - Parser for SuperH target features --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise SuperH hardware features
// such as FPU/CPU/ARCH/extensions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_SUPERHTARGETPARSER_H
#define LLVM_TARGETPARSER_SUPERHTARGETPARSER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace SuperH {

enum class ISAKind { INVALID = 0, SH1, SH2, SH2E, SH2A, SH3, SH3E, SH4, SH4A };

enum class EndianKind { INVALID = 0, LITTLE, BIG };

// SH1-4A
ISAKind parseArchISA(StringRef ArchName);

// Little/Big endian
EndianKind parseArchEndian(StringRef ArchName);

} // namespace SuperH
} // namespace llvm
#endif