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

#include "llvm/TargetParser/SuperHTargetParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace {
struct SHISA {
  StringRef Name;
  SuperH::ISAKind Kind;
};
} // namespace

static SuperH::ISAKind consumeISAKind(StringRef &ArchName) {
  // NOTE:  This list is ordered so that elements that would overrule
  //        sub-architectures are last, to avoid that exact scenario.
  static const SHISA ISANames[] = {
    {"sh2a", SuperH::ISAKind::SH2A},
    {"sh2e", SuperH::ISAKind::SH2E},
    {"sh3e", SuperH::ISAKind::SH3E},
    {"sh4a", SuperH::ISAKind::SH4A},
    {"sh1", SuperH::ISAKind::SH1},
    {"sh2", SuperH::ISAKind::SH2},
    {"sh3", SuperH::ISAKind::SH3},
    {"sh4", SuperH::ISAKind::SH4},

    // If no subtype is defined, assume 4a, this conforms to
    // what GCC does.
    {"sh", SuperH::ISAKind::SH4A},
  };

  // Check for sh[cpu]
  for (auto &KV : ISANames) {
    
    // Enforce eb suffix for sh2e and sh3e.
    if (KV.Name.ends_with("e")) {
      if (!ArchName.ends_with("eb") && !ArchName.ends_with("el"))
        continue;
    }

    if (ArchName.consume_front(KV.Name))
      return KV.Kind;
  }
  return SuperH::ISAKind::INVALID;
}

// SH1-4A
SuperH::ISAKind SuperH::parseArchISA(StringRef ArchName) {
  static const StringRef EndianStrings[] = {
    "l", "el",
    "b", "eb"
  };

  // Check for [e]l/b
  SuperH::ISAKind Kind = consumeISAKind(ArchName);
  for (auto &P : EndianStrings) {
    if (ArchName == P)
      return Kind;
  }

  return SuperH::ISAKind::INVALID;
}

// Little/Big endian
SuperH::EndianKind SuperH::parseArchEndian(StringRef ArchName) {
  return StringSwitch<SuperH::EndianKind>(ArchName)
    .EndsWith("eb", EndianKind::BIG)
    .EndsWith("b", EndianKind::BIG)
    .EndsWith("el", EndianKind::LITTLE)
    .EndsWith("l", EndianKind::LITTLE)
    .Default(EndianKind::INVALID);
}