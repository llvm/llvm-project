//===-- SFrame.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/SFrame.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;

ArrayRef<EnumEntry<sframe::Version>> sframe::getVersions() {
  static constexpr EnumEntry<Version> Versions[] = {
#define HANDLE_SFRAME_VERSION(CODE, NAME) {#NAME, sframe::Version::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };

  return ArrayRef(Versions);
}

ArrayRef<EnumEntry<sframe::Flags>> sframe::getFlags() {
  static constexpr EnumEntry<sframe::Flags> Flags[] = {
#define HANDLE_SFRAME_FLAG(CODE, NAME) {#NAME, sframe::Flags::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  return ArrayRef(Flags);
}

ArrayRef<EnumEntry<sframe::ABI>> sframe::getABIs() {
  static constexpr EnumEntry<sframe::ABI> ABIs[] = {
#define HANDLE_SFRAME_ABI(CODE, NAME) {#NAME, sframe::ABI::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  return ArrayRef(ABIs);
}
