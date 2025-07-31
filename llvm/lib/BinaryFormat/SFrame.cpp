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

ArrayRef<EnumEntry<sframe::FREType>> sframe::getFRETypes() {
  static constexpr EnumEntry<sframe::FREType> FRETypes[] = {
#define HANDLE_SFRAME_FRE_TYPE(CODE, NAME) {#NAME, sframe::FREType::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  return ArrayRef(FRETypes);
}

ArrayRef<EnumEntry<sframe::FDEType>> sframe::getFDETypes() {
  static constexpr EnumEntry<sframe::FDEType> FDETypes[] = {
#define HANDLE_SFRAME_FDE_TYPE(CODE, NAME) {#NAME, sframe::FDEType::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  return ArrayRef(FDETypes);
}

ArrayRef<EnumEntry<sframe::AArch64PAuthKey>> sframe::getAArch64PAuthKeys() {
  static constexpr EnumEntry<sframe::AArch64PAuthKey> AArch64PAuthKeys[] = {
#define HANDLE_SFRAME_AARCH64_PAUTH_KEY(CODE, NAME)                            \
  {#NAME, sframe::AArch64PAuthKey::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  return ArrayRef(AArch64PAuthKeys);
}

ArrayRef<EnumEntry<sframe::FREOffset>> sframe::getFREOffsets() {
  static constexpr EnumEntry<sframe::FREOffset> FREOffsets[] = {
#define HANDLE_SFRAME_FRE_OFFSET(CODE, NAME) {#NAME, sframe::FREOffset::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  return ArrayRef(FREOffsets);
}
