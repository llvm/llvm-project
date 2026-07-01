//===-- SFrame.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/SFrame.h"
#include "llvm/ADT/Enum.h"

using namespace llvm;

EnumStrings<sframe::Version> sframe::getVersions() {
  constexpr EnumStringDef<Version> VersionDefs[] = {
#define HANDLE_SFRAME_VERSION(CODE, NAME) {{#NAME}, sframe::Version::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto Versions = BUILD_ENUM_STRINGS(VersionDefs);
  return EnumStrings(Versions);
}

EnumStrings<sframe::Flags> sframe::getFlags() {
  constexpr EnumStringDef<sframe::Flags> FlagDefs[] = {
#define HANDLE_SFRAME_FLAG(CODE, NAME) {{#NAME}, sframe::Flags::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto Flags = BUILD_ENUM_STRINGS(FlagDefs);
  return EnumStrings(Flags);
}

EnumStrings<sframe::ABI> sframe::getABIs() {
  constexpr EnumStringDef<sframe::ABI> ABIDefs[] = {
#define HANDLE_SFRAME_ABI(CODE, NAME) {{#NAME}, sframe::ABI::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto ABIs = BUILD_ENUM_STRINGS(ABIDefs);
  return EnumStrings(ABIs);
}

EnumStrings<sframe::FREType> sframe::getFRETypes() {
  constexpr EnumStringDef<sframe::FREType> FRETypeDefs[] = {
#define HANDLE_SFRAME_FRE_TYPE(CODE, NAME) {{#NAME}, sframe::FREType::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto FRETypes = BUILD_ENUM_STRINGS(FRETypeDefs);
  return EnumStrings(FRETypes);
}

EnumStrings<sframe::FDEType> sframe::getFDETypes() {
  constexpr EnumStringDef<sframe::FDEType> FDETypeDefs[] = {
#define HANDLE_SFRAME_FDE_TYPE(CODE, NAME) {{#NAME}, sframe::FDEType::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto FDETypes = BUILD_ENUM_STRINGS(FDETypeDefs);
  return EnumStrings(FDETypes);
}

EnumStrings<sframe::AArch64PAuthKey> sframe::getAArch64PAuthKeys() {
  constexpr EnumStringDef<sframe::AArch64PAuthKey> AArch64PAuthKeyDefs[] = {
#define HANDLE_SFRAME_AARCH64_PAUTH_KEY(CODE, NAME)                            \
  {{#NAME}, sframe::AArch64PAuthKey::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto AArch64PAuthKeys =
      BUILD_ENUM_STRINGS(AArch64PAuthKeyDefs);
  return EnumStrings(AArch64PAuthKeys);
}

EnumStrings<sframe::FREOffset> sframe::getFREOffsets() {
  constexpr EnumStringDef<sframe::FREOffset> FREOffsetDefs[] = {
#define HANDLE_SFRAME_FRE_OFFSET(CODE, NAME) {{#NAME}, sframe::FREOffset::NAME},
#include "llvm/BinaryFormat/SFrameConstants.def"
  };
  static constexpr auto FREOffsets = BUILD_ENUM_STRINGS(FREOffsetDefs);
  return EnumStrings(FREOffsets);
}

EnumStrings<sframe::BaseReg> sframe::getBaseRegisters() {
  constexpr EnumStringDef<sframe::BaseReg> BaseRegDefs[] = {
      {{"FP"}, sframe::BaseReg::FP},
      {{"SP"}, sframe::BaseReg::SP},
  };
  static constexpr auto BaseRegs = BUILD_ENUM_STRINGS(BaseRegDefs);
  return EnumStrings(BaseRegs);
}
