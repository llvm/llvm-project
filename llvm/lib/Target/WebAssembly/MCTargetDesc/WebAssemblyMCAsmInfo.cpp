//===-- WebAssemblyMCAsmInfo.cpp - WebAssembly asm properties -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declarations of the WebAssemblyMCAsmInfo
/// properties.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCAsmInfo.h"
#include "WebAssemblyMCTargetDesc.h"
#include "llvm/ADT/Enum.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-mc-asm-info"

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefs[] = {
    {{"TYPEINDEX"}, WebAssembly::S_TYPEINDEX},
    {{"TBREL"}, WebAssembly::S_TBREL},
    {{"MBREL"}, WebAssembly::S_MBREL},
    {{"TLSREL"}, WebAssembly::S_TLSREL},
    {{"GOT"}, WebAssembly::S_GOT},
    {{"GOT@TLS"}, WebAssembly::S_GOT_TLS},
    {{"FUNCINDEX"}, WebAssembly::S_FUNCINDEX},
};
constexpr auto atSpecifiers = BUILD_ENUM_STRINGS(AtSpecifierDefs);

WebAssemblyMCAsmInfo::~WebAssemblyMCAsmInfo() = default; // anchor.

WebAssemblyMCAsmInfo::WebAssemblyMCAsmInfo(const Triple &T,
                                           const MCTargetOptions &Options)
    : MCAsmInfoWasm(Options) {
  CodePointerSize = CalleeSaveStackSlotSize = T.isArch64Bit() ? 8 : 4;

  // TODO: What should MaxInstLength be?

  UseDataRegionDirectives = true;

  // Use .skip instead of .zero because .zero is confusing when used with two
  // arguments (it doesn't actually zero things out).
  ZeroDirective = "\t.skip\t";

  Data8bitsDirective = "\t.int8\t";
  Data16bitsDirective = "\t.int16\t";
  Data32bitsDirective = "\t.int32\t";
  Data64bitsDirective = "\t.int64\t";

  AlignmentIsInBytes = false;
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;

  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::None;

  initializeAtSpecifiers(atSpecifiers);
}
