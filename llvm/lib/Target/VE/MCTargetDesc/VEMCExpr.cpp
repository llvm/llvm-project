//===-- VEMCExpr.cpp - VE specific MC expression classes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the assembly expression modifiers
// accepted by the VE architecture (e.g. "%hi", "%lo", ...).
//
//===----------------------------------------------------------------------===//

#include "VEMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

#define DEBUG_TYPE "vemcexpr"

VE::Fixups VE::getFixupKind(uint8_t S) {
  switch (S) {
  default:
    llvm_unreachable("Unhandled VEMCExpr::Specifier");
  case VE::S_REFLONG:
    return VE::fixup_ve_reflong;
  case VE::S_HI32:
    return VE::fixup_ve_hi32;
  case VE::S_LO32:
    return VE::fixup_ve_lo32;
  case VE::S_PC_HI32:
    return VE::fixup_ve_pc_hi32;
  case VE::S_PC_LO32:
    return VE::fixup_ve_pc_lo32;
  case VE::S_GOT_HI32:
    return VE::fixup_ve_got_hi32;
  case VE::S_GOT_LO32:
    return VE::fixup_ve_got_lo32;
  case VE::S_GOTOFF_HI32:
    return VE::fixup_ve_gotoff_hi32;
  case VE::S_GOTOFF_LO32:
    return VE::fixup_ve_gotoff_lo32;
  case VE::S_PLT_HI32:
    return VE::fixup_ve_plt_hi32;
  case VE::S_PLT_LO32:
    return VE::fixup_ve_plt_lo32;
  case VE::S_TLS_GD_HI32:
    return VE::fixup_ve_tls_gd_hi32;
  case VE::S_TLS_GD_LO32:
    return VE::fixup_ve_tls_gd_lo32;
  case VE::S_TPOFF_HI32:
    return VE::fixup_ve_tpoff_hi32;
  case VE::S_TPOFF_LO32:
    return VE::fixup_ve_tpoff_lo32;
  }
}
