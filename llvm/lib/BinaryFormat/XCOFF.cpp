//===-- llvm/BinaryFormat/XCOFF.cpp - The XCOFF file format -----*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/ADT/StringRef.h"

using namespace llvm;

#define SMC_CASE(A)                                                            \
  case XCOFF::XMC_##A:                                                         \
    return #A;
StringRef XCOFF::getMappingClassString(XCOFF::StorageMappingClass SMC) {
  switch (SMC) {
    SMC_CASE(PR)
    SMC_CASE(RO)
    SMC_CASE(DB)
    SMC_CASE(GL)
    SMC_CASE(XO)
    SMC_CASE(SV)
    SMC_CASE(SV64)
    SMC_CASE(SV3264)
    SMC_CASE(TI)
    SMC_CASE(TB)
    SMC_CASE(RW)
    SMC_CASE(TC0)
    SMC_CASE(TC)
    SMC_CASE(TD)
    SMC_CASE(DS)
    SMC_CASE(UA)
    SMC_CASE(BS)
    SMC_CASE(UC)
    SMC_CASE(TL)
    SMC_CASE(UL)
    SMC_CASE(TE)
#undef SMC_CASE
  }

  // TODO: need to add a test case for "Unknown" and other SMC.
  return "Unknown";
}

#define RELOC_CASE(A)                                                          \
  case XCOFF::A:                                                               \
    return #A;
StringRef XCOFF::getRelocationTypeString(XCOFF::RelocationType Type) {
  switch (Type) {
    RELOC_CASE(R_POS)
    RELOC_CASE(R_RL)
    RELOC_CASE(R_RLA)
    RELOC_CASE(R_NEG)
    RELOC_CASE(R_REL)
    RELOC_CASE(R_TOC)
    RELOC_CASE(R_TRL)
    RELOC_CASE(R_TRLA)
    RELOC_CASE(R_GL)
    RELOC_CASE(R_TCL)
    RELOC_CASE(R_REF)
    RELOC_CASE(R_BA)
    RELOC_CASE(R_BR)
    RELOC_CASE(R_RBA)
    RELOC_CASE(R_RBR)
    RELOC_CASE(R_TLS)
    RELOC_CASE(R_TLS_IE)
    RELOC_CASE(R_TLS_LD)
    RELOC_CASE(R_TLS_LE)
    RELOC_CASE(R_TLSM)
    RELOC_CASE(R_TLSML)
    RELOC_CASE(R_TOCU)
    RELOC_CASE(R_TOCL)
  }
  return "Unknown";
}
#undef RELOC_CASE
