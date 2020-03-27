//===-- llvm/BinaryFormat/XCOFF.cpp - The XCOFF file format -----*- C++/-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/XCOFF.h"

using namespace llvm;

StringRef XCOFF::getMappingClassString(XCOFF::StorageMappingClass SMC) {
  switch (SMC) {
  case XCOFF::XMC_DS:
    return "DS";
  case XCOFF::XMC_RW:
    return "RW";
  case XCOFF::XMC_PR:
    return "PR";
  case XCOFF::XMC_TC0:
    return "TC0";
  case XCOFF::XMC_BS:
    return "BS";
  case XCOFF::XMC_RO:
    return "RO";
  case XCOFF::XMC_UA:
    return "UA";
  case XCOFF::XMC_TC:
    return "TC";
  default:
    report_fatal_error("Unhandled storage-mapping class.");
  }
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
