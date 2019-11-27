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
