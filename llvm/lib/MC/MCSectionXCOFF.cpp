//===- lib/MC/MCSectionXCOFF.cpp - XCOFF Code Section Representation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCSectionXCOFF::~MCSectionXCOFF() = default;


void MCSectionXCOFF::PrintSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                                          raw_ostream &OS,
                                          const MCExpr *Subsection) const {
  if (getKind().isText()) {
    if (getMappingClass() != XCOFF::XMC_PR)
      report_fatal_error("Unhandled storage-mapping class for .text csect");

    OS << "\t.csect " << QualName->getName() << '\n';
    return;
  }

  if (getKind().isReadOnly()) {
    if (getMappingClass() != XCOFF::XMC_RO)
      report_fatal_error("Unhandled storage-mapping class for .rodata csect.");
    OS << "\t.csect " << QualName->getName() << '\n';
    return;
  }

  if (getKind().isData()) {
    switch (getMappingClass()) {
    case XCOFF::XMC_RW:
    case XCOFF::XMC_DS:
      OS << "\t.csect " << QualName->getName() << '\n';
      break;
    case XCOFF::XMC_TC:
      break;
    case XCOFF::XMC_TC0:
      OS << "\t.toc\n";
      break;
    default:
      report_fatal_error(
          "Unhandled storage-mapping class for .data csect.");
    }
    return;
  }

  if (getKind().isBSSLocal() || getKind().isCommon()) {
    assert((getMappingClass() == XCOFF::XMC_RW ||
            getMappingClass() == XCOFF::XMC_BS) &&
           "Generated a storage-mapping class for a common/bss csect we don't "
           "understand how to switch to.");
    assert(getCSectType() == XCOFF::XTY_CM &&
           "wrong csect type for .bss csect");
    // Don't have to print a directive for switching to section for commons.
    // '.comm' and '.lcomm' directives for the variable will create the needed
    // csect.
    return;
  }

  report_fatal_error("Printing for this SectionKind is unimplemented.");
}

bool MCSectionXCOFF::UseCodeAlign() const { return getKind().isText(); }

bool MCSectionXCOFF::isVirtualSection() const { return XCOFF::XTY_CM == Type; }
