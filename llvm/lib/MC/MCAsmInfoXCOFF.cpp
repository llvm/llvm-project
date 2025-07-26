//===- MC/MCAsmInfoXCOFF.cpp - XCOFF asm properties ------------ *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoXCOFF.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace llvm {
extern cl::opt<cl::boolOrDefault> UseLEB128Directives;
}

MCAsmInfoXCOFF::MCAsmInfoXCOFF() {
  IsAIX = true;
  IsLittleEndian = false;

  PrivateGlobalPrefix = "L..";
  PrivateLabelPrefix = "L..";
  SupportsQuotedNames = false;
  if (UseLEB128Directives == cl::BOU_UNSET)
    HasLEB128Directives = false;
  ZeroDirective = "\t.space\t";
  AsciiDirective = nullptr; // not supported
  AscizDirective = nullptr; // not supported
  CharacterLiteralSyntax = ACLS_SingleQuotePrefix;

  // Use .vbyte for data definition to avoid directives that apply an implicit
  // alignment.
  Data16bitsDirective = "\t.vbyte\t2, ";
  Data32bitsDirective = "\t.vbyte\t4, ";

  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;
  HasDotTypeDotSizeDirective = false;
  ParseInlineAsmUsingAsmParser = true;

  ExceptionsType = ExceptionHandling::AIX;
}

bool MCAsmInfoXCOFF::isAcceptableChar(char C) const {
  // QualName is allowed for a MCSymbolXCOFF, and
  // QualName contains '[' and ']'.
  if (C == '[' || C == ']')
    return true;

  // For AIX assembler, symbols may consist of numeric digits,
  // underscores, periods, uppercase or lowercase letters, or
  // any combination of these.
  return isAlnum(C) || C == '_' || C == '.';
}

bool MCAsmInfoXCOFF::useCodeAlign(const MCSection &Sec) const {
  return static_cast<const MCSectionXCOFF &>(Sec).getKind().isText();
}

MCSectionXCOFF::~MCSectionXCOFF() = default;

void MCSectionXCOFF::printCsectDirective(raw_ostream &OS) const {
  OS << "\t.csect " << QualName->getName() << "," << Log2(getAlign()) << '\n';
}

void MCSectionXCOFF::printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                                          raw_ostream &OS,
                                          uint32_t Subsection) const {
  if (getKind().isText()) {
    if (getMappingClass() != XCOFF::XMC_PR)
      report_fatal_error("Unhandled storage-mapping class for .text csect");

    printCsectDirective(OS);
    return;
  }

  if (getKind().isReadOnly()) {
    if (getMappingClass() != XCOFF::XMC_RO &&
        getMappingClass() != XCOFF::XMC_TD)
      report_fatal_error("Unhandled storage-mapping class for .rodata csect.");
    printCsectDirective(OS);
    return;
  }

  if (getKind().isReadOnlyWithRel()) {
    if (getMappingClass() != XCOFF::XMC_RW &&
        getMappingClass() != XCOFF::XMC_RO &&
        getMappingClass() != XCOFF::XMC_TD)
      report_fatal_error(
          "Unexepected storage-mapping class for ReadOnlyWithRel kind");
    printCsectDirective(OS);
    return;
  }

  // Initialized TLS data.
  if (getKind().isThreadData()) {
    // We only expect XMC_TL here for initialized TLS data.
    if (getMappingClass() != XCOFF::XMC_TL)
      report_fatal_error("Unhandled storage-mapping class for .tdata csect.");
    printCsectDirective(OS);
    return;
  }

  if (getKind().isData()) {
    switch (getMappingClass()) {
    case XCOFF::XMC_RW:
    case XCOFF::XMC_DS:
    case XCOFF::XMC_TD:
      printCsectDirective(OS);
      break;
    case XCOFF::XMC_TC:
    case XCOFF::XMC_TE:
      break;
    case XCOFF::XMC_TC0:
      OS << "\t.toc\n";
      break;
    default:
      report_fatal_error("Unhandled storage-mapping class for .data csect.");
    }
    return;
  }

  if (isCsect() && getMappingClass() == XCOFF::XMC_TD) {
    // Common csect type (uninitialized storage) does not have to print csect
    // directive for section switching unless it is local.
    if (getKind().isCommon() && !getKind().isBSSLocal())
      return;

    assert(getKind().isBSS() && "Unexpected section kind for toc-data");
    printCsectDirective(OS);
    return;
  }
  // Common csect type (uninitialized storage) does not have to print csect
  // directive for section switching.
  if (isCsect() && getCSectType() == XCOFF::XTY_CM) {
    assert((getMappingClass() == XCOFF::XMC_RW ||
            getMappingClass() == XCOFF::XMC_BS ||
            getMappingClass() == XCOFF::XMC_UL) &&
           "Generated a storage-mapping class for a common/bss/tbss csect we "
           "don't "
           "understand how to switch to.");
    // Common symbols and local zero-initialized symbols for TLS and Non-TLS are
    // eligible for .bss/.tbss csect, getKind().isThreadBSS() is used to cover
    // TLS common and zero-initialized local symbols since linkage type (in the
    // GlobalVariable) is not accessible in this class.
    assert((getKind().isBSSLocal() || getKind().isCommon() ||
            getKind().isThreadBSS()) &&
           "wrong symbol type for .bss/.tbss csect");
    // Don't have to print a directive for switching to section for commons and
    // zero-initialized TLS data. The '.comm' and '.lcomm' directives of the
    // variable will create the needed csect.
    return;
  }

  // Zero-initialized TLS data with weak or external linkage are not eligible to
  // be put into common csect.
  if (getKind().isThreadBSS()) {
    printCsectDirective(OS);
    return;
  }

  // XCOFF debug sections.
  if (getKind().isMetadata() && isDwarfSect()) {
    OS << "\n\t.dwsect " << format("0x%" PRIx32, *getDwarfSubtypeFlags())
       << '\n';
    OS << getName() << ':' << '\n';
    return;
  }

  report_fatal_error("Printing for this SectionKind is unimplemented.");
}
