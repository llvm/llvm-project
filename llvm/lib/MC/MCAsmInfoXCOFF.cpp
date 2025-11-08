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

void MCAsmInfoXCOFF::printSwitchToSection(const MCSection &Section, uint32_t,
                                          const Triple &T,
                                          raw_ostream &OS) const {
  auto &Sec = static_cast<const MCSectionXCOFF &>(Section);
  if (Sec.getKind().isText()) {
    if (Sec.getMappingClass() != XCOFF::XMC_PR)
      report_fatal_error("Unhandled storage-mapping class for .text csect");

    Sec.printCsectDirective(OS);
    return;
  }

  if (Sec.getKind().isReadOnly()) {
    if (Sec.getMappingClass() != XCOFF::XMC_RO &&
        Sec.getMappingClass() != XCOFF::XMC_TD)
      report_fatal_error("Unhandled storage-mapping class for .rodata csect.");
    Sec.printCsectDirective(OS);
    return;
  }

  if (Sec.getKind().isReadOnlyWithRel()) {
    if (Sec.getMappingClass() != XCOFF::XMC_RW &&
        Sec.getMappingClass() != XCOFF::XMC_RO &&
        Sec.getMappingClass() != XCOFF::XMC_TD)
      report_fatal_error(
          "Unexepected storage-mapping class for ReadOnlyWithRel kind");
    Sec.printCsectDirective(OS);
    return;
  }

  // Initialized TLS data.
  if (Sec.getKind().isThreadData()) {
    // We only expect XMC_TL here for initialized TLS data.
    if (Sec.getMappingClass() != XCOFF::XMC_TL)
      report_fatal_error("Unhandled storage-mapping class for .tdata csect.");
    Sec.printCsectDirective(OS);
    return;
  }

  if (Sec.getKind().isData()) {
    switch (Sec.getMappingClass()) {
    case XCOFF::XMC_RW:
    case XCOFF::XMC_DS:
    case XCOFF::XMC_TD:
      Sec.printCsectDirective(OS);
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

  if (Sec.isCsect() && Sec.getMappingClass() == XCOFF::XMC_TD) {
    // Common csect type (uninitialized storage) does not have to print
    // csect directive for section switching unless it is local.
    if (Sec.getKind().isCommon() && !Sec.getKind().isBSSLocal())
      return;

    assert(Sec.getKind().isBSS() && "Unexpected section kind for toc-data");
    Sec.printCsectDirective(OS);
    return;
  }
  // Common csect type (uninitialized storage) does not have to print csect
  // directive for section switching.
  if (Sec.isCsect() && Sec.getCSectType() == XCOFF::XTY_CM) {
    assert((Sec.getMappingClass() == XCOFF::XMC_RW ||
            Sec.getMappingClass() == XCOFF::XMC_BS ||
            Sec.getMappingClass() == XCOFF::XMC_UL) &&
           "Generated a storage-mapping class for a common/bss/tbss csect we "
           "don't "
           "understand how to switch to.");
    // Common symbols and local zero-initialized symbols for TLS and Non-TLS are
    // eligible for .bss/.tbss csect, getKind().isThreadBSS() is used to
    // cover TLS common and zero-initialized local symbols since linkage type
    // (in the GlobalVariable) is not accessible in this class.
    assert((Sec.getKind().isBSSLocal() || Sec.getKind().isCommon() ||
            Sec.getKind().isThreadBSS()) &&
           "wrong symbol type for .bss/.tbss csect");
    // Don't have to print a directive for switching to section for commons
    // and zero-initialized TLS data. The '.comm' and '.lcomm' directives of the
    // variable will create the needed csect.
    return;
  }

  // Zero-initialized TLS data with weak or external linkage are not eligible to
  // be put into common csect.
  if (Sec.getKind().isThreadBSS()) {
    Sec.printCsectDirective(OS);
    return;
  }

  // XCOFF debug sections.
  if (Sec.getKind().isMetadata() && Sec.isDwarfSect()) {
    OS << "\n\t.dwsect " << format("0x%" PRIx32, *Sec.getDwarfSubtypeFlags())
       << '\n';
    OS << Sec.getName() << ':' << '\n';
    return;
  }

  report_fatal_error("Printing for this SectionKind is unimplemented.");
}
