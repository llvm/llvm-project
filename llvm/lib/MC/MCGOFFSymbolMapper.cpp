//===- MCGOFFSymbolMapper.cpp - Maps MC section/symbol to GOFF symbols ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps a section or a symbol to the GOFF symbols it is composed of, and their
// attributes.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCGOFFSymbolMapper.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Path.h"

using namespace llvm;

namespace {
const StringLiteral CODE[2]{"C_CODE", "C_CODE64"};
const StringLiteral WSA[2]{"C_WSA", "C_WSA64"};
const StringLiteral PPA2[2]{"C_@@PPA2", "C_@@QPPA2"};

const GOFF::ESDAmode AMODE[2]{GOFF::ESD_AMODE_ANY, GOFF::ESD_AMODE_64};
const GOFF::ESDRmode RMODE[2]{GOFF::ESD_RMODE_31, GOFF::ESD_RMODE_64};

const GOFF::ESDLinkageType LINKAGE[2]{GOFF::ESD_LT_OS, GOFF::ESD_LT_XPLink};
} // namespace

GOFFSymbolMapper::GOFFSymbolMapper(MCContext &Ctx) : Ctx(Ctx) {
  IsCsectCodeNameEmpty = true;
  Is64Bit = true;
  UsesXPLINK = true;
}

GOFFSymbolMapper::GOFFSymbolMapper(MCAssembler &Asm)
    : GOFFSymbolMapper(Asm.getContext()) {
  if (!Asm.getWriter().getFileNames().empty())
    BaseName =
        sys::path::stem((*(Asm.getWriter().getFileNames().begin())).first);
}

void GOFFSymbolMapper::determineRootSD(StringRef CSectCodeName) {
  IsCsectCodeNameEmpty = CSectCodeName.empty();
  if (IsCsectCodeNameEmpty) {
    RootSDName = BaseName.str().append("#C");
  } else {
    RootSDName = CSectCodeName;
  }
  RootSDAttributes = {GOFF::ESD_TA_Rent, IsCsectCodeNameEmpty
                                             ? GOFF::ESD_BSC_Section
                                             : GOFF::ESD_BSC_Unspecified};
}

llvm::StringRef GOFFSymbolMapper::getRootSDName() const { return RootSDName; }

const SDAttr &GOFFSymbolMapper::getRootSD() const { return RootSDAttributes; }

std::pair<GOFFSectionData, bool>
GOFFSymbolMapper::getSection(const MCSectionGOFF &Section) {
  // Look up GOFFSection from name in MCSectionGOFF.
  // Customize result, e.g. csect names, 32/64 bit, etc.
  GOFFSectionData GOFFSec;
  if (Section.getName() == ".text") {
    GOFFSec.SDName = RootSDName;
    GOFFSec.SDAttributes = RootSDAttributes;
    GOFFSec.IsSDRootSD = true;
    GOFFSec.EDName = CODE[Is64Bit];
    // The GOFF alignment is encoded as log_2 value.
    uint8_t Log = Log2(Section.getAlign());
    assert(Log <= GOFF::ESD_ALIGN_4Kpage && "Alignment too large");
    GOFFSec.EDAttributes = {true,
                            GOFF::ESD_EXE_CODE,
                            AMODE[Is64Bit],
                            RMODE[Is64Bit],
                            GOFF::ESD_NS_NormalName,
                            GOFF::ESD_TS_ByteOriented,
                            GOFF::ESD_BA_Concatenate,
                            GOFF::ESD_LB_Initial,
                            GOFF::ESD_RQ_0,
                            static_cast<GOFF::ESDAlignment>(Log)};
    GOFFSec.LDorPRName = GOFFSec.SDName;
    GOFFSec.LDAttributes = {false,
                            GOFF::ESD_EXE_CODE,
                            GOFF::ESD_NS_NormalName,
                            GOFF::ESD_BST_Strong,
                            LINKAGE[UsesXPLINK],
                            AMODE[Is64Bit],
                            IsCsectCodeNameEmpty ? GOFF::ESD_BSC_Section
                                                 : GOFF::ESD_BSC_Library};
    GOFFSec.Tag = GOFFSectionData::LD;
  } else if (Section.getName() == ".ada") {
    assert(!RootSDName.empty() && "RootSD must be defined already");
    GOFFSec.SDName = RootSDName;
    GOFFSec.SDAttributes = RootSDAttributes;
    GOFFSec.IsSDRootSD = true;
    GOFFSec.EDName = WSA[Is64Bit];
    GOFFSec.EDAttributes = {false,
                            GOFF::ESD_EXE_DATA,
                            AMODE[Is64Bit],
                            RMODE[Is64Bit],
                            GOFF::ESD_NS_Parts,
                            GOFF::ESD_TS_ByteOriented,
                            GOFF::ESD_BA_Merge,
                            GOFF::ESD_LB_Deferred,
                            GOFF::ESD_RQ_1,
                            Is64Bit ? GOFF::ESD_ALIGN_Quadword
                                    : GOFF::ESD_ALIGN_Doubleword};
    ADALDName = BaseName.str().append("#S");
    GOFFSec.LDorPRName = ADALDName;
    GOFFSec.PRAttributes = {false,
                            false,
                            GOFF::ESD_EXE_DATA,
                            GOFF::ESD_NS_Parts,
                            GOFF::ESD_LT_XPLink,
                            AMODE[Is64Bit],
                            GOFF::ESD_BSC_Section,
                            GOFF::ESD_DSS_NoWarning,
                            Is64Bit ? GOFF::ESD_ALIGN_Quadword
                                    : GOFF::ESD_ALIGN_Doubleword,
                            0};
    GOFFSec.Tag = GOFFSectionData::PR;
  } else if (Section.getName().starts_with(".gcc_exception_table")) {
    GOFFSec.SDName = RootSDName;
    GOFFSec.SDAttributes = RootSDAttributes;
    GOFFSec.IsSDRootSD = true;
    GOFFSec.EDName = WSA[Is64Bit];
    GOFFSec.EDAttributes = {false,
                            GOFF::ESD_EXE_DATA,
                            AMODE[Is64Bit],
                            RMODE[Is64Bit],
                            GOFF::ESD_NS_Parts,
                            GOFF::ESD_TS_ByteOriented,
                            GOFF::ESD_BA_Merge,
                            UsesXPLINK ? GOFF::ESD_LB_Initial
                                       : GOFF::ESD_LB_Deferred,
                            GOFF::ESD_RQ_0,
                            GOFF::ESD_ALIGN_Doubleword};
    GOFFSec.LDorPRName = Section.getName();
    GOFFSec.PRAttributes = {true,
                            false,
                            GOFF::ESD_EXE_Unspecified,
                            GOFF::ESD_NS_Parts,
                            LINKAGE[UsesXPLINK],
                            AMODE[Is64Bit],
                            GOFF::ESD_BSC_Section,
                            GOFF::ESD_DSS_NoWarning,
                            GOFF::ESD_ALIGN_Fullword,
                            0};
    GOFFSec.Tag = GOFFSectionData::PR;
  } else if (Section.getName() == ".ppa2list") {
    GOFFSec.SDName = RootSDName;
    GOFFSec.SDAttributes = RootSDAttributes;
    GOFFSec.IsSDRootSD = true;
    GOFFSec.EDName = PPA2[Is64Bit];
    GOFFSec.EDAttributes = {true,
                            GOFF::ESD_EXE_DATA,
                            AMODE[Is64Bit],
                            RMODE[Is64Bit],
                            GOFF::ESD_NS_Parts,
                            GOFF::ESD_TS_ByteOriented,
                            GOFF::ESD_BA_Merge,
                            GOFF::ESD_LB_Initial,
                            GOFF::ESD_RQ_0,
                            GOFF::ESD_ALIGN_Doubleword};
    GOFFSec.LDorPRName = ".&ppa2";
    GOFFSec.PRAttributes = {true,
                            false,
                            GOFF::ESD_EXE_Unspecified,
                            GOFF::ESD_NS_Parts,
                            GOFF::ESD_LT_OS,
                            AMODE[Is64Bit],
                            GOFF::ESD_BSC_Section,
                            GOFF::ESD_DSS_NoWarning,
                            GOFF::ESD_ALIGN_Doubleword,
                            0};
    GOFFSec.Tag = GOFFSectionData::PR;
  } else if (Section.getName() == ".idrl") {
    GOFFSec.SDName = RootSDName;
    GOFFSec.SDAttributes = RootSDAttributes;
    GOFFSec.IsSDRootSD = true;
    GOFFSec.EDName = "B_IDRL";
    GOFFSec.EDAttributes = {true,
                            GOFF::ESD_EXE_Unspecified,
                            AMODE[Is64Bit],
                            RMODE[Is64Bit],
                            GOFF::ESD_NS_NormalName,
                            GOFF::ESD_TS_Structured,
                            GOFF::ESD_BA_Concatenate,
                            GOFF::ESD_LB_NoLoad,
                            GOFF::ESD_RQ_0,
                            GOFF::ESD_ALIGN_Doubleword};
    GOFFSec.Tag = GOFFSectionData::None;
  } else
    return std::pair(GOFFSec, false);
  return std::pair(GOFFSec, true);
}
