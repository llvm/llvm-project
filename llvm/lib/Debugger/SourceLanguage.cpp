//===-- SourceLanguage.cpp - Implement the SourceLanguage class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SourceLanguage class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/SourceLanguage.h"
#include "llvm/Debugger/ProgramInfo.h"
using namespace llvm;

const SourceLanguage &SourceLanguage::get(unsigned ID) {
  switch (ID) {
  case 1:  // DW_LANG_C89
  case 2:  // DW_LANG_C
  case 12: // DW_LANG_C99
    return getCFamilyInstance();

  case 4:  // DW_LANG_C_plus_plus
    return getCPlusPlusInstance();

  case 3:  // DW_LANG_Ada83
  case 5:  // DW_LANG_Cobol74
  case 6:  // DW_LANG_Cobol85
  case 7:  // DW_LANG_Fortran77
  case 8:  // DW_LANG_Fortran90
  case 9:  // DW_LANG_Pascal83
  case 10: // DW_LANG_Modula2
  case 11: // DW_LANG_Java
  case 13: // DW_LANG_Ada95
  case 14: // DW_LANG_Fortran95
  default:
    return getUnknownLanguageInstance();
  }
}


SourceFileInfo *
SourceLanguage::createSourceFileInfo(const GlobalVariable *Desc,
                                     ProgramInfo &PI) const {
  return new SourceFileInfo(Desc, *this);
}

SourceFunctionInfo *
SourceLanguage::createSourceFunctionInfo(const GlobalVariable *Desc,
                                         ProgramInfo &PI) const {
  return new SourceFunctionInfo(PI, Desc);
}
