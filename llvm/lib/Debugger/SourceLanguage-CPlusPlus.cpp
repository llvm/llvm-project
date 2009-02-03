//===-- SourceLanguage-CPlusPlus.cpp - C++ SourceLanguage impl ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SourceLanguage class for the C++ language.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/SourceLanguage.h"
using namespace llvm;

#if 0
namespace {
  struct CPPSL : public SourceLanguage {
  } TheCPlusPlusLanguageInstance;
}
#endif

const SourceLanguage &SourceLanguage::getCPlusPlusInstance() {
  return get(0);  // We don't have an implementation for C yet fall back on
                  // generic
}
