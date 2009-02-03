//===-- SourceLanguage-CFamily.cpp - C family SourceLanguage impl ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SourceLanguage class for the C family of languages
// (K&R C, C89, C99, etc).
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/SourceLanguage.h"
using namespace llvm;

#if 0
namespace {
  struct CSL : public SourceLanguage {
  } TheCSourceLanguageInstance;
}
#endif

const SourceLanguage &SourceLanguage::getCFamilyInstance() {
  return get(0);  // We don't have an implementation for C yet fall back on
                  // generic
}
