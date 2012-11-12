//===-- DIContext.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DIContext.h"
#include "DWARFContext.h"
using namespace llvm;

DIContext::~DIContext() {}

DIContext *DIContext::getDWARFContext(bool isLittleEndian,
                                      StringRef infoSection,
                                      StringRef abbrevSection,
                                      StringRef aRangeSection,
                                      StringRef lineSection,
                                      StringRef stringSection,
                                      StringRef rangeSection,
                                      const RelocAddrMap &Map) {
  return new DWARFContextInMemory(isLittleEndian, infoSection, abbrevSection,
                                  aRangeSection, lineSection, stringSection,
                                  rangeSection, Map);
}
