//===- ObjCARCRVAttr.h - ObjC ARC Attribute Analysis ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines functions which look for or remove attributes retainRV,
/// claimRV, and rv_marker.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_ANALYSIS_OBJCARCRVATTR_H
#define LLVM_LIB_ANALYSIS_OBJCARCRVATTR_H

#include "llvm/IR/InstrTypes.h"

namespace llvm {
namespace objcarc {

static inline const char *getRVMarkerModuleFlagStr() {
  return "clang.arc.retainAutoreleasedReturnValueMarker";
}

static inline const char *getRVAttrKeyStr() { return "clang.arc.rv"; }

static inline const char *getRVAttrValStr(bool Retain) {
  return Retain ? "retain" : "claim";
}

static inline bool hasRetainRVAttr(const CallBase *CB) {
  return CB->getAttribute(llvm::AttributeList::ReturnIndex, getRVAttrKeyStr())
      .getValueAsString()
      .equals(getRVAttrValStr(true));
}
static inline bool hasClaimRVAttr(const CallBase *CB) {
  return CB->getAttribute(llvm::AttributeList::ReturnIndex, getRVAttrKeyStr())
      .getValueAsString()
      .equals(getRVAttrValStr(false));
}

static inline bool hasRetainRVOrClaimRVAttr(const CallBase *CB) {
  return CB->hasRetAttr(getRVAttrKeyStr());
}

static inline void removeRetainRVOrClaimRVAttr(CallBase *CB) {
  CB->removeAttribute(llvm::AttributeList::ReturnIndex, getRVAttrKeyStr());
}

} // end namespace objcarc
} // end namespace llvm

#endif
