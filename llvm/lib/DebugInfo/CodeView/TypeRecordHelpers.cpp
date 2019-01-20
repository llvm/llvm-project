//===- TypeRecordHelpers.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/TypeRecordHelpers.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/CodeView/TypeIndexDiscovery.h"
#include "llvm/DebugInfo/CodeView/TypeDeserializer.h"

using namespace llvm;
using namespace llvm::codeview;

template <typename RecordT> static ClassOptions getUdtOptions(CVType CVT) {
  RecordT Record;
  if (auto EC = TypeDeserializer::deserializeAs<RecordT>(CVT, Record)) {
    consumeError(std::move(EC));
    return ClassOptions::None;
  }
  return Record.getOptions();
}

bool llvm::codeview::isUdtForwardRef(CVType CVT) {
  ClassOptions UdtOptions = ClassOptions::None;
  switch (CVT.kind()) {
  case LF_STRUCTURE:
  case LF_CLASS:
  case LF_INTERFACE:
    UdtOptions = getUdtOptions<ClassRecord>(std::move(CVT));
    break;
  case LF_ENUM:
    UdtOptions = getUdtOptions<EnumRecord>(std::move(CVT));
    break;
  case LF_UNION:
    UdtOptions = getUdtOptions<UnionRecord>(std::move(CVT));
    break;
  default:
    return false;
  }
  return (UdtOptions & ClassOptions::ForwardReference) != ClassOptions::None;
}

TypeIndex llvm::codeview::getModifiedType(const CVType &CVT) {
  assert(CVT.kind() == LF_MODIFIER);
  SmallVector<TypeIndex, 1> Refs;
  discoverTypeIndices(CVT, Refs);
  return Refs.front();
}
