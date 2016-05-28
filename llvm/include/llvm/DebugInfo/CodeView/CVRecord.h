//===- RecordIterator.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_RECORDITERATOR_H
#define LLVM_DEBUGINFO_CODEVIEW_RECORDITERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

template <typename Kind> struct CVRecord {
  uint32_t Length;
  Kind Type;
  ArrayRef<uint8_t> Data;
};

template <typename Kind> struct VarStreamArrayExtractor<CVRecord<Kind>> {
  Error operator()(const StreamInterface &Stream, uint32_t &Len,
                   CVRecord<Kind> &Item) const {
    const RecordPrefix *Prefix = nullptr;
    StreamReader Reader(Stream);
    if (auto EC = Reader.readObject(Prefix))
      return EC;
    Item.Length = Prefix->RecordLen;
    Item.Type = static_cast<Kind>(uint16_t(Prefix->RecordKind));
    if (auto EC = Reader.readBytes(Item.Data, Item.Length - 2))
      return EC;
    Len = Prefix->RecordLen + 2;
    return Error::success();
  }
};
}
}

#endif
