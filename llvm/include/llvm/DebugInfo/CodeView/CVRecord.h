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
#include "llvm/ADT/Optional.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/RecordSerialization.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {

namespace codeview {

template <typename Kind> class CVRecord {
public:
  CVRecord() = default;
  CVRecord(Kind K, ArrayRef<uint8_t> Data) : Type(K), RecordData(Data) {}

  uint32_t length() const { return RecordData.size(); }
  Kind kind() const { return Type; }
  ArrayRef<uint8_t> data() const { return RecordData; }

  ArrayRef<uint8_t> content() const {
    return RecordData.drop_front(sizeof(RecordPrefix));
  }

  Optional<uint32_t> hash() const { return Hash; }

  void setHash(uint32_t Value) { Hash = Value; }

  Kind Type;
  ArrayRef<uint8_t> RecordData;
  Optional<uint32_t> Hash;
};

} // end namespace codeview

template <typename Kind>
struct VarStreamArrayExtractor<codeview::CVRecord<Kind>> {
  typedef void ContextType;

  static Error extract(BinaryStreamRef Stream, uint32_t &Len,
                       codeview::CVRecord<Kind> &Item, void *Ctx) {
    using namespace codeview;
    const RecordPrefix *Prefix = nullptr;
    BinaryStreamReader Reader(Stream);
    uint32_t Offset = Reader.getOffset();

    if (auto EC = Reader.readObject(Prefix))
      return EC;
    if (Prefix->RecordLen < 2)
      return make_error<CodeViewError>(cv_error_code::corrupt_record);
    Kind K = static_cast<Kind>(uint16_t(Prefix->RecordKind));

    Reader.setOffset(Offset);
    ArrayRef<uint8_t> RawData;
    if (auto EC =
            Reader.readBytes(RawData, Prefix->RecordLen + sizeof(uint16_t)))
      return EC;
    Item = codeview::CVRecord<Kind>(K, RawData);
    Len = Item.length();
    return Error::success();
  }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_RECORDITERATOR_H
