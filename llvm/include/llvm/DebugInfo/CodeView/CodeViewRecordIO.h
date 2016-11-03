//===- CodeViewRecordIO.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_CODEVIEWRECORDIO_H
#define LLVM_DEBUGINFO_CODEVIEW_CODEVIEWRECORDIO_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/MSF/StreamWriter.h"
#include "llvm/Support/Error.h"

#include <stdint.h>
#include <type_traits>

namespace llvm {
namespace msf {
class StreamReader;
class StreamWriter;
}
namespace codeview {

class CodeViewRecordIO {
  struct ActiveRecord {
    uint16_t Kind;
  };

public:
  explicit CodeViewRecordIO(msf::StreamReader &Reader) : Reader(&Reader) {}
  explicit CodeViewRecordIO(msf::StreamWriter &Writer) : Writer(&Writer) {}

  Error beginRecord(uint16_t Kind);
  Error endRecord();
  Error mapInteger(TypeIndex &TypeInd);

  bool isReading() const { return Reader != nullptr; }
  bool isWriting() const { return !isReading(); }

  template <typename T> Error mapInteger(T &Value) {
    if (isWriting())
      return Writer->writeInteger(Value);

    return Reader->readInteger(Value);
  }

  template <typename T> Error mapEnum(T &Value) {
    using U = typename std::underlying_type<T>::type;
    U X;
    if (isWriting())
      X = static_cast<U>(Value);

    if (auto EC = mapInteger(X))
      return EC;
    if (isReading())
      Value = static_cast<T>(X);
    return Error::success();
  }

  Error mapEncodedInteger(int64_t &Value);
  Error mapEncodedInteger(uint64_t &Value);
  Error mapEncodedInteger(APSInt &Value);
  Error mapStringZ(StringRef &Value);
  Error mapGuid(StringRef &Guid);

  template <typename SizeType, typename T, typename ElementMapper>
  Error mapVectorN(T &Items, const ElementMapper &Mapper) {
    SizeType Size;
    if (isWriting()) {
      Size = static_cast<SizeType>(Items.size());
      if (auto EC = Writer->writeInteger(Size))
        return EC;

      for (auto &X : Items) {
        if (auto EC = Mapper(*this, X))
          return EC;
      }
    } else {
      if (auto EC = Reader->readInteger(Size))
        return EC;
      for (SizeType I = 0; I < Size; ++I) {
        typename T::value_type Item;
        if (auto EC = Mapper(*this, Item))
          return EC;
        Items.push_back(Item);
      }
    }

    return Error::success();
  }

  template <typename T, typename ElementMapper>
  Error mapVectorTail(T &Items, const ElementMapper &Mapper) {
    if (isWriting()) {
      for (auto &Item : Items) {
        if (auto EC = Mapper(*this, Item))
          return EC;
      }
    } else {
      typename T::value_type Field;
      // Stop when we run out of bytes or we hit record padding bytes.
      while (!Reader->empty() && Reader->peek() < 0xf0 /* LF_PAD0 */) {
        if (auto EC = Mapper(*this, Field))
          return EC;
        Items.push_back(Field);
      }
    }
    return Error::success();
  }

  Error mapByteVectorTail(ArrayRef<uint8_t> &Bytes);

  Error skipPadding();

private:
  Error writeEncodedSignedInteger(const int64_t &Value);
  Error writeEncodedUnsignedInteger(const uint64_t &Value);

  Optional<ActiveRecord> CurrentRecord;

  msf::StreamReader *Reader = nullptr;
  msf::StreamWriter *Writer = nullptr;
};
}
}

#endif
