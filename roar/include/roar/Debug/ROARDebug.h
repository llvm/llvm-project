//===-- ROARDebug.h - Debug support APIs. ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares several APIs used for implementing ROAR debugging support.
//
//===----------------------------------------------------------------------===//

#ifndef ROAR_DEBUG_ROARDEBUG_H
#define ROAR_DEBUG_ROARDEBUG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>

namespace llvm {
class Constant;
class GlobalValue;
class Type;

namespace roar {
namespace debug_support {

uint64_t ComputeFilenameHash(StringRef Basename);

// A LocationRecordReader is a helper type that is used to access the
// variable-sized location records. The LocationRecordIterator (below) uses a
// reinterpret_cast<> to return objects of these type that are "overlayed" on
// the bytes written by the compiler.
class LocationRecordReader {
public:
  // IsValid returns true if this LocationRecordReader can parse the location
  // record, and false otherwise; the other methods in this class can only be
  // invoked if IsValid() returns true.
  bool IsValid(const LocationRecordReader &end) const;

  // Returns the size (in bytes) of the variable size record that is "backing"
  // this struct. Assumes IsValid().
  size_t RecordSizeBytes() const;

  // Returns the value of this record's LineDelta field. Assumes IsValid().
  uint32_t LineDelta() const;

  // Returns the value of this record's STE field. Assumes IsValid().
  uint64_t STE() const;

private:
  // No instances of this class should be created -- users are expected to
  // reinterpret_cast<> a uint8_t pointer to the record.
  LocationRecordReader() = delete;

  const uint8_t data[1]; // variable size
};

/// An iterator class for traversing the location records for a particular file.
class LocationRecordIterator {
public:
  LocationRecordIterator(const uint8_t *ptr);

  bool operator<(const LocationRecordIterator &) const;

  bool operator==(const LocationRecordIterator &) const;

  bool operator!=(const LocationRecordIterator &) const;

  LocationRecordIterator &operator++();

  const LocationRecordReader *operator->() const;

  const LocationRecordReader &operator*() const;

private:
  const uint8_t *m_ptr;
};

/// A class for building the SourceLocationRecord list for a file.
class LocationRecordBuilder {
public:
  LocationRecordBuilder(Type *Int8Ty,
                        SmallVectorImpl<Constant *> &LocationsBuffer);

  /// Resets the Builder's internal state in preparation to emit the records for
  /// a new file.
  void StartRecord();

  /// Adds a new line to the SourceLocationRecord list. CurrLine must be greater
  /// than the last added line.
  void AddLine(uint32_t NextLine, ArrayRef<GlobalValue *> STEs);

private:
  Type *Int8Ty;
  SmallVectorImpl<Constant *> &LocationsBuffer;
  uint32_t PrevLine;
  SmallString<32> LEBBuffer;
};

} // namespace debug_support
} // namespace roar
} // namespace llvm

#endif // ROAR_DEBUG_ROARDEBUG_H
