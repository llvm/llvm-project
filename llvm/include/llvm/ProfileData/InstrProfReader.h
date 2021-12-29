//===- InstrProfReader.h - Instrumented profiling readers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for instrumentation
// based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_INSTRPROFREADER_H
#define LLVM_PROFILEDATA_INSTRPROFREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/InstrProfCorrelator.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/OnDiskHashTable.h"
#include "llvm/Support/SwapByteOrder.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace llvm {

class InstrProfReader;

/// A file format agnostic iterator over profiling data.
class InstrProfIterator {
public:
  using iterator_category = std::input_iterator_tag;
  using value_type = NamedInstrProfRecord;
  using difference_type = std::ptrdiff_t;
  using pointer = value_type *;
  using reference = value_type &;

private:
  InstrProfReader *Reader = nullptr;
  value_type Record;

  void Increment();

public:
  InstrProfIterator() = default;
  InstrProfIterator(InstrProfReader *Reader) : Reader(Reader) { Increment(); }

  InstrProfIterator &operator++() { Increment(); return *this; }
  bool operator==(const InstrProfIterator &RHS) const {
    return Reader == RHS.Reader;
  }
  bool operator!=(const InstrProfIterator &RHS) const {
    return Reader != RHS.Reader;
  }
  value_type &operator*() { return Record; }
  value_type *operator->() { return &Record; }
};

/// Base class and interface for reading profiling data of any known instrprof
/// format. Provides an iterator over NamedInstrProfRecords.
class InstrProfReader {
  instrprof_error LastError = instrprof_error::success;
  std::string LastErrorMsg;

public:
  InstrProfReader() = default;
  virtual ~InstrProfReader() = default;

  /// Read the header.  Required before reading first record.
  virtual Error readHeader() = 0;

  /// Read a single record.
  virtual Error readNextRecord(NamedInstrProfRecord &Record) = 0;

  /// Print binary ids on stream OS.
  virtual Error printBinaryIds(raw_ostream &OS) { return success(); };

  /// Iterator over profile data.
  InstrProfIterator begin() { return InstrProfIterator(this); }
  InstrProfIterator end() { return InstrProfIterator(); }

  virtual bool isIRLevelProfile() const = 0;

  virtual bool hasCSIRLevelProfile() const = 0;

  virtual bool instrEntryBBEnabled() const = 0;

  /// Return true if we must provide debug info to create PGO profiles.
  virtual bool useDebugInfoCorrelate() const { return false; }

  /// Return the PGO symtab. There are three different readers:
  /// Raw, Text, and Indexed profile readers. The first two types
  /// of readers are used only by llvm-profdata tool, while the indexed
  /// profile reader is also used by llvm-cov tool and the compiler (
  /// backend or frontend). Since creating PGO symtab can create
  /// significant runtime and memory overhead (as it touches data
  /// for the whole program), InstrProfSymtab for the indexed profile
  /// reader should be created on demand and it is recommended to be
  /// only used for dumping purpose with llvm-proftool, not with the
  /// compiler.
  virtual InstrProfSymtab &getSymtab() = 0;

  /// Compute the sum of counts and return in Sum.
  void accumulateCounts(CountSumOrPercent &Sum, bool IsCS);

protected:
  std::unique_ptr<InstrProfSymtab> Symtab;

  /// Set the current error and return same.
  Error error(instrprof_error Err, const std::string &ErrMsg = "") {
    LastError = Err;
    LastErrorMsg = ErrMsg;
    if (Err == instrprof_error::success)
      return Error::success();
    return make_error<InstrProfError>(Err, ErrMsg);
  }

  Error error(Error &&E) {
    handleAllErrors(std::move(E), [&](const InstrProfError &IPE) {
      LastError = IPE.get();
      LastErrorMsg = IPE.getMessage();
    });
    return make_error<InstrProfError>(LastError, LastErrorMsg);
  }

  /// Clear the current error and return a successful one.
  Error success() { return error(instrprof_error::success); }

public:
  /// Return true if the reader has finished reading the profile data.
  bool isEOF() { return LastError == instrprof_error::eof; }

  /// Return true if the reader encountered an error reading profiling data.
  bool hasError() { return LastError != instrprof_error::success && !isEOF(); }

  /// Get the current error.
  Error getError() {
    if (hasError())
      return make_error<InstrProfError>(LastError, LastErrorMsg);
    return Error::success();
  }

  /// Factory method to create an appropriately typed reader for the given
  /// instrprof file.
  static Expected<std::unique_ptr<InstrProfReader>>
  create(const Twine &Path, const InstrProfCorrelator *Correlator = nullptr);

  static Expected<std::unique_ptr<InstrProfReader>>
  create(std::unique_ptr<MemoryBuffer> Buffer,
         const InstrProfCorrelator *Correlator = nullptr);
};

/// Reader for the simple text based instrprof format.
///
/// This format is a simple text format that's suitable for test data. Records
/// are separated by one or more blank lines, and record fields are separated by
/// new lines.
///
/// Each record consists of a function name, a function hash, a number of
/// counters, and then each counter value, in that order.
class TextInstrProfReader : public InstrProfReader {
private:
  /// The profile data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// Iterator over the profile data.
  line_iterator Line;
  bool IsIRLevelProfile = false;
  bool HasCSIRLevelProfile = false;
  bool InstrEntryBBEnabled = false;

  Error readValueProfileData(InstrProfRecord &Record);

public:
  TextInstrProfReader(std::unique_ptr<MemoryBuffer> DataBuffer_)
      : DataBuffer(std::move(DataBuffer_)), Line(*DataBuffer, true, '#') {}
  TextInstrProfReader(const TextInstrProfReader &) = delete;
  TextInstrProfReader &operator=(const TextInstrProfReader &) = delete;

  /// Return true if the given buffer is in text instrprof format.
  static bool hasFormat(const MemoryBuffer &Buffer);

  bool isIRLevelProfile() const override { return IsIRLevelProfile; }

  bool hasCSIRLevelProfile() const override { return HasCSIRLevelProfile; }

  bool instrEntryBBEnabled() const override { return InstrEntryBBEnabled; }

  /// Read the header.
  Error readHeader() override;

  /// Read a single record.
  Error readNextRecord(NamedInstrProfRecord &Record) override;

  InstrProfSymtab &getSymtab() override {
    assert(Symtab.get());
    return *Symtab.get();
  }
};

/// Reader for the raw instrprof binary format from runtime.
///
/// This format is a raw memory dump of the instrumentation-based profiling data
/// from the runtime.  It has no index.
///
/// Templated on the unsigned type whose size matches pointers on the platform
/// that wrote the profile.
template <class IntPtrT>
class RawInstrProfReader : public InstrProfReader {
private:
  /// The profile data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// If available, this hold the ProfileData array used to correlate raw
  /// instrumentation data to their functions.
  const InstrProfCorrelatorImpl<IntPtrT> *Correlator;
  bool ShouldSwapBytes;
  // The value of the version field of the raw profile data header. The lower 56
  // bits specifies the format version and the most significant 8 bits specify
  // the variant types of the profile.
  uint64_t Version;
  uint64_t CountersDelta;
  uint64_t NamesDelta;
  const RawInstrProf::ProfileData<IntPtrT> *Data;
  const RawInstrProf::ProfileData<IntPtrT> *DataEnd;
  const char *CountersStart;
  const char *CountersEnd;
  const char *NamesStart;
  const char *NamesEnd;
  // After value profile is all read, this pointer points to
  // the header of next profile data (if exists)
  const uint8_t *ValueDataStart;
  uint32_t ValueKindLast;
  uint32_t CurValueDataSize;

  uint64_t BinaryIdsSize;
  const uint8_t *BinaryIdsStart;

public:
  RawInstrProfReader(std::unique_ptr<MemoryBuffer> DataBuffer,
                     const InstrProfCorrelator *Correlator)
      : DataBuffer(std::move(DataBuffer)),
        Correlator(dyn_cast_or_null<const InstrProfCorrelatorImpl<IntPtrT>>(
            Correlator)) {}
  RawInstrProfReader(const RawInstrProfReader &) = delete;
  RawInstrProfReader &operator=(const RawInstrProfReader &) = delete;

  static bool hasFormat(const MemoryBuffer &DataBuffer);
  Error readHeader() override;
  Error readNextRecord(NamedInstrProfRecord &Record) override;
  Error printBinaryIds(raw_ostream &OS) override;

  bool isIRLevelProfile() const override {
    return (Version & VARIANT_MASK_IR_PROF) != 0;
  }

  bool hasCSIRLevelProfile() const override {
    return (Version & VARIANT_MASK_CSIR_PROF) != 0;
  }

  bool instrEntryBBEnabled() const override {
    return (Version & VARIANT_MASK_INSTR_ENTRY) != 0;
  }

  bool useDebugInfoCorrelate() const override {
    return (Version & VARIANT_MASK_DBG_CORRELATE) != 0;
  }

  InstrProfSymtab &getSymtab() override {
    assert(Symtab.get());
    return *Symtab.get();
  }

private:
  Error createSymtab(InstrProfSymtab &Symtab);
  Error readNextHeader(const char *CurrentPos);
  Error readHeader(const RawInstrProf::Header &Header);

  template <class IntT> IntT swap(IntT Int) const {
    return ShouldSwapBytes ? sys::getSwappedBytes(Int) : Int;
  }

  support::endianness getDataEndianness() const {
    support::endianness HostEndian = getHostEndianness();
    if (!ShouldSwapBytes)
      return HostEndian;
    if (HostEndian == support::little)
      return support::big;
    else
      return support::little;
  }

  inline uint8_t getNumPaddingBytes(uint64_t SizeInBytes) {
    return 7 & (sizeof(uint64_t) - SizeInBytes % sizeof(uint64_t));
  }

  Error readName(NamedInstrProfRecord &Record);
  Error readFuncHash(NamedInstrProfRecord &Record);
  Error readRawCounts(InstrProfRecord &Record);
  Error readValueProfilingData(InstrProfRecord &Record);
  bool atEnd() const { return Data == DataEnd; }

  void advanceData() {
    // `CountersDelta` is a constant zero when using debug info correlation.
    if (!Correlator) {
      // The initial CountersDelta is the in-memory address difference between
      // the data and counts sections:
      // start(__llvm_prf_cnts) - start(__llvm_prf_data)
      // As we advance to the next record, we maintain the correct CountersDelta
      // with respect to the next record.
      CountersDelta -= sizeof(*Data);
    }
    Data++;
    ValueDataStart += CurValueDataSize;
  }

  const char *getNextHeaderPos() const {
      assert(atEnd());
      return (const char *)ValueDataStart;
  }

  StringRef getName(uint64_t NameRef) const {
    return Symtab->getFuncName(swap(NameRef));
  }

  int getCounterTypeSize() const { return sizeof(uint64_t); }
};

using RawInstrProfReader32 = RawInstrProfReader<uint32_t>;
using RawInstrProfReader64 = RawInstrProfReader<uint64_t>;

namespace IndexedInstrProf {

enum class HashT : uint32_t;

} // end namespace IndexedInstrProf

/// Trait for lookups into the on-disk hash table for the binary instrprof
/// format.
class InstrProfLookupTrait {
  std::vector<NamedInstrProfRecord> DataBuffer;
  IndexedInstrProf::HashT HashType;
  unsigned FormatVersion;
  // Endianness of the input value profile data.
  // It should be LE by default, but can be changed
  // for testing purpose.
  support::endianness ValueProfDataEndianness = support::little;

public:
  InstrProfLookupTrait(IndexedInstrProf::HashT HashType, unsigned FormatVersion)
      : HashType(HashType), FormatVersion(FormatVersion) {}

  using data_type = ArrayRef<NamedInstrProfRecord>;

  using internal_key_type = StringRef;
  using external_key_type = StringRef;
  using hash_value_type = uint64_t;
  using offset_type = uint64_t;

  static bool EqualKey(StringRef A, StringRef B) { return A == B; }
  static StringRef GetInternalKey(StringRef K) { return K; }
  static StringRef GetExternalKey(StringRef K) { return K; }

  hash_value_type ComputeHash(StringRef K);

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&D) {
    using namespace support;

    offset_type KeyLen = endian::readNext<offset_type, little, unaligned>(D);
    offset_type DataLen = endian::readNext<offset_type, little, unaligned>(D);
    return std::make_pair(KeyLen, DataLen);
  }

  StringRef ReadKey(const unsigned char *D, offset_type N) {
    return StringRef((const char *)D, N);
  }

  bool readValueProfilingData(const unsigned char *&D,
                              const unsigned char *const End);
  data_type ReadData(StringRef K, const unsigned char *D, offset_type N);

  // Used for testing purpose only.
  void setValueProfDataEndianness(support::endianness Endianness) {
    ValueProfDataEndianness = Endianness;
  }
};

struct InstrProfReaderIndexBase {
  virtual ~InstrProfReaderIndexBase() = default;

  // Read all the profile records with the same key pointed to the current
  // iterator.
  virtual Error getRecords(ArrayRef<NamedInstrProfRecord> &Data) = 0;

  // Read all the profile records with the key equal to FuncName
  virtual Error getRecords(StringRef FuncName,
                                     ArrayRef<NamedInstrProfRecord> &Data) = 0;
  virtual void advanceToNextKey() = 0;
  virtual bool atEnd() const = 0;
  virtual void setValueProfDataEndianness(support::endianness Endianness) = 0;
  virtual uint64_t getVersion() const = 0;
  virtual bool isIRLevelProfile() const = 0;
  virtual bool hasCSIRLevelProfile() const = 0;
  virtual bool instrEntryBBEnabled() const = 0;
  virtual Error populateSymtab(InstrProfSymtab &) = 0;
};

using OnDiskHashTableImplV3 =
    OnDiskIterableChainedHashTable<InstrProfLookupTrait>;

template <typename HashTableImpl>
class InstrProfReaderItaniumRemapper;

template <typename HashTableImpl>
class InstrProfReaderIndex : public InstrProfReaderIndexBase {
private:
  std::unique_ptr<HashTableImpl> HashTable;
  typename HashTableImpl::data_iterator RecordIterator;
  uint64_t FormatVersion;

  friend class InstrProfReaderItaniumRemapper<HashTableImpl>;

public:
  InstrProfReaderIndex(const unsigned char *Buckets,
                       const unsigned char *const Payload,
                       const unsigned char *const Base,
                       IndexedInstrProf::HashT HashType, uint64_t Version);
  ~InstrProfReaderIndex() override = default;

  Error getRecords(ArrayRef<NamedInstrProfRecord> &Data) override;
  Error getRecords(StringRef FuncName,
                   ArrayRef<NamedInstrProfRecord> &Data) override;
  void advanceToNextKey() override { RecordIterator++; }

  bool atEnd() const override {
    return RecordIterator == HashTable->data_end();
  }

  void setValueProfDataEndianness(support::endianness Endianness) override {
    HashTable->getInfoObj().setValueProfDataEndianness(Endianness);
  }

  uint64_t getVersion() const override { return GET_VERSION(FormatVersion); }

  bool isIRLevelProfile() const override {
    return (FormatVersion & VARIANT_MASK_IR_PROF) != 0;
  }

  bool hasCSIRLevelProfile() const override {
    return (FormatVersion & VARIANT_MASK_CSIR_PROF) != 0;
  }

  bool instrEntryBBEnabled() const override {
    return (FormatVersion & VARIANT_MASK_INSTR_ENTRY) != 0;
  }

  Error populateSymtab(InstrProfSymtab &Symtab) override {
    return Symtab.create(HashTable->keys());
  }
};

/// Name matcher supporting fuzzy matching of symbol names to names in profiles.
class InstrProfReaderRemapper {
public:
  virtual ~InstrProfReaderRemapper() {}
  virtual Error populateRemappings() { return Error::success(); }
  virtual Error getRecords(StringRef FuncName,
                           ArrayRef<NamedInstrProfRecord> &Data) = 0;
};

/// Reader for the indexed binary instrprof format.
class IndexedInstrProfReader : public InstrProfReader {
private:
  /// The profile data file contents.
  std::unique_ptr<MemoryBuffer> DataBuffer;
  /// The profile remapping file contents.
  std::unique_ptr<MemoryBuffer> RemappingBuffer;
  /// The index into the profile data.
  std::unique_ptr<InstrProfReaderIndexBase> Index;
  /// The profile remapping file contents.
  std::unique_ptr<InstrProfReaderRemapper> Remapper;
  /// Profile summary data.
  std::unique_ptr<ProfileSummary> Summary;
  /// Context sensitive profile summary data.
  std::unique_ptr<ProfileSummary> CS_Summary;
  // Index to the current record in the record array.
  unsigned RecordIndex;

  // Read the profile summary. Return a pointer pointing to one byte past the
  // end of the summary data if it exists or the input \c Cur.
  // \c UseCS indicates whether to use the context-sensitive profile summary.
  const unsigned char *readSummary(IndexedInstrProf::ProfVersion Version,
                                   const unsigned char *Cur, bool UseCS);

public:
  IndexedInstrProfReader(
      std::unique_ptr<MemoryBuffer> DataBuffer,
      std::unique_ptr<MemoryBuffer> RemappingBuffer = nullptr)
      : DataBuffer(std::move(DataBuffer)),
        RemappingBuffer(std::move(RemappingBuffer)), RecordIndex(0) {}
  IndexedInstrProfReader(const IndexedInstrProfReader &) = delete;
  IndexedInstrProfReader &operator=(const IndexedInstrProfReader &) = delete;

  /// Return the profile version.
  uint64_t getVersion() const { return Index->getVersion(); }
  bool isIRLevelProfile() const override { return Index->isIRLevelProfile(); }
  bool hasCSIRLevelProfile() const override {
    return Index->hasCSIRLevelProfile();
  }

  bool instrEntryBBEnabled() const override {
    return Index->instrEntryBBEnabled();
  }

  /// Return true if the given buffer is in an indexed instrprof format.
  static bool hasFormat(const MemoryBuffer &DataBuffer);

  /// Read the file header.
  Error readHeader() override;
  /// Read a single record.
  Error readNextRecord(NamedInstrProfRecord &Record) override;

  /// Return the NamedInstrProfRecord associated with FuncName and FuncHash
  Expected<InstrProfRecord> getInstrProfRecord(StringRef FuncName,
                                               uint64_t FuncHash);

  /// Fill Counts with the profile data for the given function name.
  Error getFunctionCounts(StringRef FuncName, uint64_t FuncHash,
                          std::vector<uint64_t> &Counts);

  /// Return the maximum of all known function counts.
  /// \c UseCS indicates whether to use the context-sensitive count.
  uint64_t getMaximumFunctionCount(bool UseCS) {
    if (UseCS) {
      assert(CS_Summary && "No context sensitive profile summary");
      return CS_Summary->getMaxFunctionCount();
    } else {
      assert(Summary && "No profile summary");
      return Summary->getMaxFunctionCount();
    }
  }

  /// Factory method to create an indexed reader.
  static Expected<std::unique_ptr<IndexedInstrProfReader>>
  create(const Twine &Path, const Twine &RemappingPath = "");

  static Expected<std::unique_ptr<IndexedInstrProfReader>>
  create(std::unique_ptr<MemoryBuffer> Buffer,
         std::unique_ptr<MemoryBuffer> RemappingBuffer = nullptr);

  // Used for testing purpose only.
  void setValueProfDataEndianness(support::endianness Endianness) {
    Index->setValueProfDataEndianness(Endianness);
  }

  // See description in the base class. This interface is designed
  // to be used by llvm-profdata (for dumping). Avoid using this when
  // the client is the compiler.
  InstrProfSymtab &getSymtab() override;

  /// Return the profile summary.
  /// \c UseCS indicates whether to use the context-sensitive summary.
  ProfileSummary &getSummary(bool UseCS) {
    if (UseCS) {
      assert(CS_Summary && "No context sensitive summary");
      return *(CS_Summary.get());
    } else {
      assert(Summary && "No profile summary");
      return *(Summary.get());
    }
  }
};

} // end namespace llvm

#endif // LLVM_PROFILEDATA_INSTRPROFREADER_H
