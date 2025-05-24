//===- lib/MC/GOFFObjectWriter.cpp - GOFF File Writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements GOFF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "goff-writer"

namespace {

// The standard System/390 convention is to name the high-order (leftmost) bit
// in a byte as bit zero. The Flags type helps to set bits in a byte according
// to this numeration order.
class Flags {
  uint8_t Val;

  constexpr static uint8_t bits(uint8_t BitIndex, uint8_t Length, uint8_t Value,
                                uint8_t OldValue) {
    assert(BitIndex < 8 && "Bit index out of bounds!");
    assert(Length + BitIndex <= 8 && "Bit length too long!");

    uint8_t Mask = ((1 << Length) - 1) << (8 - BitIndex - Length);
    Value = Value << (8 - BitIndex - Length);
    assert((Value & Mask) == Value && "Bits set outside of range!");

    return (OldValue & ~Mask) | Value;
  }

public:
  constexpr Flags() : Val(0) {}
  constexpr Flags(uint8_t BitIndex, uint8_t Length, uint8_t Value)
      : Val(bits(BitIndex, Length, Value, 0)) {}

  void set(uint8_t BitIndex, uint8_t Length, uint8_t Value) {
    Val = bits(BitIndex, Length, Value, Val);
  }

  constexpr operator uint8_t() const { return Val; }
};

// Common flag values on records.

// Flag: This record is continued.
constexpr uint8_t RecContinued = Flags(7, 1, 1);

// Flag: This record is a continuation.
constexpr uint8_t RecContinuation = Flags(6, 1, 1);

// The GOFFOstream is responsible to write the data into the fixed physical
// records of the format. A user of this class announces the begin of a new
// logical record. While writing the payload, the physical records are created
// for the data. Possible fill bytes at the end of a physical record are written
// automatically. In principle, the GOFFOstream is agnostic of the endianness of
// the payload. However, it also supports writing data in big endian byte order.
//
// The physical records use the flag field to indicate if the there is a
// successor and predecessor record. To be able to set these flags while
// writing, the basic implementation idea is to always buffer the last seen
// physical record.
class GOFFOstream {
  /// The underlying raw_pwrite_stream.
  raw_pwrite_stream &OS;

  /// The number of logical records emitted so far.
  uint32_t LogicalRecords = 0;

  /// The number of physical records emitted so far.
  uint32_t PhysicalRecords = 0;

  /// The size of the buffer. Same as the payload size of a physical record.
  static constexpr uint8_t BufferSize = GOFF::PayloadLength;

  /// Current position in buffer.
  char *BufferPtr = Buffer;

  /// Static allocated buffer for the stream.
  char Buffer[BufferSize];

  /// The type of the current logical record, and the flags (aka continued and
  /// continuation indicators) for the previous (physical) record.
  uint8_t TypeAndFlags = 0;

public:
  GOFFOstream(raw_pwrite_stream &OS);
  ~GOFFOstream();

  raw_pwrite_stream &getOS() { return OS; }
  size_t getWrittenSize() const { return PhysicalRecords * GOFF::RecordLength; }
  uint32_t getNumLogicalRecords() { return LogicalRecords; }

  /// Write the specified bytes.
  void write(const char *Ptr, size_t Size);

  /// Write zeroes, up to a maximum of 16 bytes.
  void write_zeros(unsigned NumZeros);

  /// Support for endian-specific data.
  template <typename value_type> void writebe(value_type Value) {
    Value =
        support::endian::byte_swap<value_type>(Value, llvm::endianness::big);
    write((const char *)&Value, sizeof(value_type));
  }

  /// Begin a new logical record. Implies finalizing the previous record.
  void newRecord(GOFF::RecordType Type);

  /// Ends a logical record.
  void finalizeRecord();

private:
  /// Updates the continued/continuation flags, and writes the record prefix of
  /// a physical record.
  void updateFlagsAndWritePrefix(bool IsContinued);

  /// Returns the remaining size in the buffer.
  size_t getRemainingSize();
};
} // namespace

GOFFOstream::GOFFOstream(raw_pwrite_stream &OS) : OS(OS) {}

GOFFOstream::~GOFFOstream() { finalizeRecord(); }

void GOFFOstream::updateFlagsAndWritePrefix(bool IsContinued) {
  // Update the flags based on the previous state and the flag IsContinued.
  if (TypeAndFlags & RecContinued)
    TypeAndFlags |= RecContinuation;
  if (IsContinued)
    TypeAndFlags |= RecContinued;
  else
    TypeAndFlags &= ~RecContinued;

  OS << static_cast<unsigned char>(GOFF::PTVPrefix) // Record Type
     << static_cast<unsigned char>(TypeAndFlags)    // Continuation
     << static_cast<unsigned char>(0);              // Version

  ++PhysicalRecords;
}

size_t GOFFOstream::getRemainingSize() {
  return size_t(&Buffer[BufferSize] - BufferPtr);
}

void GOFFOstream::write(const char *Ptr, size_t Size) {
  size_t RemainingSize = getRemainingSize();

  // Data fits into the buffer.
  if (LLVM_LIKELY(Size <= RemainingSize)) {
    memcpy(BufferPtr, Ptr, Size);
    BufferPtr += Size;
    return;
  }

  // Otherwise the buffer is partially filled or full, and data does not fit
  // into it.
  updateFlagsAndWritePrefix(/*IsContinued=*/true);
  OS.write(Buffer, size_t(BufferPtr - Buffer));
  if (RemainingSize > 0) {
    OS.write(Ptr, RemainingSize);
    Ptr += RemainingSize;
    Size -= RemainingSize;
  }

  while (Size > BufferSize) {
    updateFlagsAndWritePrefix(/*IsContinued=*/true);
    OS.write(Ptr, BufferSize);
    Ptr += BufferSize;
    Size -= BufferSize;
  }

  // The remaining bytes fit into the buffer.
  memcpy(Buffer, Ptr, Size);
  BufferPtr = &Buffer[Size];
}

void GOFFOstream::write_zeros(unsigned NumZeros) {
  assert(NumZeros <= 16 && "Range for zeros too large");

  // Handle the common case first: all fits in the buffer.
  size_t RemainingSize = getRemainingSize();
  if (LLVM_LIKELY(RemainingSize >= NumZeros)) {
    memset(BufferPtr, 0, NumZeros);
    BufferPtr += NumZeros;
    return;
  }

  // Otherwise some field value is cleared.
  static char Zeros[16] = {
      0,
  };
  write(Zeros, NumZeros);
}

void GOFFOstream::newRecord(GOFF::RecordType Type) {
  finalizeRecord();
  TypeAndFlags = Type << 4;
  ++LogicalRecords;
}

void GOFFOstream::finalizeRecord() {
  if (Buffer == BufferPtr)
    return;
  updateFlagsAndWritePrefix(/*IsContinued=*/false);
  OS.write(Buffer, size_t(BufferPtr - Buffer));
  OS.write_zeros(getRemainingSize());
  BufferPtr = Buffer;
}

namespace {

class GOFFObjectWriter : public MCObjectWriter {
  // The target specific GOFF writer instance.
  std::unique_ptr<MCGOFFObjectTargetWriter> TargetObjectWriter;

  // The stream used to write the GOFF records.
  GOFFOstream OS;

public:
  GOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                   raw_pwrite_stream &OS)
      : TargetObjectWriter(std::move(MOTW)), OS(OS) {}

  ~GOFFObjectWriter() override {}

  // Write GOFF records.
  void writeHeader();
  void writeEnd();

  // Implementation of the MCObjectWriter interface.
  void recordRelocation(const MCFragment &F, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override {}
  uint64_t writeObject(MCAssembler &Asm) override;
};
} // end anonymous namespace

void GOFFObjectWriter::writeHeader() {
  OS.newRecord(GOFF::RT_HDR);
  OS.write_zeros(1);       // Reserved
  OS.writebe<uint32_t>(0); // Target Hardware Environment
  OS.writebe<uint32_t>(0); // Target Operating System Environment
  OS.write_zeros(2);       // Reserved
  OS.writebe<uint16_t>(0); // CCSID
  OS.write_zeros(16);      // Character Set name
  OS.write_zeros(16);      // Language Product Identifier
  OS.writebe<uint32_t>(1); // Architecture Level
  OS.writebe<uint16_t>(0); // Module Properties Length
  OS.write_zeros(6);       // Reserved
}

void GOFFObjectWriter::writeEnd() {
  uint8_t F = GOFF::END_EPR_None;
  uint8_t AMODE = 0;
  uint32_t ESDID = 0;

  // TODO Set Flags/AMODE/ESDID for entry point.

  OS.newRecord(GOFF::RT_END);
  OS.writebe<uint8_t>(Flags(6, 2, F)); // Indicator flags
  OS.writebe<uint8_t>(AMODE);          // AMODE
  OS.write_zeros(3);                   // Reserved
  // The record count is the number of logical records. In principle, this value
  // is available as OS.logicalRecords(). However, some tools rely on this field
  // being zero.
  OS.writebe<uint32_t>(0);     // Record Count
  OS.writebe<uint32_t>(ESDID); // ESDID (of entry point)
}

uint64_t GOFFObjectWriter::writeObject(MCAssembler &Asm) {
  writeHeader();
  writeEnd();

  // Make sure all records are written.
  OS.finalizeRecord();

  LLVM_DEBUG(dbgs() << "Wrote " << OS.getNumLogicalRecords()
                    << " logical records.");

  return OS.getWrittenSize();
}

std::unique_ptr<MCObjectWriter>
llvm::createGOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                             raw_pwrite_stream &OS) {
  return std::make_unique<GOFFObjectWriter>(std::move(MOTW), OS);
}
