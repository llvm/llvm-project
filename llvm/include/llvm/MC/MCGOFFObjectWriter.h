//===- MCGOFFObjectWriter.h - GOFF Object Writer ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCGOFFOBJECTWRITER_H
#define LLVM_MC_MCGOFFOBJECTWRITER_H

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/Endian.h"

namespace llvm {
class MCAssembler;
class MCObjectWriter;
class raw_pwrite_stream;

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

  raw_pwrite_stream &getOS();
  size_t getWrittenSize() const;
  uint32_t getNumLogicalRecords();

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

class GOFFWriter {
  GOFFOstream OS;
  [[maybe_unused]] MCAssembler &Asm;

  void writeHeader();
  void writeEnd();

public:
  GOFFWriter(raw_pwrite_stream &OS, MCAssembler &Asm);
  uint64_t writeObject();
};

class MCGOFFObjectTargetWriter : public MCObjectTargetWriter {
protected:
  MCGOFFObjectTargetWriter() = default;

public:
  virtual ~MCGOFFObjectTargetWriter() = default;

  Triple::ObjectFormatType getFormat() const override { return Triple::GOFF; }

  static bool classof(const MCObjectTargetWriter *W) {
    return W->getFormat() == Triple::GOFF;
  }
};

/// \brief Construct a new GOFF writer instance.
///
/// \param MOTW - The target-specific GOFF writer subclass.
/// \param OS - The stream to write to.
/// \returns The constructed object writer.
std::unique_ptr<MCObjectWriter>
createGOFFObjectWriter(std::unique_ptr<MCGOFFObjectTargetWriter> MOTW,
                       raw_pwrite_stream &OS);
} // namespace llvm

#endif
