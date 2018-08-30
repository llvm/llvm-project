//===- FDRRecords.h - XRay Flight Data Recorder Mode Records --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Define types and operations on these types that represent the different kinds
// of records we encounter in XRay flight data recorder mode traces.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_XRAY_FDRRECORDS_H_
#define LLVM_LIB_XRAY_FDRRECORDS_H_

#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/XRay/XRayRecord.h"
#include <cstdint>

namespace llvm {
namespace xray {

class RecordVisitor;
class RecordInitializer;

class Record {
protected:
  enum class Type {
    Unknown,
    Function,
    Metadata,
  };

public:
  Record(const Record &) = delete;
  Record(Record &&) = delete;
  Record &operator=(const Record &) = delete;
  Record &operator=(Record &&) = delete;
  Record() = default;

  virtual Type type() const = 0;

  // Each Record should be able to apply an abstract visitor, and choose the
  // appropriate function in the visitor to invoke, given its own type.
  virtual Error apply(RecordVisitor &V) = 0;

  virtual ~Record() = default;
};

class MetadataRecord : public Record {
protected:
  static constexpr int kMetadataBodySize = 15;
  friend class RecordInitializer;

public:
  enum class MetadataType : unsigned {
    Unknown,
    BufferExtents,
    WallClockTime,
    NewCPUId,
    TSCWrap,
    CustomEvent,
    CallArg,
    PIDEntry,
    NewBuffer,
    EndOfBuffer,
  };

  Type type() const override { return Type::Metadata; }

  // All metadata records must know to provide the type of their open
  // metadata record.
  virtual MetadataType metadataType() const = 0;

  virtual ~MetadataRecord() = default;
};

// What follows are specific Metadata record types which encapsulate the
// information associated with specific metadata record types in an FDR mode
// log.
class BufferExtents : public MetadataRecord {
  uint64_t Size = 0;
  friend class RecordInitializer;

public:
  BufferExtents() = default;
  explicit BufferExtents(uint64_t S) : MetadataRecord(), Size(S) {}

  MetadataType metadataType() const override {
    return MetadataType::BufferExtents;
  }

  uint64_t size() const { return Size; }

  Error apply(RecordVisitor &V) override;
};

class WallclockRecord : public MetadataRecord {
  uint64_t Seconds = 0;
  uint32_t Nanos = 0;
  friend class RecordInitializer;

public:
  WallclockRecord() = default;
  explicit WallclockRecord(uint64_t S, uint32_t N)
      : MetadataRecord(), Seconds(S), Nanos(N) {}

  MetadataType metadataType() const override {
    return MetadataType::WallClockTime;
  }

  uint64_t seconds() const { return Seconds; }
  uint32_t nanos() const { return Nanos; }

  Error apply(RecordVisitor &V) override;
};

class NewCPUIDRecord : public MetadataRecord {
  uint16_t CPUId = 0;
  friend class RecordInitializer;

public:
  NewCPUIDRecord() = default;
  explicit NewCPUIDRecord(uint16_t C) : MetadataRecord(), CPUId(C) {}

  MetadataType metadataType() const override { return MetadataType::NewCPUId; }

  uint16_t cpuid() const { return CPUId; }

  Error apply(RecordVisitor &V) override;
};

class TSCWrapRecord : public MetadataRecord {
  uint64_t BaseTSC = 0;
  friend class RecordInitializer;

public:
  TSCWrapRecord() = default;
  explicit TSCWrapRecord(uint64_t B) : MetadataRecord(), BaseTSC(B) {}

  MetadataType metadataType() const override { return MetadataType::TSCWrap; }

  uint64_t tsc() const { return BaseTSC; }

  Error apply(RecordVisitor &V) override;
};

class CustomEventRecord : public MetadataRecord {
  int32_t Size = 0;
  uint64_t TSC = 0;
  std::string Data{};
  friend class RecordInitializer;

public:
  CustomEventRecord() = default;
  explicit CustomEventRecord(uint64_t S, uint64_t T, std::string D)
      : MetadataRecord(), Size(S), TSC(T), Data(std::move(D)) {}

  MetadataType metadataType() const override {
    return MetadataType::CustomEvent;
  }

  int32_t size() const { return Size; }
  uint64_t tsc() const { return TSC; }
  StringRef data() const { return Data; }

  Error apply(RecordVisitor &V) override;
};

class CallArgRecord : public MetadataRecord {
  uint64_t Arg;
  friend class RecordInitializer;

public:
  CallArgRecord() = default;
  explicit CallArgRecord(uint64_t A) : MetadataRecord(), Arg(A) {}

  MetadataType metadataType() const override { return MetadataType::CallArg; }

  uint64_t arg() const { return Arg; }

  Error apply(RecordVisitor &V) override;
};

class PIDRecord : public MetadataRecord {
  uint64_t PID = 0;
  friend class RecordInitializer;

public:
  PIDRecord() = default;
  explicit PIDRecord(uint64_t P) : MetadataRecord(), PID(P) {}

  MetadataType metadataType() const override { return MetadataType::PIDEntry; }

  uint64_t pid() const { return PID; }

  Error apply(RecordVisitor &V) override;
};

class NewBufferRecord : public MetadataRecord {
  int32_t TID = 0;
  friend class RecordInitializer;

public:
  NewBufferRecord() = default;
  explicit NewBufferRecord(int32_t T) : MetadataRecord(), TID(T) {}

  MetadataType metadataType() const override { return MetadataType::NewBuffer; }

  int32_t tid() const { return TID; }

  Error apply(RecordVisitor &V) override;
};

class EndBufferRecord : public MetadataRecord {
public:
  EndBufferRecord() = default;

  MetadataType metadataType() const override {
    return MetadataType::EndOfBuffer;
  }

  Error apply(RecordVisitor &V) override;
};

class FunctionRecord : public Record {
  RecordTypes Kind;
  int32_t FuncId;
  uint32_t Delta;
  friend class RecordInitializer;

  static constexpr unsigned kFunctionRecordSize = 8;

public:
  FunctionRecord() = default;
  explicit FunctionRecord(RecordTypes K, int32_t F, uint32_t D)
      : Record(), Kind(K), FuncId(F), Delta(D) {}

  Type type() const override { return Type::Function; }

  // A function record is a concrete record type which has a number of common
  // properties.
  RecordTypes recordType() const { return Kind; }
  int32_t functionId() const { return FuncId; }
  uint64_t delta() const { return Delta; }

  Error apply(RecordVisitor &V) override;
};

class RecordVisitor {
public:
  virtual ~RecordVisitor() = default;

  // Support all specific kinds of records:
  virtual Error visit(BufferExtents &) = 0;
  virtual Error visit(WallclockRecord &) = 0;
  virtual Error visit(NewCPUIDRecord &) = 0;
  virtual Error visit(TSCWrapRecord &) = 0;
  virtual Error visit(CustomEventRecord &) = 0;
  virtual Error visit(CallArgRecord &) = 0;
  virtual Error visit(PIDRecord &) = 0;
  virtual Error visit(NewBufferRecord &) = 0;
  virtual Error visit(EndBufferRecord &) = 0;
  virtual Error visit(FunctionRecord &) = 0;
};

class RecordInitializer : public RecordVisitor {
  DataExtractor &E;
  uint32_t &OffsetPtr;

public:
  explicit RecordInitializer(DataExtractor &DE, uint32_t &OP)
      : RecordVisitor(), E(DE), OffsetPtr(OP) {}

  Error visit(BufferExtents &) override;
  Error visit(WallclockRecord &) override;
  Error visit(NewCPUIDRecord &) override;
  Error visit(TSCWrapRecord &) override;
  Error visit(CustomEventRecord &) override;
  Error visit(CallArgRecord &) override;
  Error visit(PIDRecord &) override;
  Error visit(NewBufferRecord &) override;
  Error visit(EndBufferRecord &) override;
  Error visit(FunctionRecord &) override;
};

} // namespace xray
} // namespace llvm

#endif // LLVM_LIB_XRAY_FDRRECORDS_H_
