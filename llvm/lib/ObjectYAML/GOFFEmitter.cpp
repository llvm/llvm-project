//===- yaml2goff - Convert YAML to a GOFF object file ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The GOFF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/ObjectYAML/GOFFYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

// Common flag values on records.
enum {
  // Flag: This record is continued.
  Rec_Continued = 1 << 0,

  // Flag: This record is a continuation.
  Rec_Continuation = 1 << 1,
};

template <typename ValueType> struct BinaryBeImpl {
  ValueType Value;
  BinaryBeImpl(ValueType V) : Value(V) {}
};

template <typename ValueType>
raw_ostream &operator<<(raw_ostream &OS, const BinaryBeImpl<ValueType> &BBE) {
  char Buffer[sizeof(BBE.Value)];
  support::endian::write<ValueType, llvm::endianness::big, support::unaligned>(
      Buffer, BBE.Value);
  OS.write(Buffer, sizeof(BBE.Value));
  return OS;
}

template <typename ValueType> BinaryBeImpl<ValueType> binaryBe(ValueType V) {
  return BinaryBeImpl<ValueType>(V);
}

struct ZerosImpl {
  size_t NumBytes;
};

raw_ostream &operator<<(raw_ostream &OS, const ZerosImpl &Z) {
  OS.write_zeros(Z.NumBytes);
  return OS;
}

ZerosImpl zeros(const size_t NumBytes) { return ZerosImpl{NumBytes}; }

// The GOFFOstream is responsible to write the data into the fixed physical
// records of the format. A user of this class announces the start of a new
// logical record, and writes the full logical block. The physical records are
// created while the content is written to the underlying stream. Possible fill
// bytes at the end of a physical record are written automatically.
// The implementation aims at simplicity, not speed.
class GOFFOStream {
public:
  explicit GOFFOStream(raw_ostream &OS)
      : OS(OS), CurrentType(GOFF::RecordType(-1)) {}

  GOFFOStream &operator<<(StringRef Str) {
    write(Str);
    return *this;
  }

  void newRecord(GOFF::RecordType Type) { CurrentType = Type; }

private:
  // The underlying raw_ostream.
  raw_ostream &OS;

  // The type of the current (logical) record.
  GOFF::RecordType CurrentType;

  // Write the record prefix of a physical record, using the current record
  // type.
  void writeRecordPrefix(uint8_t Flags);

  // Write a logical record.
  void write(StringRef Str);
};

void GOFFOStream::writeRecordPrefix(uint8_t Flags) {
  uint8_t TypeAndFlags = Flags | (CurrentType << 4);
  OS << binaryBe(static_cast<unsigned char>(GOFF::PTVPrefix))
     << binaryBe(static_cast<unsigned char>(TypeAndFlags))
     << binaryBe(static_cast<unsigned char>(0));
}

void GOFFOStream::write(StringRef Str) {
  // The flags are determined by the flags of the prvious record, and by the
  // remaining size of data.
  uint8_t Flags = 0;
  size_t Ptr = 0;
  size_t Size = Str.size();
  while (Size >= GOFF::RecordContentLength) {
    if (Flags) {
      Flags |= Rec_Continuation;
      if (Size == GOFF::RecordContentLength)
        Flags &= ~Rec_Continued;
    } else
      Flags |= (Size == GOFF::RecordContentLength) ? 0 : Rec_Continued;
    writeRecordPrefix(Flags);
    OS.write(&Str.data()[Ptr], GOFF::RecordContentLength);
    Size -= GOFF::RecordContentLength;
    Ptr += GOFF::RecordContentLength;
  }
  if (Size) {
    Flags &= ~Rec_Continued;
    writeRecordPrefix(Flags);
    OS.write(&Str.data()[Ptr], Size);
    OS.write_zeros(GOFF::RecordContentLength - Size);
  }
}

// A LogicalRecord buffers the data of a record.
class LogicalRecord : public raw_svector_ostream {
  GOFFOStream &OS;
  SmallVector<char, 0> Buffer;

  void anchor() override {};

public:
  LogicalRecord(GOFFOStream &OS) : raw_svector_ostream(Buffer), OS(OS) {}
  ~LogicalRecord() override { OS << str(); }

  LogicalRecord &operator<<(yaml::BinaryRef B) {
    B.writeAsBinary(*this);
    return *this;
  }
};

class GOFFState {
  void writeHeader(GOFFYAML::ModuleHeader &ModHdr);
  void writeEnd(GOFFYAML::EndOfModule &EndMod);

  void reportError(const Twine &Msg) {
    ErrHandler(Msg);
    HasError = true;
  }

  GOFFState(raw_ostream &OS, GOFFYAML::Object &Doc,
            yaml::ErrorHandler ErrHandler)
      : GW(OS), Doc(Doc), ErrHandler(ErrHandler), HasError(false) {}

  bool writeObject();

public:
  static bool writeGOFF(raw_ostream &OS, GOFFYAML::Object &Doc,
                        yaml::ErrorHandler ErrHandler);

private:
  GOFFOStream GW;
  GOFFYAML::Object &Doc;
  yaml::ErrorHandler ErrHandler;
  bool HasError;
};

void GOFFState::writeHeader(GOFFYAML::ModuleHeader &ModHdr) {
  GW.newRecord(GOFF::RT_HDR);
  LogicalRecord LR(GW);
  LR << zeros(45)                          // Reserved.
     << binaryBe(ModHdr.ArchitectureLevel) // The architecture level.
     << binaryBe(ModHdr.PropertiesLength)  // Length of module properties.
     << zeros(6);                          // Reserved.
  if (ModHdr.Properties)
    LR << *ModHdr.Properties; // Module properties.
}

void GOFFState::writeEnd(GOFFYAML::EndOfModule &EndMod) {
  SmallString<16> EntryName;
  if (std::error_code EC =
          ConverterEBCDIC::convertToEBCDIC(EndMod.EntryName, EntryName))
    reportError("Conversion error on " + EndMod.EntryName);

  GW.newRecord(GOFF::RT_END);
  LogicalRecord LR(GW);
  LR << binaryBe(uint8_t(EndMod.Flags)) // The flags.
     << binaryBe(uint8_t(EndMod.AMODE)) // The addressing mode.
     << zeros(3)                        // Reserved.
     << binaryBe(EndMod.RecordCount)    // The record count.
     << binaryBe(EndMod.ESDID)          // ESDID of the entry point.
     << zeros(4)                        // Reserved.
     << binaryBe(EndMod.Offset)         // Offset of entry point.
     << binaryBe(EndMod.NameLength)     // Length of external name.
     << EntryName;                      // Name of the entry point.
}

bool GOFFState::writeObject() {
  for (auto &RecPtr : Doc.Records) {
    auto *Rec = RecPtr.get();
    switch (Rec->getKind()) {
    case GOFFYAML::RecordBase::RBK_ModuleHeader:
      writeHeader(*static_cast<GOFFYAML::ModuleHeader *>(Rec));
      break;
    case GOFFYAML::RecordBase::RBK_EndOfModule:
      writeEnd(*static_cast<GOFFYAML::EndOfModule *>(Rec));
      break;
    case GOFFYAML::RecordBase::RBK_RelocationDirectory:
    case GOFFYAML::RecordBase::RBK_Symbol:
    case GOFFYAML::RecordBase::RBK_Text:
    case GOFFYAML::RecordBase::RBK_DeferredLength:
      llvm_unreachable(("Not yet implemented"));
    }
    if (HasError)
      return false;
  }
  return true;
}

bool GOFFState::writeGOFF(raw_ostream &OS, GOFFYAML::Object &Doc,
                          yaml::ErrorHandler ErrHandler) {
  GOFFState State(OS, Doc, ErrHandler);
  return State.writeObject();
}
} // namespace

namespace llvm {
namespace yaml {

bool yaml2goff(llvm::GOFFYAML::Object &Doc, raw_ostream &Out,
               ErrorHandler ErrHandler) {
  return GOFFState::writeGOFF(Out, Doc, ErrHandler);
}

} // namespace yaml
} // namespace llvm
