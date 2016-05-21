//===- SymbolStream.cpp - PDB Symbol Stream Access ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/SymbolStream.h"

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/RawConstants.h"
#include "llvm/DebugInfo/PDB/Raw/RawError.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace llvm::pdb;

// Symbol stream is an array of symbol records. Each record starts with
// length and type fields followed by type-specfic fields.
namespace {
struct SymbolHeader {
  ulittle16_t Len; // Record length
  ulittle16_t Type;
};

// For S_PUB32 symbol type.
struct DataSym32 {
  ulittle32_t TypIndex; // Type index, or Metadata token if a managed symbol
  ulittle32_t off;
  ulittle16_t seg;
  char Name[1];
};

// For S_PROCREF symbol type.
struct RefSym {
  ulittle32_t SumName;   // SUC of the name (?)
  ulittle32_t SymOffset; // Offset of actual symbol in $$Symbols
  ulittle16_t Mod;       // Module containing the actual symbol
  char Name[1];
};
}

SymbolStream::SymbolStream(PDBFile &File, uint32_t StreamNum)
    : Stream(StreamNum, File) {}

SymbolStream::~SymbolStream() {}

Error SymbolStream::reload() { return Error::success(); }

Expected<std::string> SymbolStream::getSymbolName(uint32_t Off) const {
  StreamReader Reader(Stream);
  Reader.setOffset(Off);

  // Read length field.
  SymbolHeader Hdr;
  if (Reader.readObject(&Hdr))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupted symbol stream.");

  // Read the entire record.
  std::vector<uint8_t> Buf(Hdr.Len - sizeof(Hdr.Type));
  if (Reader.readBytes(Buf))
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Corrupted symbol stream.");

  switch (Hdr.Type) {
  case codeview::S_PUB32:
    return reinterpret_cast<DataSym32 *>(Buf.data())->Name;
  case codeview::S_PROCREF:
    return reinterpret_cast<RefSym *>(Buf.data())->Name;
  default:
    return make_error<RawError>(raw_error_code::corrupt_file,
                                "Unknown symbol type");
  }
  return Error::success();
}
