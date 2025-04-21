#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Endian.h"

namespace llvm {

uint64_t
DataAccessProfData::addStringRef(StringRef Str,
                                 llvm::MapVector<StringRef, uint64_t> &Map) {
  auto [Iter, Inserted] = Map.insert({Str, Map.size()});
  return Iter->second;
}

uint64_t DataAccessProfData::addSymbolName(StringRef SymbolName) {
  return addStringRef(SymbolName, SymbolNames);
}
uint64_t DataAccessProfData::addFileName(StringRef FileName) {
  return addStringRef(FileName, FileNames);
}

DataAccessProfRecord &DataAccessProfData::addRecord(uint64_t SymbolNameIndex,
                                                    uint64_t StringContentHash,
                                                    uint64_t AccessCount) {
  Records.push_back({SymbolNameIndex, StringContentHash, AccessCount});
  return Records.back();
}

SmallVector<StringRef> DataAccessProfData::getSymbolNames() const {
  return llvm::to_vector(llvm::map_range(
      SymbolNames, [](const auto &Pair) { return Pair.first; }));
}

SmallVector<StringRef> DataAccessProfData::getFileNames() const {
  return llvm::to_vector(
      llvm::map_range(FileNames, [](const auto &Pair) { return Pair.first; }));
}

Error DataAccessProfData::deserialize(const unsigned char *&Ptr) {
  if (Error E = deserializeSymbolNames(Ptr))
    return E;
  if (Error E = deserializeFileNames(Ptr))
    return E;
  if (Error E = deserializeRecords(Ptr))
    return E;
  return Error::success();
}

Error DataAccessProfData::deserializeSymbolNames(const unsigned char *&Ptr) {
  uint64_t SymbolNameLen =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

  StringRef MaybeCompressedSymbolNames((const char *)Ptr, SymbolNameLen);

  std::function<Error(StringRef)> addSymbolName = [this](StringRef SymbolName) {
    this->addSymbolName(SymbolName);
    return Error::success();
  };
  if (Error E = readAndDecodeStrings(MaybeCompressedSymbolNames, addSymbolName))
    return E;

  Ptr += alignTo(SymbolNameLen, 8);
  return Error::success();
}

Error DataAccessProfData::deserializeFileNames(const unsigned char *&Ptr) {
  uint64_t FileNameLen =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

  StringRef MaybeCompressedFileNames((const char *)Ptr, FileNameLen);

  std::function<Error(StringRef)> addFileName = [this](StringRef FileName) {
    this->addFileName(FileName);
    return Error::success();
  };
  if (Error E = readAndDecodeStrings(MaybeCompressedFileNames, addFileName))
    return E;

  Ptr += alignTo(FileNameLen, 8);
  return Error::success();
}

Error DataAccessProfData::deserializeRecords(const unsigned char *&Ptr) {
  uint64_t NumRecords =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  Records.reserve(NumRecords);
  for (uint64_t I = 0; I < NumRecords; ++I) {
    uint64_t SymbolNameIndex =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    uint64_t StringContentHash =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    uint64_t AccessCount =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

    DataAccessProfRecord &Record =
        addRecord(SymbolNameIndex, StringContentHash, AccessCount);

    uint64_t NumLocations =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    Record.Locations.reserve(NumLocations);
    for (uint64_t J = 0; J < NumLocations; ++J) {
      uint64_t FileNameIndex =
          support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
      uint32_t Line =
          support::endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
      Record.Locations.push_back({FileNameIndex, Line});
    }
  }
  return Error::success();
}
} // namespace llvm
