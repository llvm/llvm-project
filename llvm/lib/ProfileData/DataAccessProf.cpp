#include "llvm/ProfileData/DataAccessProf.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <sys/types.h>

namespace llvm {
namespace memprof {

// If `Map` has an entry keyed by `Str`, returns the entry iterator. Otherwise,
// creates an owned copy of `Str`, adds a map entry for it and returns the
// iterator.
static std::pair<StringRef, uint64_t>
saveStringToMap(DataAccessProfData::StringToIndexMap &Map,
                llvm::UniqueStringSaver &Saver, StringRef Str) {
  auto [Iter, Inserted] = Map.try_emplace(Saver.save(Str), Map.size());
  return *Iter;
}

// Returns the canonical name or error.
static Expected<StringRef> getCanonicalName(StringRef Name) {
  if (Name.empty())
    return make_error<StringError>("Empty symbol name",
                                   llvm::errc::invalid_argument);
  return InstrProfSymtab::getCanonicalName(Name);
}

std::optional<DataAccessProfRecord>
DataAccessProfData::getProfileRecord(const SymbolHandleRef SymbolID) const {
  auto Key = SymbolID;
  if (std::holds_alternative<StringRef>(SymbolID)) {
    auto NameOrErr = getCanonicalName(std::get<StringRef>(SymbolID));
    // If name canonicalization fails, suppress the error inside.
    if (!NameOrErr) {
      assert(
          std::get<StringRef>(SymbolID).empty() &&
          "Name canonicalization only fails when stringified string is empty.");
      return std::nullopt;
    }
    Key = *NameOrErr;
  }

  auto It = Records.find(Key);
  if (It != Records.end()) {
    return DataAccessProfRecord(Key, It->second.AccessCount,
                                It->second.Locations);
  }

  return std::nullopt;
}

bool DataAccessProfData::isKnownColdSymbol(const SymbolHandleRef SymID) const {
  if (std::holds_alternative<uint64_t>(SymID))
    return KnownColdHashes.contains(std::get<uint64_t>(SymID));
  return KnownColdSymbols.contains(std::get<StringRef>(SymID));
}

Error DataAccessProfData::setDataAccessProfile(SymbolHandleRef Symbol,
                                               uint64_t AccessCount) {
  uint64_t RecordID = -1;
  const bool IsStringLiteral = std::holds_alternative<uint64_t>(Symbol);
  SymbolHandleRef Key;
  if (IsStringLiteral) {
    RecordID = std::get<uint64_t>(Symbol);
    Key = RecordID;
  } else {
    auto CanonicalName = getCanonicalName(std::get<StringRef>(Symbol));
    if (!CanonicalName)
      return CanonicalName.takeError();
    std::tie(Key, RecordID) =
        saveStringToMap(StrToIndexMap, Saver, *CanonicalName);
  }

  auto [Iter, Inserted] =
      Records.try_emplace(Key, RecordID, AccessCount, IsStringLiteral);
  if (!Inserted)
    return make_error<StringError>("Duplicate symbol or string literal added. "
                                   "User of DataAccessProfData should "
                                   "aggregate count for the same symbol. ",
                                   llvm::errc::invalid_argument);

  return Error::success();
}

Error DataAccessProfData::setDataAccessProfile(
    SymbolHandleRef SymbolID, uint64_t AccessCount,
    ArrayRef<SourceLocation> Locations) {
  if (Error E = setDataAccessProfile(SymbolID, AccessCount))
    return E;

  auto &Record = Records.back().second;
  for (const auto &Location : Locations)
    Record.Locations.push_back(
        {saveStringToMap(StrToIndexMap, Saver, Location.FileName).first,
         Location.Line});

  return Error::success();
}

Error DataAccessProfData::addKnownSymbolWithoutSamples(
    SymbolHandleRef SymbolID) {
  if (std::holds_alternative<uint64_t>(SymbolID)) {
    KnownColdHashes.insert(std::get<uint64_t>(SymbolID));
    return Error::success();
  }
  auto CanonicalName = getCanonicalName(std::get<StringRef>(SymbolID));
  if (!CanonicalName)
    return CanonicalName.takeError();
  KnownColdSymbols.insert(
      saveStringToMap(StrToIndexMap, Saver, *CanonicalName).first);
  return Error::success();
}

Error DataAccessProfData::deserialize(const unsigned char *&Ptr) {
  uint64_t NumSampledSymbols =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  uint64_t NumColdKnownSymbols =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  if (Error E = deserializeSymbolsAndFilenames(Ptr, NumSampledSymbols,
                                               NumColdKnownSymbols))
    return E;

  uint64_t Num =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  for (uint64_t I = 0; I < Num; ++I)
    KnownColdHashes.insert(
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr));

  return deserializeRecords(Ptr);
}

Error DataAccessProfData::serializeSymbolsAndFilenames(ProfOStream &OS) const {
  OS.write(StrToIndexMap.size());
  OS.write(KnownColdSymbols.size());

  std::vector<std::string> Strs;
  Strs.reserve(StrToIndexMap.size() + KnownColdSymbols.size());
  for (const auto &Str : StrToIndexMap)
    Strs.push_back(Str.first.str());
  for (const auto &Str : KnownColdSymbols)
    Strs.push_back(Str.str());

  std::string CompressedStrings;
  if (!Strs.empty())
    if (Error E = collectGlobalObjectNameStrings(
            Strs, compression::zlib::isAvailable(), CompressedStrings))
      return E;
  const uint64_t CompressedStringLen = CompressedStrings.length();
  // Record the length of compressed string.
  OS.write(CompressedStringLen);
  // Write the chars in compressed strings.
  for (char C : CompressedStrings)
    OS.writeByte(static_cast<uint8_t>(C));
  // Pad up to a multiple of 8.
  // InstrProfReader could read bytes according to 'CompressedStringLen'.
  const uint64_t PaddedLength = alignTo(CompressedStringLen, 8);
  for (uint64_t K = CompressedStringLen; K < PaddedLength; K++)
    OS.writeByte(0);
  return Error::success();
}

uint64_t
DataAccessProfData::getEncodedIndex(const SymbolHandleRef SymbolID) const {
  if (std::holds_alternative<uint64_t>(SymbolID))
    return std::get<uint64_t>(SymbolID);

  auto Iter = StrToIndexMap.find(std::get<StringRef>(SymbolID));
  assert(Iter != StrToIndexMap.end() &&
         "String literals not found in StrToIndexMap");
  return Iter->second;
}

Error DataAccessProfData::serialize(ProfOStream &OS) const {
  if (Error E = serializeSymbolsAndFilenames(OS))
    return E;
  OS.write(KnownColdHashes.size());
  for (const auto &Hash : KnownColdHashes)
    OS.write(Hash);
  OS.write((uint64_t)(Records.size()));
  for (const auto &[Key, Rec] : Records) {
    OS.write(getEncodedIndex(Rec.SymbolID));
    OS.writeByte(Rec.IsStringLiteral);
    OS.write(Rec.AccessCount);
    OS.write(Rec.Locations.size());
    for (const auto &Loc : Rec.Locations) {
      OS.write(getEncodedIndex(Loc.FileName));
      OS.write32(Loc.Line);
    }
  }
  return Error::success();
}

Error DataAccessProfData::deserializeSymbolsAndFilenames(
    const unsigned char *&Ptr, const uint64_t NumSampledSymbols,
    const uint64_t NumColdKnownSymbols) {
  uint64_t Len =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

  // The first NumSampledSymbols strings are symbols with samples, and next
  // NumColdKnownSymbols strings are known cold symbols.
  uint64_t StringCnt = 0;
  std::function<Error(StringRef)> addName = [&](StringRef Name) {
    if (StringCnt < NumSampledSymbols)
      saveStringToMap(StrToIndexMap, Saver, Name);
    else
      KnownColdSymbols.insert(Saver.save(Name));
    ++StringCnt;
    return Error::success();
  };
  if (Error E =
          readAndDecodeStrings(StringRef((const char *)Ptr, Len), addName))
    return E;

  Ptr += alignTo(Len, 8);
  return Error::success();
}

Error DataAccessProfData::deserializeRecords(const unsigned char *&Ptr) {
  SmallVector<StringRef> Strings =
      llvm::to_vector(llvm::make_first_range(getStrToIndexMapRef()));

  uint64_t NumRecords =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

  for (uint64_t I = 0; I < NumRecords; ++I) {
    uint64_t ID =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

    bool IsStringLiteral =
        support::endian::readNext<uint8_t, llvm::endianness::little>(Ptr);

    uint64_t AccessCount =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

    SymbolHandleRef SymbolID;
    if (IsStringLiteral)
      SymbolID = ID;
    else
      SymbolID = Strings[ID];
    if (Error E = setDataAccessProfile(SymbolID, AccessCount))
      return E;

    auto &Record = Records.back().second;

    uint64_t NumLocations =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

    Record.Locations.reserve(NumLocations);
    for (uint64_t J = 0; J < NumLocations; ++J) {
      uint64_t FileNameIndex =
          support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
      uint32_t Line =
          support::endian::readNext<uint32_t, llvm::endianness::little>(Ptr);
      Record.Locations.push_back({Strings[FileNameIndex], Line});
    }
  }
  return Error::success();
}
} // namespace memprof
} // namespace llvm
