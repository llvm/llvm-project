#ifndef LLVM_PROFILEDATA_DATAACCESSPROF_H_
#define LLVM_PROFILEDATA_DATAACCESSPROF_H_

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace llvm {

struct DataLocation {
  // Filenames are duplicated.
  uint64_t FileNameIndex;
  uint32_t Line;
};

struct DataAccessProfRecord {
  uint64_t SymbolNameIndex;
  uint64_t StringContentHash;
  uint64_t AccessCount;
  llvm::SmallVector<DataLocation> Locations;
};

class DataAccessProfData {
  uint64_t addStringRef(StringRef Str,
                        llvm::MapVector<StringRef, uint64_t> &Map);

  Error deserializeSymbolNames(const unsigned char *&Ptr);

  Error deserializeFileNames(const unsigned char *&Ptr);

  Error deserializeRecords(const unsigned char *&Ptr);

public:
  // Canonicalized symbol names.
  llvm::MapVector<StringRef, uint64_t> SymbolNames;
  llvm::MapVector<StringRef, uint64_t> FileNames;
  llvm::SmallVector<DataAccessProfRecord> Records;

  uint64_t addSymbolName(StringRef SymbolName);
  uint64_t addFileName(StringRef FileName);

  Error deserialize(const unsigned char *&Ptr);

  DataAccessProfRecord &addRecord(uint64_t SymbolNameIndex,
                                  uint64_t StringContentHash,
                                  uint64_t AccessCount);

  SmallVector<StringRef> getSymbolNames() const;

  SmallVector<StringRef> getFileNames() const;
};
} // namespace llvm

#endif // LLVM_PROFILEDATA_DATAACCESSPROF_H_
