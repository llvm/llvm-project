#ifndef LLVM_BINARYFORMAT_SQELF_H
#define LLVM_BINARYFORMAT_SQELF_H

#include "llvm/Support/raw_ostream.h"
#include <sqlite3.h>

namespace llvm {
namespace BinaryFormat {
class SQELF {

public:
  typedef struct Metadata {
    std::string Type;
    std::string Arch;
    unsigned int Version;
  } Metadata;

public:
  SQELF();
  virtual ~SQELF();

  SQELF &setMetadata(const Metadata &M);
  Metadata getMetadata() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SQELF &BF);

private:
  sqlite3 *DB;
};
} // namespace BinaryFormat
} // namespace llvm

#endif // LLVM_BINARYFORMAT_SQELF_H