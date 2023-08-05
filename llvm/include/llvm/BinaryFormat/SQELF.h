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

  /// Sets the Metadata row in the appropriate table
  ///
  /// Will create a Database insert statement
  ///
  /// \param M the metadata to insert
  /// \returns itself for fluent use
  SQELF &setMetadata(const Metadata &M);

  /// Retrieves the Metadata row in the appropriate table
  ///
  /// Will create a Database select statement
  Metadata getMetadata() const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SQELF &BF);

private:
  sqlite3 *DB;
};
} // namespace BinaryFormat
} // namespace llvm

#endif // LLVM_BINARYFORMAT_SQELF_H