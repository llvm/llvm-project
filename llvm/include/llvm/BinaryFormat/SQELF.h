#ifndef LLVM_BINARYFORMAT_SQELF_H
#define LLVM_BINARYFORMAT_SQELF_H

#include <sqlite3.h>

namespace llvm {
namespace BinaryFormat {
class SQELF {

public:
  SQELF();
  virtual ~SQELF();

  // TODO(fzakaria): maybe mark SQELFObjectWriter as a friend
  // which is only able to call this.
  sqlite3 *getSqliteDatabase() const { return db; }

private:
  sqlite3 *db;
};
} // namespace BinaryFormat
} // namespace llvm

#endif // LLVM_BINARYFORMAT_SQELF_H