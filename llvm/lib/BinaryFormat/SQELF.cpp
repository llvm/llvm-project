#include "llvm/BinaryFormat/SQELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sqlite3.h>

using namespace llvm;
using namespace BinaryFormat;

static void writeMetadataToDatabase(sqlite3 *DB, const SQELF::Metadata &M);
static void writeInMemoryDatabaseToStream(llvm::raw_ostream &OS, sqlite3 *DB);
static void initializeTables(sqlite3 *DB);

SQELF::SQELF(const Metadata &M) : M(M) {
  int ResultCode = sqlite3_open(":memory:", &DB);
  if (ResultCode != SQLITE_OK) {
    report_fatal_error("Could not create an in-memory sqlite database");
  }
  initializeTables(DB);
}

SQELF::~SQELF() {
  int ResultCode = sqlite3_close(DB);
  if (ResultCode != SQLITE_OK) {
    report_fatal_error(
        "Could not close in-memory sqlite database; likely database is locked");
  }
}

namespace llvm {
namespace BinaryFormat {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SQELF &BF) {
  writeMetadataToDatabase(BF.DB, BF.M);
  writeInMemoryDatabaseToStream(OS, BF.DB);
  return OS;
}
} // namespace BinaryFormat
} // namespace llvm

// TODO(fzakaria): Is there a better preferred way to create large
// text files?
const char *CreateMetadataTableSQL =
#include "./sql/create_metadata.sql"
    ;

void initializeTables(sqlite3 *DB) {

  char *ErrorMessage = nullptr;
  int RC =
      sqlite3_exec(DB, CreateMetadataTableSQL, nullptr, nullptr, &ErrorMessage);
  if (RC != SQLITE_OK) {
    report_fatal_error(formatv("failed to create sqlite3 table: {0}",
                               std::string(ErrorMessage)));
    sqlite3_free(ErrorMessage);
  }
}

void writeMetadataToDatabase(sqlite3 *DB, const SQELF::Metadata &M) {}

/**
 * @brief The SQELF ObjectFormat stores it's internal representation as an
 * in-memory database. We however want to pipe this to the object stream.
 * This function handles that conversion by first dumping the database
 * to a temporary file.
 */
static void writeInMemoryDatabaseToStream(llvm::raw_ostream &OS, sqlite3 *DB) {
  llvm::SmallString<64> TempFilename;
  if (llvm::sys::fs::createTemporaryFile("temp", "db", TempFilename)) {
    report_fatal_error("Could not create temporary file");
    return;
  }

  // Save the in-memory database to a temporary file.
  sqlite3 *TempDB;
  if (sqlite3_open(TempFilename.c_str(), &TempDB) != SQLITE_OK) {
    report_fatal_error("failed to open sqlite3 database: " + TempFilename);
    return;
  }

  sqlite3_backup *Backup = sqlite3_backup_init(TempDB, "main", DB, "main");
  if (Backup) {
    sqlite3_backup_step(Backup, -1);
    sqlite3_backup_finish(Backup);
  }

  if (sqlite3_close(TempDB) != SQLITE_OK) {
    report_fatal_error("failed to close sqlite3 database: " + TempFilename);
    return;
  }

  // Open the temporary file
  auto FileBuffer = llvm::MemoryBuffer::getFile(TempFilename.c_str());

  // Write the temporary file contents to the output stream.
  OS << FileBuffer->get()->getBuffer();

  // Delete the temporary file.
  std::remove(TempFilename.c_str());
}