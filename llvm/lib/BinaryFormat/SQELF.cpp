#include "llvm/BinaryFormat/SQELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sqlite3.h>

using namespace llvm;
using namespace BinaryFormat;

static void writeInMemoryDatabaseToStream(llvm::raw_ostream &os, sqlite3 *db);

static void initializeTables(sqlite3 *db);

SQELF::SQELF() {
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) {
    report_fatal_error("Could not create an in-memory sqlite database");
  }
  initializeTables(db);
}

SQELF::~SQELF() {
  int rc = sqlite3_close(db);
  if (rc != SQLITE_OK) {
    report_fatal_error(
        "Could not close in-memory sqlite database; likely database is locked");
  }
}

namespace llvm {
namespace BinaryFormat {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SQELF &BF) {
  writeInMemoryDatabaseToStream(OS, BF.db);
  return OS;
}
} // namespace BinaryFormat
} // namespace llvm

// TODO(fzakaria): Is there a better preferred way to create large
// text files?
const char *CREATE_METADATA_TABLE_SQL =
#include "./sql/create_metadata.sql"
    ;

void initializeTables(sqlite3 *db) {

  char *errMsg = nullptr;
  int rc =
      sqlite3_exec(db, CREATE_METADATA_TABLE_SQL, nullptr, nullptr, &errMsg);
  if (rc != SQLITE_OK) {
    report_fatal_error(
        formatv("failed to create sqlite3 table: {0}", std::string(errMsg)));
    sqlite3_free(errMsg);
  }
}

/**
 * @brief The SQELF ObjectFormat stores it's internal representation as an
 * in-memory database. We however want to pipe this to the object stream.
 * This function handles that conversion by first dumping the database
 * to a temporary file.
 */
static void writeInMemoryDatabaseToStream(llvm::raw_ostream &OS, sqlite3 *db) {
  llvm::SmallString<64> tempFilename;
  if (llvm::sys::fs::createTemporaryFile("temp", "db", tempFilename)) {
    report_fatal_error("Could not create temporary file");
    return;
  }

  // Save the in-memory database to a temporary file.
  sqlite3 *tempDb;
  if (sqlite3_open(tempFilename.c_str(), &tempDb) != SQLITE_OK) {
    report_fatal_error("failed to open sqlite3 database: " + tempFilename);
    return;
  }

  sqlite3_backup *backup = sqlite3_backup_init(tempDb, "main", db, "main");
  if (backup) {
    sqlite3_backup_step(backup, -1);
    sqlite3_backup_finish(backup);
  }

  if (sqlite3_close(tempDb) != SQLITE_OK) {
    report_fatal_error("failed to close sqlite3 database: " + tempFilename);
    return;
  }

  // Open the temporary file
  auto fileBuffer = llvm::MemoryBuffer::getFile(tempFilename.c_str());

  // Write the temporary file contents to the output stream.
  OS << fileBuffer->get()->getBuffer();

  // Delete the temporary file.
  std::remove(tempFilename.c_str());
}