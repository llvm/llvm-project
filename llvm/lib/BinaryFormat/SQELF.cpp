#include "llvm/BinaryFormat/SQELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sqlite3.h>

using namespace llvm;
using namespace BinaryFormat;

static void writeInMemoryDatabaseToStream(llvm::raw_ostream &OS, sqlite3 *DB);
static void initializeTables(sqlite3 *DB);

SQELF::SQELF() {
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

SQELF &SQELF::setMetadata(const Metadata &M) {
  const char *SQL =
      "INSERT INTO Metadata (e_type, e_machine, e_version) VALUES (?, ?, ?)";
  sqlite3_stmt *STMT;

  // Prepare the SQL statement
  int ResultCode = sqlite3_prepare_v2(DB, SQL, -1, &STMT, nullptr);
  if (ResultCode != SQLITE_OK) {
    report_fatal_error(formatv("Failed to prepare metadata statement: {0}",
                               sqlite3_errmsg(DB)));
  }

  // Bind the Metadata fields to the SQL statement
  sqlite3_bind_text(STMT, 1, M.Type.c_str(), -1, SQLITE_STATIC);
  sqlite3_bind_text(STMT, 2, M.Arch.c_str(), -1, SQLITE_STATIC);
  sqlite3_bind_int(STMT, 3, M.Version);

  // Execute the SQL statement
  if (sqlite3_step(STMT) != SQLITE_DONE) {
    report_fatal_error(formatv("Failed to insert metadata statement: {0}",
                               sqlite3_errmsg(DB)));
  }

  // Finalize the statement to release resources
  sqlite3_finalize(STMT);

  return *this;
}

SQELF::Metadata SQELF::getMetadata() const {
  const char *SQL = "SELECT e_type, e_machine, e_version FROM Metadata LIMIT 1";
  sqlite3_stmt *STMT;
  Metadata M;

  // Prepare the SQL statement
  int ResultCode = sqlite3_prepare_v2(DB, SQL, -1, &STMT, nullptr);
  if (ResultCode != SQLITE_OK) {
    report_fatal_error(formatv("Failed to prepare metadata statement: {0}",
                               sqlite3_errmsg(DB)));
  }

  // Execute the SQL statement and populate the Metadata structure
  if (sqlite3_step(STMT) != SQLITE_ROW) {
    report_fatal_error(
        formatv("Failed to retrieve metadata: {0}", sqlite3_errmsg(DB)));
  }

  M.Type = reinterpret_cast<const char *>(sqlite3_column_text(STMT, 0));
  M.Arch = reinterpret_cast<const char *>(sqlite3_column_text(STMT, 1));
  M.Version = sqlite3_column_int(STMT, 2);

  // Finalize the statement to release resources
  sqlite3_finalize(STMT);

  return M;
}

namespace llvm {
namespace BinaryFormat {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SQELF &BF) {
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