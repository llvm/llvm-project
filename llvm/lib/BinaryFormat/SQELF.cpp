#include "llvm/BinaryFormat/SQELF.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace BinaryFormat;

SQELF::SQELF() {
  int rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) {
    report_fatal_error("Could not create an in-memory sqlite database");
  }
}

SQELF::~SQELF() {
  int rc = sqlite3_close(db);
  if (rc != SQLITE_OK) {
    report_fatal_error(
        "Could not close in-memory sqlite database; likely database is locked");
  }
}