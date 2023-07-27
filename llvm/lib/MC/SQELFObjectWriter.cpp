//
//===----------------------------------------------------------------------===//
//
// This file implements SQELF object file writer information.
//
//===----------------------------------------------------------------------===//
#include "llvm/BinaryFormat/SQELF.h"
#include "llvm/MC/MCSQELFObjectWriter.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <sqlite3.h>

using namespace llvm;

static void writeInMemoryDatabaseToStream(llvm::raw_pwrite_stream &os,
                                          sqlite3 *db);

class SQELFObjectWriter : public MCObjectWriter {
  raw_pwrite_stream &OS;
  /// The target specific ELF writer instance.
  std::unique_ptr<MCSQELFObjectTargetWriter> TargetObjectWriter;

public:
  SQELFObjectWriter(std::unique_ptr<MCSQELFObjectTargetWriter> MOTW,
                    raw_pwrite_stream &OS)
      : OS(OS), TargetObjectWriter(std::move(MOTW)) {}

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override;

  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
};

void SQELFObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                 const MCAsmLayout &Layout) {}

void SQELFObjectWriter::recordRelocation(MCAssembler &Asm,
                                         const MCAsmLayout &Layout,
                                         const MCFragment *Fragment,
                                         const MCFixup &Fixup, MCValue Target,
                                         uint64_t &FixedValue) {}

uint64_t SQELFObjectWriter::writeObject(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) {
  BinaryFormat::SQELF sqlelf{};
  writeInMemoryDatabaseToStream(OS, sqlelf.getSqliteDatabase());
  return 0;
}

std::unique_ptr<MCObjectWriter>
llvm::createSQELFObjectWriter(std::unique_ptr<MCSQELFObjectTargetWriter> MOTW,
                              raw_pwrite_stream &OS) {
  return std::make_unique<SQELFObjectWriter>(std::move(MOTW), OS);
}

/**
 * @brief The SQELF ObjectFormat stores it's internal representation as an
 * in-memory database. We however want to pipe this to the object stream.
 * This function handles that conversion by first dumping the database
 * to a temporary file.
 */
static void writeInMemoryDatabaseToStream(llvm::raw_pwrite_stream &os,
                                          sqlite3 *db) {
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
  os << fileBuffer->get()->getBuffer();

  // Delete the temporary file.
  std::remove(tempFilename.c_str());
}
