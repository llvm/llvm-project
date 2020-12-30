//===- llvm-elfabi.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "ErrorCollector.h"
#include "llvm/InterfaceStub/ELFObjHandler.h"
#include "llvm/InterfaceStub/TBEHandler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace llvm {
namespace elfabi {

enum class FileFormat { TBE, ELF };

} // end namespace elfabi
} // end namespace llvm

using namespace llvm;
using namespace llvm::elfabi;

// Command line flags:
cl::opt<FileFormat> InputFileFormat(
    cl::desc("Force input file format:"),
    cl::values(clEnumValN(FileFormat::TBE, "tbe",
                          "Read `input` as text-based ELF stub"),
               clEnumValN(FileFormat::ELF, "elf",
                          "Read `input` as ELF binary")));
cl::opt<std::string> InputFilePath(cl::Positional, cl::desc("input"),
                                   cl::Required);
cl::opt<std::string>
    EmitTBE("emit-tbe",
            cl::desc("Emit a text-based ELF stub (.tbe) from the input file"),
            cl::value_desc("path"));
cl::opt<std::string>
    SOName("soname",
           cl::desc("Manually set the DT_SONAME entry of any emitted files"),
           cl::value_desc("name"));
cl::opt<ELFTarget> BinaryOutputTarget(
    "output-target", cl::desc("Create a binary stub for the specified target"),
    cl::values(clEnumValN(ELFTarget::ELF32LE, "elf32-little",
                          "32-bit little-endian ELF stub"),
               clEnumValN(ELFTarget::ELF32BE, "elf32-big",
                          "32-bit big-endian ELF stub"),
               clEnumValN(ELFTarget::ELF64LE, "elf64-little",
                          "64-bit little-endian ELF stub"),
               clEnumValN(ELFTarget::ELF64BE, "elf64-big",
                          "64-bit big-endian ELF stub")));
cl::opt<std::string> BinaryOutputFilePath(cl::Positional, cl::desc("output"));
cl::opt<bool> WriteIfChanged(
    "write-if-changed",
    cl::desc("Write the output file only if it is new or has changed."));

/// writeTBE() writes a Text-Based ELF stub to a file using the latest version
/// of the YAML parser.
static Error writeTBE(StringRef FilePath, ELFStub &Stub) {
  // Write TBE to memory first.
  std::string TBEStr;
  raw_string_ostream OutStr(TBEStr);
  Error YAMLErr = writeTBEToOutputStream(OutStr, Stub);
  if (YAMLErr)
    return YAMLErr;
  OutStr.flush();

  if (WriteIfChanged) {
    if (ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
            MemoryBuffer::getFile(FilePath)) {
      // Compare TBE output with existing TBE file.
      // If TBE file unchanged, abort updating.
      if ((*BufOrError)->getBuffer() == TBEStr)
        return Error::success();
    }
  }
  // Open TBE file for writing.
  std::error_code SysErr;
  raw_fd_ostream Out(FilePath, SysErr);
  if (SysErr)
    return createStringError(SysErr, "Couldn't open `%s` for writing",
                             FilePath.data());
  Out << TBEStr;
  return Error::success();
}

/// readInputFile populates an ELFStub by attempting to read the
/// input file using both the TBE and binary ELF parsers.
static Expected<std::unique_ptr<ELFStub>> readInputFile(StringRef FilePath) {
  // Read in file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
      MemoryBuffer::getFile(FilePath);
  if (!BufOrError) {
    return createStringError(BufOrError.getError(), "Could not open `%s`",
                             FilePath.data());
  }

  std::unique_ptr<MemoryBuffer> FileReadBuffer = std::move(*BufOrError);
  ErrorCollector EC(/*UseFatalErrors=*/false);

  // First try to read as a binary (fails fast if not binary).
  if (InputFileFormat.getNumOccurrences() == 0 ||
      InputFileFormat == FileFormat::ELF) {
    Expected<std::unique_ptr<ELFStub>> StubFromELF =
        readELFFile(FileReadBuffer->getMemBufferRef());
    if (StubFromELF) {
      return std::move(*StubFromELF);
    }
    EC.addError(StubFromELF.takeError(), "BinaryRead");
  }

  // Fall back to reading as a tbe.
  if (InputFileFormat.getNumOccurrences() == 0 ||
      InputFileFormat == FileFormat::TBE) {
    Expected<std::unique_ptr<ELFStub>> StubFromTBE =
        readTBEFromBuffer(FileReadBuffer->getBuffer());
    if (StubFromTBE) {
      return std::move(*StubFromTBE);
    }
    EC.addError(StubFromTBE.takeError(), "YamlParse");
  }

  // If both readers fail, build a new error that includes all information.
  EC.addError(createStringError(errc::not_supported,
                                "No file readers succeeded reading `%s` "
                                "(unsupported/malformed file?)",
                                FilePath.data()),
              "ReadInputFile");
  EC.escalateToFatal();
  return EC.makeError();
}

static void fatalError(Error Err) {
  WithColor::defaultErrorHandler(std::move(Err));
  exit(1);
}

int main(int argc, char *argv[]) {
  // Parse arguments.
  cl::ParseCommandLineOptions(argc, argv);

  Expected<std::unique_ptr<ELFStub>> StubOrErr = readInputFile(InputFilePath);
  if (!StubOrErr)
    fatalError(StubOrErr.takeError());

  std::unique_ptr<ELFStub> TargetStub = std::move(StubOrErr.get());

  // Change SoName before emitting stubs.
  if (SOName.getNumOccurrences() == 1)
    TargetStub->SoName = SOName;

  if (EmitTBE.getNumOccurrences() == 1) {
    TargetStub->TbeVersion = TBEVersionCurrent;
    Error TBEWriteError = writeTBE(EmitTBE, *TargetStub);
    if (TBEWriteError)
      fatalError(std::move(TBEWriteError));
  }

  // Write out binary ELF stub.
  if (BinaryOutputFilePath.getNumOccurrences() == 1) {
    if (BinaryOutputTarget.getNumOccurrences() == 0)
      fatalError(createStringError(errc::not_supported,
                                   "no binary output target specified."));
    Error BinaryWriteError = writeBinaryStub(
        BinaryOutputFilePath, *TargetStub, BinaryOutputTarget, WriteIfChanged);
    if (BinaryWriteError)
      fatalError(std::move(BinaryWriteError));
  }
}
