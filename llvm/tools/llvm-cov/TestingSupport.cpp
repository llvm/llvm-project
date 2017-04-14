//===- TestingSupport.cpp - Convert objects files into test files --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <system_error>

using namespace llvm;
using namespace object;

int convertForTestingMain(int argc, const char *argv[]) {
  cl::opt<std::string> InputSourceFile(cl::Positional, cl::Required,
                                       cl::desc("<Source file>"));

  cl::opt<std::string> OutputFilename(
      "o", cl::Required,
      cl::desc(
          "File with the profile data obtained after an instrumented run"));

  cl::ParseCommandLineOptions(argc, argv, "LLVM code coverage tool\n");

  auto ObjErr = llvm::object::ObjectFile::createObjectFile(InputSourceFile);
  if (!ObjErr) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    logAllUnhandledErrors(ObjErr.takeError(), OS, "");
    OS.flush();
    errs() << "error: " << Buf;
    return 1;
  }
  ObjectFile *OF = ObjErr.get().getBinary();
  auto BytesInAddress = OF->getBytesInAddress();
  if (BytesInAddress != 8) {
    errs() << "error: 64 bit binary expected\n";
    return 1;
  }

  // Look for the sections that we are interested in.
  int FoundSectionCount = 0;
  SectionRef ProfileNames, CoverageMapping;
  for (const auto &Section : OF->sections()) {
    StringRef Name;
    if (Section.getName(Name))
      return 1;
    // TODO: with the current getInstrProfXXXSectionName interfaces, the
    // the  host tool is limited to read coverage section on
    // binaries with compatible profile section naming scheme as the host
    // platform. Currently, COFF format binaries have different section
    // naming scheme from the all the rest.
    if (Name == llvm::getInstrProfNameSectionName()) {
      ProfileNames = Section;
    } else if (Name == llvm::getInstrProfCoverageSectionName()) {
      CoverageMapping = Section;
    } else
      continue;
    ++FoundSectionCount;
  }
  if (FoundSectionCount != 2)
    return 1;

  // Get the contents of the given sections.
  uint64_t ProfileNamesAddress = ProfileNames.getAddress();
  StringRef CoverageMappingData;
  StringRef ProfileNamesData;
  if (CoverageMapping.getContents(CoverageMappingData) ||
      ProfileNames.getContents(ProfileNamesData))
    return 1;

  int FD;
  if (auto Err =
          sys::fs::openFileForWrite(OutputFilename, FD, sys::fs::F_None)) {
    errs() << "error: " << Err.message() << "\n";
    return 1;
  }

  raw_fd_ostream OS(FD, true);
  OS << "llvmcovmtestdata";
  encodeULEB128(ProfileNamesData.size(), OS);
  encodeULEB128(ProfileNamesAddress, OS);
  OS << ProfileNamesData;
  // Coverage mapping data is expected to have an alignment of 8.
  for (unsigned Pad = OffsetToAlignment(OS.tell(), 8); Pad; --Pad)
    OS.write(uint8_t(0));
  OS << CoverageMappingData;

  return 0;
}
