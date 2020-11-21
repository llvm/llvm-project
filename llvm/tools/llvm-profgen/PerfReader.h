//===-- PerfReader.h - perfscript reader -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_PERFREADER_H
#define LLVM_TOOLS_LLVM_PROFGEN_PERFREADER_H
#include "ErrorHandling.h"
#include "ProfiledBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include <fstream>
#include <list>
#include <map>
#include <vector>

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

// Stream based trace line iterator
class TraceStream {
  std::string CurrentLine;
  std::ifstream Fin;
  bool IsAtEoF = false;
  uint64_t LineNumber = 0;

public:
  TraceStream(StringRef Filename) : Fin(Filename.str()) {
    if (!Fin.good())
      exitWithError("Error read input perf script file", Filename);
    advance();
  }

  StringRef getCurrentLine() {
    assert(!IsAtEoF && "Line iterator reaches the End-of-File!");
    return CurrentLine;
  }

  uint64_t getLineNumber() { return LineNumber; }

  bool isAtEoF() { return IsAtEoF; }

  // Read the next line
  void advance() {
    if (!std::getline(Fin, CurrentLine)) {
      IsAtEoF = true;
      return;
    }
    LineNumber++;
  }
};

// Filename to binary map
using BinaryMap = StringMap<ProfiledBinary>;
// Address to binary map for fast look-up
using AddressBinaryMap = std::map<uint64_t, ProfiledBinary *>;

// Load binaries and read perf trace to parse the events and samples
class PerfReader {

  BinaryMap BinaryTable;
  AddressBinaryMap AddrToBinaryMap; // Used by address-based lookup.

  // The parsed MMap event
  struct MMapEvent {
    uint64_t PID = 0;
    uint64_t BaseAddress = 0;
    uint64_t Size = 0;
    uint64_t Offset = 0;
    StringRef BinaryPath;
  };

  /// Load symbols and disassemble the code of a give binary.
  /// Also register the binary in the binary table.
  ///
  ProfiledBinary &loadBinary(const StringRef BinaryPath,
                             bool AllowNameConflict = true);
  void updateBinaryAddress(const MMapEvent &Event);

public:
  PerfReader(cl::list<std::string> &BinaryFilenames);

  /// Parse a single line of a PERF_RECORD_MMAP2 event looking for a
  /// mapping between the binary name and its memory layout.
  ///
  void parseMMap2Event(TraceStream &TraceIt);
  void parseEvent(TraceStream &TraceIt);
  // Parse perf events and samples
  void parseTrace(StringRef Filename);
  void parsePerfTraces(cl::list<std::string> &PerfTraceFilenames);
};

} // end namespace sampleprof
} // end namespace llvm

#endif
