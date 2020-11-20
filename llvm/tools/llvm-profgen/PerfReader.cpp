//===-- PerfReader.cpp - perfscript reader  ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PerfReader.h"

static cl::opt<bool> ShowMmapEvents("show-mmap-events", cl::ReallyHidden,
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::desc("Print binary load events."));

namespace llvm {
namespace sampleprof {

PerfReader::PerfReader(cl::list<std::string> &BinaryFilenames) {
  // Load the binaries.
  for (auto Filename : BinaryFilenames)
    loadBinary(Filename, /*AllowNameConflict*/ false);
}

ProfiledBinary &PerfReader::loadBinary(const StringRef BinaryPath,
                                       bool AllowNameConflict) {
  // The binary table is currently indexed by the binary name not the full
  // binary path. This is because the user-given path may not match the one
  // that was actually executed.
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  // Call to load the binary in the ctor of ProfiledBinary.
  auto Ret = BinaryTable.insert({BinaryName, ProfiledBinary(BinaryPath)});

  if (!Ret.second && !AllowNameConflict) {
    std::string ErrorMsg = "Binary name conflict: " + BinaryPath.str() +
                           " and " + Ret.first->second.getPath().str() + " \n";
    exitWithError(ErrorMsg);
  }

  return Ret.first->second;
}

void PerfReader::updateBinaryAddress(const MMapEvent &Event) {
  // Load the binary.
  StringRef BinaryPath = Event.BinaryPath;
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  auto I = BinaryTable.find(BinaryName);
  // Drop the event which doesn't belong to user-provided binaries
  // or if its image is loaded at the same address
  if (I == BinaryTable.end() || Event.BaseAddress == I->second.getBaseAddress())
    return;

  ProfiledBinary &Binary = I->second;

  // A binary image could be uploaded and then reloaded at different
  // place, so update the address map here
  AddrToBinaryMap.erase(Binary.getBaseAddress());
  AddrToBinaryMap[Event.BaseAddress] = &Binary;

  // Update binary load address.
  Binary.setBaseAddress(Event.BaseAddress);
}

void PerfReader::parseMMap2Event(TraceStream &TraceIt) {
  // Parse a line like:
  //  PERF_RECORD_MMAP2 2113428/2113428: [0x7fd4efb57000(0x204000) @ 0
  //  08:04 19532229 3585508847]: r-xp /usr/lib64/libdl-2.17.so
  constexpr static const char *const Pattern =
      "PERF_RECORD_MMAP2 ([0-9]+)/[0-9]+: "
      "\\[(0x[a-f0-9]+)\\((0x[a-f0-9]+)\\) @ "
      "(0x[a-f0-9]+|0) .*\\]: [-a-z]+ (.*)";
  // Field 0 - whole line
  // Field 1 - PID
  // Field 2 - base address
  // Field 3 - mmapped size
  // Field 4 - page offset
  // Field 5 - binary path
  enum EventIndex {
    WHOLE_LINE = 0,
    PID = 1,
    BASE_ADDRESS = 2,
    MMAPPED_SIZE = 3,
    PAGE_OFFSET = 4,
    BINARY_PATH = 5
  };

  Regex RegMmap2(Pattern);
  SmallVector<StringRef, 6> Fields;
  bool R = RegMmap2.match(TraceIt.getCurrentLine(), &Fields);
  if (!R) {
    std::string ErrorMsg = "Cannot parse mmap event: Line" +
                           Twine(TraceIt.getLineNumber()).str() + ": " +
                           TraceIt.getCurrentLine().str() + " \n";
    exitWithError(ErrorMsg);
  }
  MMapEvent Event;
  Fields[PID].getAsInteger(10, Event.PID);
  Fields[BASE_ADDRESS].getAsInteger(0, Event.BaseAddress);
  Fields[MMAPPED_SIZE].getAsInteger(0, Event.Size);
  Fields[PAGE_OFFSET].getAsInteger(0, Event.Offset);
  Event.BinaryPath = Fields[BINARY_PATH];
  updateBinaryAddress(Event);
  if (ShowMmapEvents) {
    outs() << "Mmap: Binary " << Event.BinaryPath << " loaded at "
           << format("0x%" PRIx64 ":", Event.BaseAddress) << " \n";
  }
}

void PerfReader::parseEvent(TraceStream &TraceIt) {
  if (TraceIt.getCurrentLine().startswith("PERF_RECORD_MMAP2"))
    parseMMap2Event(TraceIt);

  TraceIt.advance();
}

void PerfReader::parseTrace(StringRef Filename) {
  // Trace line iterator
  TraceStream TraceIt(Filename);
  while (!TraceIt.isAtEoF()) {
    parseEvent(TraceIt);
  }
}

void PerfReader::parsePerfTraces(cl::list<std::string> &PerfTraceFilenames) {
  // Parse perf traces.
  for (auto Filename : PerfTraceFilenames)
    parseTrace(Filename);
}

} // namespace sampleprof
} // namespace llvm
