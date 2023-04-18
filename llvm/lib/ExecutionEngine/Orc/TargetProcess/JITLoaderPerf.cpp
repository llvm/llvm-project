//===------- JITLoaderPerf.cpp - Register profiler objects ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Register objects for access by profilers via the perf JIT interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderPerf.h"

#include "llvm/ExecutionEngine/Orc/Shared/PerfSharedStructs.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"

#include <mutex>
#include <optional>

#ifdef __linux__

#include <sys/mman.h> // mmap()
#include <time.h>     // clock_gettime(), time(), localtime_r() */
#include <unistd.h>   // for read(), close()

#define DEBUG_TYPE "orc"

// language identifier (XXX: should we generate something better from debug
// info?)
#define JIT_LANG "llvm-IR"
#define LLVM_PERF_JIT_MAGIC                                                    \
  ((uint32_t)'J' << 24 | (uint32_t)'i' << 16 | (uint32_t)'T' << 8 |            \
   (uint32_t)'D')
#define LLVM_PERF_JIT_VERSION 1

using namespace llvm;
using namespace llvm::orc;

struct PerfState {
  // cache lookups
  uint32_t Pid;

  // base directory for output data
  std::string JitPath;

  // output data stream, closed via Dumpstream
  int DumpFd = -1;

  // output data stream
  std::unique_ptr<raw_fd_ostream> Dumpstream;

  // perf mmap marker
  void *MarkerAddr = NULL;
};

// prevent concurrent dumps from messing up the output file
static std::mutex Mutex;
static std::optional<PerfState> state;

struct RecHeader {
  uint32_t Id;
  uint32_t TotalSize;
  uint64_t Timestamp;
};

struct DIR {
  RecHeader Prefix;
  uint64_t CodeAddr;
  uint64_t NrEntry;
};

struct DIE {
  uint64_t CodeAddr;
  uint32_t Line;
  uint32_t Discrim;
};

struct CLR {
  RecHeader Prefix;
  uint32_t Pid;
  uint32_t Tid;
  uint64_t Vma;
  uint64_t CodeAddr;
  uint64_t CodeSize;
  uint64_t CodeIndex;
};

struct UWR {
  RecHeader Prefix;
  uint64_t UnwindDataSize;
  uint64_t EhFrameHeaderSize;
  uint64_t MappedSize;
};

static inline uint64_t timespec_to_ns(const struct timespec *ts) {
  const uint64_t NanoSecPerSec = 1000000000;
  return ((uint64_t)ts->tv_sec * NanoSecPerSec) + ts->tv_nsec;
}

static inline uint64_t perf_get_timestamp() {
  struct timespec ts;
  int ret;

  ret = clock_gettime(CLOCK_MONOTONIC, &ts);
  if (ret)
    return 0;

  return timespec_to_ns(&ts);
}

static void writeDebugRecord(const PerfJITDebugInfoRecord &DebugRecord) {
  assert(state && "PerfState not initialized");
  LLVM_DEBUG(dbgs() << "Writing debug record with "
                    << DebugRecord.Entries.size() << " entries\n");
  size_t Written = 0;
  DIR dir{RecHeader{static_cast<uint32_t>(DebugRecord.Prefix.Id),
                    DebugRecord.Prefix.TotalSize, perf_get_timestamp()},
          DebugRecord.CodeAddr, DebugRecord.Entries.size()};
  state->Dumpstream->write(reinterpret_cast<const char *>(&dir), sizeof(dir));
  Written += sizeof(dir);
  for (auto &die : DebugRecord.Entries) {
    DIE d{die.Addr, die.Lineno, die.Discrim};
    state->Dumpstream->write(reinterpret_cast<const char *>(&d), sizeof(d));
    state->Dumpstream->write(die.Name.data(), die.Name.size() + 1);
    Written += sizeof(d) + die.Name.size() + 1;
  }
  LLVM_DEBUG(dbgs() << "wrote " << Written << " bytes of debug info\n");
}

static void writeCodeRecord(const PerfJITCodeLoadRecord &CodeRecord) {
  assert(state && "PerfState not initialized");
  uint32_t Tid = get_threadid();
  LLVM_DEBUG(dbgs() << "Writing code record with code size "
                    << CodeRecord.CodeSize << " and code index "
                    << CodeRecord.CodeIndex << "\n");
  CLR clr{RecHeader{static_cast<uint32_t>(CodeRecord.Prefix.Id),
                    CodeRecord.Prefix.TotalSize, perf_get_timestamp()},
          state->Pid,
          Tid,
          CodeRecord.Vma,
          CodeRecord.CodeAddr,
          CodeRecord.CodeSize,
          CodeRecord.CodeIndex};
  LLVM_DEBUG(dbgs() << "wrote " << sizeof(clr) << " bytes of CLR, "
                    << CodeRecord.Name.size() + 1 << " bytes of name, "
                    << CodeRecord.CodeSize << " bytes of code\n");
  state->Dumpstream->write(reinterpret_cast<const char *>(&clr), sizeof(clr));
  state->Dumpstream->write(CodeRecord.Name.data(), CodeRecord.Name.size() + 1);
  state->Dumpstream->write((const char *)CodeRecord.CodeAddr,
                           CodeRecord.CodeSize);
}

static void
writeUnwindRecord(const PerfJITCodeUnwindingInfoRecord &UnwindRecord) {
  assert(state && "PerfState not initialized");
  dbgs() << "Writing unwind record with unwind data size "
         << UnwindRecord.UnwindDataSize << " and EH frame header size "
         << UnwindRecord.EHFrameHdrSize << " and mapped size "
         << UnwindRecord.MappedSize << "\n";
  UWR uwr{RecHeader{static_cast<uint32_t>(UnwindRecord.Prefix.Id),
                    UnwindRecord.Prefix.TotalSize, perf_get_timestamp()},
          UnwindRecord.UnwindDataSize, UnwindRecord.EHFrameHdrSize,
          UnwindRecord.MappedSize};
  LLVM_DEBUG(dbgs() << "wrote " << sizeof(uwr) << " bytes of UWR, "
                    << UnwindRecord.EHFrameHdrSize
                    << " bytes of EH frame header, "
                    << UnwindRecord.UnwindDataSize - UnwindRecord.EHFrameHdrSize
                    << " bytes of EH frame\n");
  state->Dumpstream->write(reinterpret_cast<const char *>(&uwr), sizeof(uwr));
  if (UnwindRecord.EHFrameHdrAddr) {
    state->Dumpstream->write((const char *)UnwindRecord.EHFrameHdrAddr,
                             UnwindRecord.EHFrameHdrSize);
  } else {
    state->Dumpstream->write(UnwindRecord.EHFrameHdr.data(),
                             UnwindRecord.EHFrameHdrSize);
  }
  state->Dumpstream->write((const char *)UnwindRecord.EHFrameAddr,
                           UnwindRecord.UnwindDataSize -
                               UnwindRecord.EHFrameHdrSize);
}

static Error registerJITLoaderPerfImpl(const PerfJITRecordBatch &Batch) {
  if (!state) {
    return make_error<StringError>("PerfState not initialized",
                                   inconvertibleErrorCode());
  }

  // Serialize the batch
  std::lock_guard<std::mutex> Lock(Mutex);
  if (Batch.UnwindingRecord.Prefix.TotalSize > 0) {
    writeUnwindRecord(Batch.UnwindingRecord);
  }
  for (const auto &DebugInfo : Batch.DebugInfoRecords) {
    writeDebugRecord(DebugInfo);
  }
  for (const auto &CodeLoad : Batch.CodeLoadRecords) {
    writeCodeRecord(CodeLoad);
  }

  state->Dumpstream->flush();

  return Error::success();
}

struct Header {
  uint32_t Magic;     // characters "JiTD"
  uint32_t Version;   // header version
  uint32_t TotalSize; // total size of header
  uint32_t ElfMach;   // elf mach target
  uint32_t Pad1;      // reserved
  uint32_t Pid;
  uint64_t Timestamp; // timestamp
  uint64_t Flags;     // flags
};

static Error OpenMarker(PerfState &state) {
  // We mmap the jitdump to create an MMAP RECORD in perf.data file.  The mmap
  // is captured either live (perf record running when we mmap) or in deferred
  // mode, via /proc/PID/maps. The MMAP record is used as a marker of a jitdump
  // file for more meta data info about the jitted code. Perf report/annotate
  // detect this special filename and process the jitdump file.
  //
  // Mapping must be PROT_EXEC to ensure it is captured by perf record
  // even when not using -d option.
  state.MarkerAddr =
      ::mmap(NULL, sys::Process::getPageSizeEstimate(), PROT_READ | PROT_EXEC,
             MAP_PRIVATE, state.DumpFd, 0);

  if (state.MarkerAddr == MAP_FAILED) {
    return make_error<llvm::StringError>("could not mmap JIT marker",
                                         inconvertibleErrorCode());
  }
  return Error::success();
}

void CloseMarker(PerfState &state) {
  if (!state.MarkerAddr)
    return;

  munmap(state.MarkerAddr, sys::Process::getPageSizeEstimate());
  state.MarkerAddr = nullptr;
}

static Expected<Header> FillMachine(PerfState &state) {
  Header hdr;
  hdr.Magic = LLVM_PERF_JIT_MAGIC;
  hdr.Version = LLVM_PERF_JIT_VERSION;
  hdr.TotalSize = sizeof(hdr);
  hdr.Pid = state.Pid;
  hdr.Timestamp = perf_get_timestamp();

  char id[16];
  struct {
    uint16_t e_type;
    uint16_t e_machine;
  } info;

  size_t RequiredMemory = sizeof(id) + sizeof(info);

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileSlice("/proc/self/exe", RequiredMemory, 0);

  // This'll not guarantee that enough data was actually read from the
  // underlying file. Instead the trailing part of the buffer would be
  // zeroed. Given the ELF signature check below that seems ok though,
  // it's unlikely that the file ends just after that, and the
  // consequence would just be that perf wouldn't recognize the
  // signature.
  if (!MB) {
    return make_error<llvm::StringError>("could not open /proc/self/exe",
                                         MB.getError());
  }

  memcpy(&id, (*MB)->getBufferStart(), sizeof(id));
  memcpy(&info, (*MB)->getBufferStart() + sizeof(id), sizeof(info));

  // check ELF signature
  if (id[0] != 0x7f || id[1] != 'E' || id[2] != 'L' || id[3] != 'F') {
    return make_error<llvm::StringError>("invalid ELF signature",
                                         inconvertibleErrorCode());
  }

  hdr.ElfMach = info.e_machine;

  return hdr;
}

static Error InitDebuggingDir(PerfState &state) {
  time_t Time;
  struct tm LocalTime;
  char TimeBuffer[sizeof("YYYYMMDD")];
  SmallString<64> Path;

  // search for location to dump data to
  if (const char *BaseDir = getenv("JITDUMPDIR"))
    Path.append(BaseDir);
  else if (!sys::path::home_directory(Path))
    Path = ".";

  // create debug directory
  Path += "/.debug/jit/";
  if (auto EC = sys::fs::create_directories(Path)) {
    std::string errstr;
    raw_string_ostream errstream(errstr);
    errstream << "could not create jit cache directory " << Path << ": "
              << EC.message() << "\n";
    return make_error<StringError>(std::move(errstr), inconvertibleErrorCode());
  }

  // create unique directory for dump data related to this process
  time(&Time);
  localtime_r(&Time, &LocalTime);
  strftime(TimeBuffer, sizeof(TimeBuffer), "%Y%m%d", &LocalTime);
  Path += JIT_LANG "-jit-";
  Path += TimeBuffer;

  SmallString<128> UniqueDebugDir;

  using sys::fs::createUniqueDirectory;
  if (auto EC = createUniqueDirectory(Path, UniqueDebugDir)) {
    std::string errstr;
    raw_string_ostream errstream(errstr);
    errstream << "could not create unique jit cache directory "
              << UniqueDebugDir << ": " << EC.message() << "\n";
    return make_error<StringError>(std::move(errstr), inconvertibleErrorCode());
  }

  state.JitPath = std::string(UniqueDebugDir.str());

  return Error::success();
}

static Error registerJITLoaderPerfStartImpl() {
  PerfState tentative;
  tentative.Pid = sys::Process::getProcessId();
  // check if clock-source is supported
  if (!perf_get_timestamp()) {
    return make_error<StringError>("kernel does not support CLOCK_MONOTONIC",
                                   inconvertibleErrorCode());
  }

  if (auto err = InitDebuggingDir(tentative)) {
    return std::move(err);
  }

  std::string Filename;
  raw_string_ostream FilenameBuf(Filename);
  FilenameBuf << tentative.JitPath << "/jit-" << tentative.Pid << ".dump";

  // Need to open ourselves, because we need to hand the FD to OpenMarker() and
  // raw_fd_ostream doesn't expose the FD.
  using sys::fs::openFileForWrite;
  if (auto EC = openFileForReadWrite(FilenameBuf.str(), tentative.DumpFd,
                                     sys::fs::CD_CreateNew, sys::fs::OF_None)) {
    std::string errstr;
    raw_string_ostream errstream(errstr);
    errstream << "could not open JIT dump file " << FilenameBuf.str() << ": "
              << EC.message() << "\n";
    return make_error<StringError>(std::move(errstr), inconvertibleErrorCode());
  }

  tentative.Dumpstream =
      std::make_unique<raw_fd_ostream>(tentative.DumpFd, true);

  auto header = FillMachine(tentative);
  if (!header) {
    return header.takeError();
  }

  // signal this process emits JIT information
  if (auto err = OpenMarker(tentative)) {
    return std::move(err);
  }

  tentative.Dumpstream->write(reinterpret_cast<const char *>(&header.get()),
                              sizeof(*header));

  // Everything initialized, can do profiling now.
  if (tentative.Dumpstream->has_error()) {
    return make_error<StringError>("could not write JIT dump header",
                                   inconvertibleErrorCode());
  }
  state = std::move(tentative);
  return Error::success();
}

static Error registerJITLoaderPerfEndImpl() {
  if (!state) {
    return make_error<StringError>("PerfState not initialized",
                                   inconvertibleErrorCode());
  }
  RecHeader close;
  close.Id = static_cast<uint32_t>(PerfJITRecordType::JIT_CODE_CLOSE);
  close.TotalSize = sizeof(close);
  close.Timestamp = perf_get_timestamp();
  state->Dumpstream->write(reinterpret_cast<const char *>(&close),
                           sizeof(close));
  if (state->MarkerAddr) {
    CloseMarker(*state);
  }
  state.reset();
  return Error::success();
}

extern "C" llvm::orc::shared::CWrapperFunctionResult
llvm_orc_registerJITLoaderPerfImpl(const char *Data, uint64_t Size) {
  using namespace orc::shared;
  return WrapperFunction<SPSError(SPSPerfJITRecordBatch)>::handle(
             Data, Size, registerJITLoaderPerfImpl)
      .release();
}

extern "C" llvm::orc::shared::CWrapperFunctionResult
llvm_orc_registerJITLoaderPerfStart(const char *Data, uint64_t Size) {
  using namespace orc::shared;
  return WrapperFunction<SPSError()>::handle(Data, Size,
                                             registerJITLoaderPerfStartImpl)
      .release();
}

extern "C" llvm::orc::shared::CWrapperFunctionResult
llvm_orc_registerJITLoaderPerfEnd(const char *Data, uint64_t Size) {
  using namespace orc::shared;
  return WrapperFunction<SPSError()>::handle(Data, Size,
                                             registerJITLoaderPerfEndImpl)
      .release();
}

#else

static Error badOS() {
  return make_error<StringError>(
      "unsupported OS (perf support is only available on linux!)",
      inconvertibleErrorCode());
}

static Error badOSBatch(PerfJITRecordBatch &Batch) { return badOS(); }

extern "C" llvm::orc::shared::CWrapperFunctionResult
llvm_orc_registerJITLoaderPerfImpl(const char *Data, uint64_t Size) {
  using namespace orc::shared;
  return WrapperFunction<SPSError(SPSPerfJITRecordBatch)>::handle(Data, Size,
                                                                  badOSBatch)
      .release();
}

extern "C" llvm::orc::shared::CWrapperFunctionResult
llvm_orc_registerJITLoaderPerfStart(const char *Data, uint64_t Size) {
  using namespace orc::shared;
  return WrapperFunction<SPSError()>::handle(Data, Size, badOS).release();
}

extern "C" llvm::orc::shared::CWrapperFunctionResult
llvm_orc_registerJITLoaderPerfEnd(const char *Data, uint64_t Size) {
  using namespace orc::shared;
  return WrapperFunction<SPSError()>::handle(Data, Size, badOS).release();
}

#endif