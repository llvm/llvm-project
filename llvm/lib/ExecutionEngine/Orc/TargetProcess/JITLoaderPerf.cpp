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
#include "PerfJITUnwindInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ExecutionEngine/Orc/Shared/PerfSharedStructs.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Threading.h"

#include <mutex>
#include <optional>

#ifdef __linux__

#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h> // mmap()
#include <sys/stat.h>
#include <time.h> // clock_gettime()
#include <unistd.h>

#define DEBUG_TYPE "orc"

#define LLVM_PERF_JIT_MAGIC                                                    \
  ((uint32_t)'J' << 24 | (uint32_t)'i' << 16 | (uint32_t)'T' << 8 |            \
   (uint32_t)'D')
#define LLVM_PERF_JIT_VERSION 1

using namespace llvm;
using namespace llvm::orc;

struct PerfState {
  // cache lookups
  uint32_t Pid;

  // output data stream, closed via Dumpstream
  int DumpFd = -1;

  // output data stream
  std::unique_ptr<raw_fd_ostream> Dumpstream;

  // perf mmap marker
  void *MarkerAddr = nullptr;

  // Keep one dump and code-index sequence for the executor's whole lifetime,
  // even when all plugins have been destroyed and are subsequently recreated.
  uint64_t NextCodeIndex = 0;
  unsigned Users = 1;
};

// prevent concurrent dumps from messing up the output file
static std::mutex Mutex;
static std::optional<PerfState> State;

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

static inline uint64_t timespec_to_ns(const struct timespec *TS) {
  const uint64_t NanoSecPerSec = 1000000000;
  return ((uint64_t)TS->tv_sec * NanoSecPerSec) + TS->tv_nsec;
}

static inline uint64_t perf_get_timestamp() {
  timespec TS;
  if (clock_gettime(CLOCK_MONOTONIC, &TS))
    return 0;

  return timespec_to_ns(&TS);
}

static void writeDebugRecord(const PerfJITDebugInfoRecord &DebugRecord) {
  assert(State && "PerfState not initialized");
  LLVM_DEBUG(dbgs() << "Writing debug record with "
                    << DebugRecord.Entries.size() << " entries\n");
  [[maybe_unused]] size_t Written = 0;
  DIR Dir{RecHeader{static_cast<uint32_t>(DebugRecord.Prefix.Id),
                    DebugRecord.Prefix.TotalSize, perf_get_timestamp()},
          DebugRecord.CodeAddr, DebugRecord.Entries.size()};
  State->Dumpstream->write(reinterpret_cast<const char *>(&Dir), sizeof(Dir));
  Written += sizeof(Dir);
  for (auto &Die : DebugRecord.Entries) {
    DIE d{Die.Addr, Die.Lineno, Die.Discrim};
    State->Dumpstream->write(reinterpret_cast<const char *>(&d), sizeof(d));
    State->Dumpstream->write(Die.Name.data(), Die.Name.size() + 1);
    Written += sizeof(d) + Die.Name.size() + 1;
  }
  LLVM_DEBUG(dbgs() << "wrote " << Written << " bytes of debug info\n");
}

static void writeCodeRecord(const PerfJITCodeLoadRecord &CodeRecord) {
  assert(State && "PerfState not initialized");
  uint32_t Tid = get_threadid();
  uint64_t CodeIndex = State->NextCodeIndex++;
  LLVM_DEBUG(dbgs() << "Writing code record with code size "
                    << CodeRecord.CodeSize << " and code index " << CodeIndex
                    << "\n");
  CLR Clr{RecHeader{static_cast<uint32_t>(CodeRecord.Prefix.Id),
                    CodeRecord.Prefix.TotalSize, perf_get_timestamp()},
          State->Pid,
          Tid,
          CodeRecord.Vma,
          CodeRecord.CodeAddr,
          CodeRecord.CodeSize,
          CodeIndex};
  LLVM_DEBUG(dbgs() << "wrote " << sizeof(Clr) << " bytes of CLR, "
                    << CodeRecord.Name.size() + 1 << " bytes of name, "
                    << CodeRecord.CodeSize << " bytes of code\n");
  State->Dumpstream->write(reinterpret_cast<const char *>(&Clr), sizeof(Clr));
  State->Dumpstream->write(CodeRecord.Name.data(), CodeRecord.Name.size() + 1);
  State->Dumpstream->write((const char *)CodeRecord.CodeAddr,
                           CodeRecord.CodeSize);
}

static Error writeUnwindRecord(StringRef Frame) {
  assert(State && "PerfState not initialized");
  // The synthesized ELF contains this header immediately after .eh_frame.
  // A PC-relative pointer remains valid at any ELF load bias, unlike the
  // original absolute pointer into the executor's .eh_frame allocation.
  constexpr uint64_t HeaderSize = 8;
  if (Frame.size() > INT32_MAX - 4)
    return make_error<StringError>("Perf .eh_frame is too large",
                                   inconvertibleErrorCode());
  char Header[HeaderSize] = {1, dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4,
                             char(dwarf::DW_EH_PE_omit),
                             char(dwarf::DW_EH_PE_omit)};
  support::endian::write32(Header + 4, uint32_t(-int32_t(Frame.size()) - 4),
                           sys::IsLittleEndianHost ? endianness::little
                                                   : endianness::big);
  UWR Uwr{RecHeader{
              static_cast<uint32_t>(PerfJITRecordType::JIT_CODE_UNWINDING_INFO),
              uint32_t(sizeof(UWR) + Frame.size() + HeaderSize),
              perf_get_timestamp()},
          Frame.size() + HeaderSize, HeaderSize, 0};
  State->Dumpstream->write(reinterpret_cast<const char *>(&Uwr), sizeof(Uwr));
  State->Dumpstream->write(Frame.data(), Frame.size());
  State->Dumpstream->write(Header, HeaderSize);
  return Error::success();
}

static Error registerJITLoaderPerfImpl(const PerfJITRecordBatch &Batch) {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!State || !State->Users)
    return make_error<StringError>("PerfState not initialized",
                                   inconvertibleErrorCode());

  // Profiler metadata is optional. A malformed or unsupported CFI record must
  // not turn successful JIT materialization into an allocation-action failure.
  std::optional<PerfJITUnwindInfo> UnwindInfo;
  const auto &U = Batch.UnwindingRecord;
  auto SkipUnwind = [](Error Err) {
    std::string Message = toString(std::move(Err));
    LLVM_DEBUG(dbgs() << "Skipping perf unwind information: " << Message
                      << "\n");
  };
  if (U.Prefix.TotalSize > 0) {
    if (U.Prefix.Id != PerfJITRecordType::JIT_CODE_UNWINDING_INFO ||
        U.UnwindDataSize > UINT32_MAX - sizeof(UWR) ||
        U.Prefix.TotalSize != sizeof(UWR) + U.UnwindDataSize ||
        U.UnwindDataSize < U.EHFrameHdrSize ||
        U.UnwindDataSize - U.EHFrameHdrSize > SIZE_MAX ||
        U.EHFrameAddr > UINTPTR_MAX ||
        (!U.EHFrameAddr && U.UnwindDataSize != U.EHFrameHdrSize)) {
      SkipUnwind(make_error<StringError>("Invalid perf unwind data size",
                                         inconvertibleErrorCode()));
    } else {
      StringRef Frame(reinterpret_cast<const char *>(U.EHFrameAddr),
                      U.UnwindDataSize - U.EHFrameHdrSize);
      auto Info = PerfJITUnwindInfo::create(
          Frame, U.EHFrameAddr, sizeof(void *), sys::IsLittleEndianHost);
      if (Info)
        UnwindInfo = std::move(*Info);
      else
        SkipUnwind(Info.takeError());
    }
  }

  // perf consumes pending unwind/debug records on the next CODE_LOAD. Emit
  // each function's metadata separately, in the synthetic ELF's address space.
  for (const auto &CodeLoad : Batch.CodeLoadRecords) {
    if (UnwindInfo) {
      auto Frame =
          UnwindInfo->emitEHFrame(CodeLoad.CodeAddr, CodeLoad.CodeSize);
      if (Frame) {
        if (auto Err = writeUnwindRecord(*Frame))
          SkipUnwind(std::move(Err));
      } else {
        SkipUnwind(Frame.takeError());
      }
    }
    for (const auto &DebugInfo : Batch.DebugInfoRecords)
      if (DebugInfo.CodeAddr == CodeLoad.CodeAddr) {
        writeDebugRecord(DebugInfo);
        break;
      }
    writeCodeRecord(CodeLoad);
  }
  State->Dumpstream->flush();
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

static Error OpenMarker(PerfState &State) {
  // We mmap the jitdump to create an MMAP RECORD in perf.data file.  The mmap
  // is captured either live (perf record running when we mmap) or in deferred
  // mode, via /proc/PID/maps. The MMAP record is used as a marker of a jitdump
  // file for more meta data info about the jitted code. Perf report/annotate
  // detect this special filename and process the jitdump file.
  //
  // Mapping must be PROT_EXEC to ensure it is captured by perf record
  // even when not using -d option.
  State.MarkerAddr =
      ::mmap(NULL, sys::Process::getPageSizeEstimate(), PROT_READ | PROT_EXEC,
             MAP_PRIVATE, State.DumpFd, 0);

  if (State.MarkerAddr == MAP_FAILED)
    return make_error<llvm::StringError>("could not mmap JIT marker",
                                         inconvertibleErrorCode());

  return Error::success();
}

void CloseMarker(PerfState &State) {
  if (!State.MarkerAddr)
    return;

  munmap(State.MarkerAddr, sys::Process::getPageSizeEstimate());
  State.MarkerAddr = nullptr;
}

static Expected<Header> FillMachine(PerfState &State) {
  Header Hdr = {};
  Hdr.Magic = LLVM_PERF_JIT_MAGIC;
  Hdr.Version = LLVM_PERF_JIT_VERSION;
  Hdr.TotalSize = sizeof(Hdr);
  Hdr.Pid = State.Pid;
  Hdr.Timestamp = perf_get_timestamp();

  char Id[16];
  struct {
    uint16_t e_type;
    uint16_t e_machine;
  } Info;

  size_t RequiredMemory = sizeof(Id) + sizeof(Info);

  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileSlice("/proc/self/exe", RequiredMemory, 0);

  // This'll not guarantee that enough data was actually read from the
  // underlying file. Instead the trailing part of the buffer would be
  // zeroed. Given the ELF signature check below that seems ok though,
  // it's unlikely that the file ends just after that, and the
  // consequence would just be that perf wouldn't recognize the
  // signature.
  if (!MB)
    return make_error<llvm::StringError>("could not open /proc/self/exe",
                                         MB.getError());

  memcpy(&Id, (*MB)->getBufferStart(), sizeof(Id));
  memcpy(&Info, (*MB)->getBufferStart() + sizeof(Id), sizeof(Info));

  // check ELF signature
  if (Id[0] != 0x7f || Id[1] != 'E' || Id[2] != 'L' || Id[3] != 'F')
    return make_error<llvm::StringError>("invalid ELF signature",
                                         inconvertibleErrorCode());

  Hdr.ElfMach = Info.e_machine;

  return Hdr;
}

static Error registerJITLoaderPerfStartImpl() {
  std::lock_guard<std::mutex> Lock(Mutex);
  uint32_t Pid = sys::Process::getProcessId();
  if (State && State->Pid == Pid) {
    ++State->Users;
    return Error::success();
  }
  if (State) {
    CloseMarker(*State);
    State.reset();
  }
  PerfState Tentative;
  Tentative.Pid = Pid;
  // check if clock-source is supported
  if (!perf_get_timestamp())
    return make_error<StringError>("kernel does not support CLOCK_MONOTONIC",
                                   inconvertibleErrorCode());

  std::string Filename;
  raw_string_ostream FilenameBuf(Filename);
  // perf inject writes JIT DSOs beside the dump. Its libdw unwinder
  // recognizes /tmp/jitted-, so this must not use a cache subdirectory.
  FilenameBuf << "/tmp/jit-" << Tentative.Pid << ".dump";

  // This is the first start in this process. Replace data left by an earlier
  // process with the same PID, but do not follow links when opening it.
  Tentative.DumpFd =
      ::open(Filename.c_str(),
             O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (Tentative.DumpFd < 0)
    return make_error<StringError>(
        "could not open JIT dump file " + Filename,
        std::error_code(errno, std::generic_category()));
  Tentative.Dumpstream =
      std::make_unique<raw_fd_ostream>(Tentative.DumpFd, true);
  struct stat Status;
  if (::fstat(Tentative.DumpFd, &Status) < 0)
    return make_error<StringError>(
        "could not stat JIT dump file",
        std::error_code(errno, std::generic_category()));
  if (!S_ISREG(Status.st_mode) || Status.st_uid != ::geteuid() ||
      Status.st_nlink != 1)
    return make_error<StringError>(
        "JIT dump must be an owned regular file without links",
        inconvertibleErrorCode());
  if (::ftruncate(Tentative.DumpFd, 0) < 0)
    return make_error<StringError>(
        "could not truncate JIT dump file",
        std::error_code(errno, std::generic_category()));

  auto Header = FillMachine(Tentative);
  if (!Header)
    return Header.takeError();

  Tentative.Dumpstream->write(reinterpret_cast<const char *>(&Header.get()),
                              sizeof(*Header));
  Tentative.Dumpstream->flush();

  // Everything initialized, can do profiling now.
  if (Tentative.Dumpstream->has_error())
    return make_error<StringError>("could not write JIT dump header",
                                   inconvertibleErrorCode());

  // Keep this marker for subsequent starts too, so perf inject sees one dump.
  if (auto Err = OpenMarker(Tentative))
    return Err;

  State = std::move(Tentative);
  return Error::success();
}

static Error registerJITLoaderPerfEndImpl() {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (!State || !State->Users)
    return make_error<StringError>("PerfState not initialized",
                                   inconvertibleErrorCode());

  --State->Users;
  State->Dumpstream->flush();
  // CLOSE is optional. Retain the stream, marker and counter so a later start
  // appends records after the same header instead of starting a second dump.
  return Error::success();
}

extern "C" llvm::orc::shared::CWrapperFunctionBuffer
llvm_orc_registerJITLoaderPerfImpl(const char *ArgData, size_t ArgSize) {
  using namespace orc::shared;
  return WrapperFunction<SPSError(SPSPerfJITRecordBatch)>::handle(
             ArgData, ArgSize, registerJITLoaderPerfImpl)
      .release();
}

extern "C" llvm::orc::shared::CWrapperFunctionBuffer
llvm_orc_registerJITLoaderPerfStart(const char *ArgData, size_t ArgSize) {
  using namespace orc::shared;
  return WrapperFunction<SPSError()>::handle(ArgData, ArgSize,
                                             registerJITLoaderPerfStartImpl)
      .release();
}

extern "C" llvm::orc::shared::CWrapperFunctionBuffer
llvm_orc_registerJITLoaderPerfEnd(const char *ArgData, size_t ArgSize) {
  using namespace orc::shared;
  return WrapperFunction<SPSError()>::handle(ArgData, ArgSize,
                                             registerJITLoaderPerfEndImpl)
      .release();
}

#else

using namespace llvm;
using namespace llvm::orc;

static Error badOS() {
  using namespace llvm;
  return llvm::make_error<StringError>(
      "unsupported OS (perf support is only available on linux!)",
      inconvertibleErrorCode());
}

static Error badOSBatch(PerfJITRecordBatch &Batch) { return badOS(); }

extern "C" llvm::orc::shared::CWrapperFunctionBuffer
llvm_orc_registerJITLoaderPerfImpl(const char *ArgData, size_t ArgSize) {
  using namespace shared;
  return WrapperFunction<SPSError(SPSPerfJITRecordBatch)>::handle(
             ArgData, ArgSize, badOSBatch)
      .release();
}

extern "C" llvm::orc::shared::CWrapperFunctionBuffer
llvm_orc_registerJITLoaderPerfStart(const char *ArgData, size_t ArgSize) {
  using namespace shared;
  return WrapperFunction<SPSError()>::handle(ArgData, ArgSize, badOS).release();
}

extern "C" llvm::orc::shared::CWrapperFunctionBuffer
llvm_orc_registerJITLoaderPerfEnd(const char *ArgData, size_t ArgSize) {
  using namespace shared;
  return WrapperFunction<SPSError()>::handle(ArgData, ArgSize, badOS).release();
}

#endif
