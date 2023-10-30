//===- RawMemProfReader.cpp - Instrumented memory profiling reader --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading MemProf profiling data.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableObjectFile.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/BuildID.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/RawMemProfReader.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "memprof"

namespace llvm {
namespace memprof {
namespace {
template <class T = uint64_t> inline T alignedRead(const char *Ptr) {
  static_assert(std::is_pod<T>::value, "Not a pod type.");
  assert(reinterpret_cast<size_t>(Ptr) % sizeof(T) == 0 && "Unaligned Read");
  return *reinterpret_cast<const T *>(Ptr);
}

Error checkBuffer(const MemoryBuffer &Buffer) {
  if (!RawMemProfReader::hasFormat(Buffer))
    return make_error<InstrProfError>(instrprof_error::bad_magic);

  if (Buffer.getBufferSize() == 0)
    return make_error<InstrProfError>(instrprof_error::empty_raw_profile);

  if (Buffer.getBufferSize() < sizeof(Header)) {
    return make_error<InstrProfError>(instrprof_error::truncated);
  }

  // The size of the buffer can be > header total size since we allow repeated
  // serialization of memprof profiles to the same file.
  uint64_t TotalSize = 0;
  const char *Next = Buffer.getBufferStart();
  while (Next < Buffer.getBufferEnd()) {
    auto *H = reinterpret_cast<const Header *>(Next);
    if (H->Version != MEMPROF_RAW_VERSION) {
      return make_error<InstrProfError>(instrprof_error::unsupported_version);
    }

    TotalSize += H->TotalSize;
    Next += H->TotalSize;
  }

  if (Buffer.getBufferSize() != TotalSize) {
    return make_error<InstrProfError>(instrprof_error::malformed);
  }
  return Error::success();
}

llvm::SmallVector<SegmentEntry> readSegmentEntries(const char *Ptr) {
  using namespace support;

  const uint64_t NumItemsToRead =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  llvm::SmallVector<SegmentEntry> Items;
  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    Items.push_back(*reinterpret_cast<const SegmentEntry *>(
        Ptr + I * sizeof(SegmentEntry)));
  }
  return Items;
}

llvm::SmallVector<std::pair<uint64_t, MemInfoBlock>>
readMemInfoBlocks(const char *Ptr) {
  using namespace support;

  const uint64_t NumItemsToRead =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  llvm::SmallVector<std::pair<uint64_t, MemInfoBlock>> Items;
  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    const uint64_t Id =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    const MemInfoBlock MIB = *reinterpret_cast<const MemInfoBlock *>(Ptr);
    Items.push_back({Id, MIB});
    // Only increment by size of MIB since readNext implicitly increments.
    Ptr += sizeof(MemInfoBlock);
  }
  return Items;
}

CallStackMap readStackInfo(const char *Ptr) {
  using namespace support;

  const uint64_t NumItemsToRead =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  CallStackMap Items;

  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    const uint64_t StackId =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    const uint64_t NumPCs =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);

    SmallVector<uint64_t> CallStack;
    for (uint64_t J = 0; J < NumPCs; J++) {
      CallStack.push_back(
          endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr));
    }

    Items[StackId] = CallStack;
  }
  return Items;
}

// Merges the contents of stack information in \p From to \p To. Returns true if
// any stack ids observed previously map to a different set of program counter
// addresses.
bool mergeStackMap(const CallStackMap &From, CallStackMap &To) {
  for (const auto &IdStack : From) {
    auto I = To.find(IdStack.first);
    if (I == To.end()) {
      To[IdStack.first] = IdStack.second;
    } else {
      // Check that the PCs are the same (in order).
      if (IdStack.second != I->second)
        return true;
    }
  }
  return false;
}

Error report(Error E, const StringRef Context) {
  return joinErrors(createStringError(inconvertibleErrorCode(), Context),
                    std::move(E));
}

bool isRuntimePath(const StringRef Path) {
  const StringRef Filename = llvm::sys::path::filename(Path);
  // This list should be updated in case new files with additional interceptors
  // are added to the memprof runtime.
  return Filename.equals("memprof_malloc_linux.cpp") ||
         Filename.equals("memprof_interceptors.cpp") ||
         Filename.equals("memprof_new_delete.cpp");
}

std::string getBuildIdString(const SegmentEntry &Entry) {
  // If the build id is unset print a helpful string instead of all zeros.
  if (Entry.BuildIdSize == 0)
    return "<None>";

  std::string Str;
  raw_string_ostream OS(Str);
  for (size_t I = 0; I < Entry.BuildIdSize; I++) {
    OS << format_hex_no_prefix(Entry.BuildId[I], 2);
  }
  return OS.str();
}
} // namespace

Expected<std::unique_ptr<RawMemProfReader>>
RawMemProfReader::create(const Twine &Path, const StringRef ProfiledBinary,
                         bool KeepName) {
  auto BufferOr = MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = BufferOr.getError())
    return report(errorCodeToError(EC), Path.getSingleStringRef());

  std::unique_ptr<MemoryBuffer> Buffer(BufferOr.get().release());
  return create(std::move(Buffer), ProfiledBinary, KeepName);
}

Expected<std::unique_ptr<RawMemProfReader>>
RawMemProfReader::create(std::unique_ptr<MemoryBuffer> Buffer,
                         const StringRef ProfiledBinary, bool KeepName) {
  if (Error E = checkBuffer(*Buffer))
    return report(std::move(E), Buffer->getBufferIdentifier());

  if (ProfiledBinary.empty()) {
    // Peek the build ids to print a helpful error message.
    const std::vector<std::string> BuildIds = peekBuildIds(Buffer.get());
    std::string ErrorMessage(
        R"(Path to profiled binary is empty, expected binary with one of the following build ids:
)");
    for (const auto &Id : BuildIds) {
      ErrorMessage += "\n BuildId: ";
      ErrorMessage += Id;
    }
    return report(
        make_error<StringError>(ErrorMessage, inconvertibleErrorCode()),
        /*Context=*/"");
  }

  auto BinaryOr = llvm::object::createBinary(ProfiledBinary);
  if (!BinaryOr) {
    return report(BinaryOr.takeError(), ProfiledBinary);
  }

  // Use new here since constructor is private.
  std::unique_ptr<RawMemProfReader> Reader(
      new RawMemProfReader(std::move(BinaryOr.get()), KeepName));
  if (Error E = Reader->initialize(std::move(Buffer))) {
    return std::move(E);
  }
  return std::move(Reader);
}

bool RawMemProfReader::hasFormat(const StringRef Path) {
  auto BufferOr = MemoryBuffer::getFileOrSTDIN(Path);
  if (!BufferOr)
    return false;

  std::unique_ptr<MemoryBuffer> Buffer(BufferOr.get().release());
  return hasFormat(*Buffer);
}

bool RawMemProfReader::hasFormat(const MemoryBuffer &Buffer) {
  if (Buffer.getBufferSize() < sizeof(uint64_t))
    return false;
  // Aligned read to sanity check that the buffer was allocated with at least 8b
  // alignment.
  const uint64_t Magic = alignedRead(Buffer.getBufferStart());
  return Magic == MEMPROF_RAW_MAGIC_64;
}

void RawMemProfReader::printYAML(raw_ostream &OS) {
  uint64_t NumAllocFunctions = 0, NumMibInfo = 0;
  for (const auto &KV : FunctionProfileData) {
    const size_t NumAllocSites = KV.second.AllocSites.size();
    if (NumAllocSites > 0) {
      NumAllocFunctions++;
      NumMibInfo += NumAllocSites;
    }
  }

  OS << "MemprofProfile:\n";
  OS << "  Summary:\n";
  OS << "    Version: " << MEMPROF_RAW_VERSION << "\n";
  OS << "    NumSegments: " << SegmentInfo.size() << "\n";
  OS << "    NumMibInfo: " << NumMibInfo << "\n";
  OS << "    NumAllocFunctions: " << NumAllocFunctions << "\n";
  OS << "    NumStackOffsets: " << StackMap.size() << "\n";
  // Print out the segment information.
  OS << "  Segments:\n";
  for (const auto &Entry : SegmentInfo) {
    OS << "  -\n";
    OS << "    BuildId: " << getBuildIdString(Entry) << "\n";
    OS << "    Start: 0x" << llvm::utohexstr(Entry.Start) << "\n";
    OS << "    End: 0x" << llvm::utohexstr(Entry.End) << "\n";
    OS << "    Offset: 0x" << llvm::utohexstr(Entry.Offset) << "\n";
  }
  // Print out the merged contents of the profiles.
  OS << "  Records:\n";
  for (const auto &Entry : *this) {
    OS << "  -\n";
    OS << "    FunctionGUID: " << Entry.first << "\n";
    Entry.second.print(OS);
  }
}

Error RawMemProfReader::initialize(std::unique_ptr<MemoryBuffer> DataBuffer) {
  const StringRef FileName = Binary.getBinary()->getFileName();

  auto *ElfObject = dyn_cast<object::ELFObjectFileBase>(Binary.getBinary());
  if (!ElfObject) {
    return report(make_error<StringError>(Twine("Not an ELF file: "),
                                          inconvertibleErrorCode()),
                  FileName);
  }

  // Check whether the profiled binary was built with position independent code
  // (PIC). Perform sanity checks for assumptions we rely on to simplify
  // symbolization.
  auto* Elf64LEObject = llvm::cast<llvm::object::ELF64LEObjectFile>(ElfObject);
  const llvm::object::ELF64LEFile& ElfFile = Elf64LEObject->getELFFile();
  auto PHdrsOr = ElfFile.program_headers();
  if (!PHdrsOr)
    return report(
        make_error<StringError>(Twine("Could not read program headers: "),
                                inconvertibleErrorCode()),
        FileName);

  int NumExecutableSegments = 0;
  for (const auto &Phdr : *PHdrsOr) {
    if (Phdr.p_type == ELF::PT_LOAD) {
      if (Phdr.p_flags & ELF::PF_X) {
        // We assume only one text segment in the main binary for simplicity and
        // reduce the overhead of checking multiple ranges during symbolization.
        if (++NumExecutableSegments > 1) {
          return report(
              make_error<StringError>(
                  "Expect only one executable load segment in the binary",
                  inconvertibleErrorCode()),
              FileName);
        }
        // Segment will always be loaded at a page boundary, expect it to be
        // aligned already. Assume 4K pagesize for the machine from which the
        // profile has been collected. This should be fine for now, in case we
        // want to support other pagesizes it can be recorded in the raw profile
        // during collection.
        PreferredTextSegmentAddress = Phdr.p_vaddr;
        assert(Phdr.p_vaddr == (Phdr.p_vaddr & ~(0x1000 - 1U)) &&
               "Expect p_vaddr to always be page aligned");
        assert(Phdr.p_offset == 0 && "Expect p_offset = 0 for symbolization.");
      }
    }
  }

  auto Triple = ElfObject->makeTriple();
  if (!Triple.isX86())
    return report(make_error<StringError>(Twine("Unsupported target: ") +
                                              Triple.getArchName(),
                                          inconvertibleErrorCode()),
                  FileName);

  auto *Object = cast<object::ObjectFile>(Binary.getBinary());
  std::unique_ptr<DIContext> Context = DWARFContext::create(
      *Object, DWARFContext::ProcessDebugRelocations::Process);

  auto SOFOr = symbolize::SymbolizableObjectFile::create(
      Object, std::move(Context), /*UntagAddresses=*/false);
  if (!SOFOr)
    return report(SOFOr.takeError(), FileName);
  Symbolizer = std::move(SOFOr.get());

  // Process the raw profile.
  if (Error E = readRawProfile(std::move(DataBuffer)))
    return E;

  if (Error E = setupForSymbolization())
    return E;

  if (Error E = symbolizeAndFilterStackFrames())
    return E;

  return mapRawProfileToRecords();
}

Error RawMemProfReader::setupForSymbolization() {
  auto *Object = cast<object::ObjectFile>(Binary.getBinary());
  object::BuildIDRef BinaryId = object::getBuildID(Object);
  if (BinaryId.empty())
    return make_error<StringError>(Twine("No build id found in binary ") +
                                       Binary.getBinary()->getFileName(),
                                   inconvertibleErrorCode());

  int NumMatched = 0;
  for (const auto &Entry : SegmentInfo) {
    llvm::ArrayRef<uint8_t> SegmentId(Entry.BuildId, Entry.BuildIdSize);
    if (BinaryId == SegmentId) {
      // We assume only one text segment in the main binary for simplicity and
      // reduce the overhead of checking multiple ranges during symbolization.
      if (++NumMatched > 1) {
        return make_error<StringError>(
            "We expect only one executable segment in the profiled binary",
            inconvertibleErrorCode());
      }
      ProfiledTextSegmentStart = Entry.Start;
      ProfiledTextSegmentEnd = Entry.End;
    }
  }
  assert(NumMatched != 0 && "No matching executable segments in segment info.");
  assert((PreferredTextSegmentAddress == 0 ||
          (PreferredTextSegmentAddress == ProfiledTextSegmentStart)) &&
         "Expect text segment address to be 0 or equal to profiled text "
         "segment start.");
  return Error::success();
}

Error RawMemProfReader::mapRawProfileToRecords() {
  // Hold a mapping from function to each callsite location we encounter within
  // it that is part of some dynamic allocation context. The location is stored
  // as a pointer to a symbolized list of inline frames.
  using LocationPtr = const llvm::SmallVector<FrameId> *;
  llvm::MapVector<GlobalValue::GUID, llvm::SetVector<LocationPtr>>
      PerFunctionCallSites;

  // Convert the raw profile callstack data into memprof records. While doing so
  // keep track of related contexts so that we can fill these in later.
  for (const auto &Entry : CallstackProfileData) {
    const uint64_t StackId = Entry.first;

    auto It = StackMap.find(StackId);
    if (It == StackMap.end())
      return make_error<InstrProfError>(
          instrprof_error::malformed,
          "memprof callstack record does not contain id: " + Twine(StackId));

    // Construct the symbolized callstack.
    llvm::SmallVector<FrameId> Callstack;
    Callstack.reserve(It->getSecond().size());

    llvm::ArrayRef<uint64_t> Addresses = It->getSecond();
    for (size_t I = 0; I < Addresses.size(); I++) {
      const uint64_t Address = Addresses[I];
      assert(SymbolizedFrame.count(Address) > 0 &&
             "Address not found in SymbolizedFrame map");
      const SmallVector<FrameId> &Frames = SymbolizedFrame[Address];

      assert(!idToFrame(Frames.back()).IsInlineFrame &&
             "The last frame should not be inlined");

      // Record the callsites for each function. Skip the first frame of the
      // first address since it is the allocation site itself that is recorded
      // as an alloc site.
      for (size_t J = 0; J < Frames.size(); J++) {
        if (I == 0 && J == 0)
          continue;
        // We attach the entire bottom-up frame here for the callsite even
        // though we only need the frames up to and including the frame for
        // Frames[J].Function. This will enable better deduplication for
        // compression in the future.
        const GlobalValue::GUID Guid = idToFrame(Frames[J]).Function;
        PerFunctionCallSites[Guid].insert(&Frames);
      }

      // Add all the frames to the current allocation callstack.
      Callstack.append(Frames.begin(), Frames.end());
    }

    // We attach the memprof record to each function bottom-up including the
    // first non-inline frame.
    for (size_t I = 0; /*Break out using the condition below*/; I++) {
      const Frame &F = idToFrame(Callstack[I]);
      auto Result =
          FunctionProfileData.insert({F.Function, IndexedMemProfRecord()});
      IndexedMemProfRecord &Record = Result.first->second;
      Record.AllocSites.emplace_back(Callstack, Entry.second);

      if (!F.IsInlineFrame)
        break;
    }
  }

  // Fill in the related callsites per function.
  for (const auto &[Id, Locs] : PerFunctionCallSites) {
    // Some functions may have only callsite data and no allocation data. Here
    // we insert a new entry for callsite data if we need to.
    auto Result = FunctionProfileData.insert({Id, IndexedMemProfRecord()});
    IndexedMemProfRecord &Record = Result.first->second;
    for (LocationPtr Loc : Locs) {
      Record.CallSites.push_back(*Loc);
    }
  }

  return Error::success();
}

Error RawMemProfReader::symbolizeAndFilterStackFrames() {
  // The specifier to use when symbolization is requested.
  const DILineInfoSpecifier Specifier(
      DILineInfoSpecifier::FileLineInfoKind::RawValue,
      DILineInfoSpecifier::FunctionNameKind::LinkageName);

  // For entries where all PCs in the callstack are discarded, we erase the
  // entry from the stack map.
  llvm::SmallVector<uint64_t> EntriesToErase;
  // We keep track of all prior discarded entries so that we can avoid invoking
  // the symbolizer for such entries.
  llvm::DenseSet<uint64_t> AllVAddrsToDiscard;
  for (auto &Entry : StackMap) {
    for (const uint64_t VAddr : Entry.getSecond()) {
      // Check if we have already symbolized and cached the result or if we
      // don't want to attempt symbolization since we know this address is bad.
      // In this case the address is also removed from the current callstack.
      if (SymbolizedFrame.count(VAddr) > 0 ||
          AllVAddrsToDiscard.contains(VAddr))
        continue;

      Expected<DIInliningInfo> DIOr = Symbolizer->symbolizeInlinedCode(
          getModuleOffset(VAddr), Specifier, /*UseSymbolTable=*/false);
      if (!DIOr)
        return DIOr.takeError();
      DIInliningInfo DI = DIOr.get();

      // Drop frames which we can't symbolize or if they belong to the runtime.
      if (DI.getFrame(0).FunctionName == DILineInfo::BadString ||
          isRuntimePath(DI.getFrame(0).FileName)) {
        AllVAddrsToDiscard.insert(VAddr);
        continue;
      }

      for (size_t I = 0, NumFrames = DI.getNumberOfFrames(); I < NumFrames;
           I++) {
        const auto &DIFrame = DI.getFrame(I);
        const uint64_t Guid =
            IndexedMemProfRecord::getGUID(DIFrame.FunctionName);
        const Frame F(Guid, DIFrame.Line - DIFrame.StartLine, DIFrame.Column,
                      // Only the last entry is not an inlined location.
                      I != NumFrames - 1);
        // Here we retain a mapping from the GUID to canonical symbol name
        // instead of adding it to the frame object directly to reduce memory
        // overhead. This is because there can be many unique frames,
        // particularly for callsite frames.
        if (KeepSymbolName) {
          StringRef CanonicalName =
              sampleprof::FunctionSamples::getCanonicalFnName(
                  DIFrame.FunctionName);
          GuidToSymbolName.insert({Guid, CanonicalName.str()});
        }

        const FrameId Hash = F.hash();
        IdToFrame.insert({Hash, F});
        SymbolizedFrame[VAddr].push_back(Hash);
      }
    }

    auto &CallStack = Entry.getSecond();
    llvm::erase_if(CallStack, [&AllVAddrsToDiscard](const uint64_t A) {
      return AllVAddrsToDiscard.contains(A);
    });
    if (CallStack.empty())
      EntriesToErase.push_back(Entry.getFirst());
  }

  // Drop the entries where the callstack is empty.
  for (const uint64_t Id : EntriesToErase) {
    StackMap.erase(Id);
    CallstackProfileData.erase(Id);
  }

  if (StackMap.empty())
    return make_error<InstrProfError>(
        instrprof_error::malformed,
        "no entries in callstack map after symbolization");

  return Error::success();
}

std::vector<std::string>
RawMemProfReader::peekBuildIds(MemoryBuffer *DataBuffer) {
  const char *Next = DataBuffer->getBufferStart();
  // Use a set + vector since a profile file may contain multiple raw profile
  // dumps, each with segment information. We want them unique and in order they
  // were stored in the profile; the profiled binary should be the first entry.
  // The runtime uses dl_iterate_phdr and the "... first object visited by
  // callback is the main program."
  // https://man7.org/linux/man-pages/man3/dl_iterate_phdr.3.html
  std::vector<std::string> BuildIds;
  llvm::SmallSet<std::string, 10> BuildIdsSet;
  while (Next < DataBuffer->getBufferEnd()) {
    auto *Header = reinterpret_cast<const memprof::Header *>(Next);

    const llvm::SmallVector<SegmentEntry> Entries =
        readSegmentEntries(Next + Header->SegmentOffset);

    for (const auto &Entry : Entries) {
      const std::string Id = getBuildIdString(Entry);
      if (BuildIdsSet.contains(Id))
        continue;
      BuildIds.push_back(Id);
      BuildIdsSet.insert(Id);
    }

    Next += Header->TotalSize;
  }
  return BuildIds;
}

Error RawMemProfReader::readRawProfile(
    std::unique_ptr<MemoryBuffer> DataBuffer) {
  const char *Next = DataBuffer->getBufferStart();

  while (Next < DataBuffer->getBufferEnd()) {
    auto *Header = reinterpret_cast<const memprof::Header *>(Next);

    // Read in the segment information, check whether its the same across all
    // profiles in this binary file.
    const llvm::SmallVector<SegmentEntry> Entries =
        readSegmentEntries(Next + Header->SegmentOffset);
    if (!SegmentInfo.empty() && SegmentInfo != Entries) {
      // We do not expect segment information to change when deserializing from
      // the same binary profile file. This can happen if dynamic libraries are
      // loaded/unloaded between profile dumping.
      return make_error<InstrProfError>(
          instrprof_error::malformed,
          "memprof raw profile has different segment information");
    }
    SegmentInfo.assign(Entries.begin(), Entries.end());

    // Read in the MemInfoBlocks. Merge them based on stack id - we assume that
    // raw profiles in the same binary file are from the same process so the
    // stackdepot ids are the same.
    for (const auto &Value : readMemInfoBlocks(Next + Header->MIBOffset)) {
      if (CallstackProfileData.count(Value.first)) {
        CallstackProfileData[Value.first].Merge(Value.second);
      } else {
        CallstackProfileData[Value.first] = Value.second;
      }
    }

    // Read in the callstack for each ids. For multiple raw profiles in the same
    // file, we expect that the callstack is the same for a unique id.
    const CallStackMap CSM = readStackInfo(Next + Header->StackOffset);
    if (StackMap.empty()) {
      StackMap = CSM;
    } else {
      if (mergeStackMap(CSM, StackMap))
        return make_error<InstrProfError>(
            instrprof_error::malformed,
            "memprof raw profile got different call stack for same id");
    }

    Next += Header->TotalSize;
  }

  return Error::success();
}

object::SectionedAddress
RawMemProfReader::getModuleOffset(const uint64_t VirtualAddress) {
  if (VirtualAddress > ProfiledTextSegmentStart &&
      VirtualAddress <= ProfiledTextSegmentEnd) {
    // For PIE binaries, the preferred address is zero and we adjust the virtual
    // address by start of the profiled segment assuming that the offset of the
    // segment in the binary is zero. For non-PIE binaries the preferred and
    // profiled segment addresses should be equal and this is a no-op.
    const uint64_t AdjustedAddress =
        VirtualAddress + PreferredTextSegmentAddress - ProfiledTextSegmentStart;
    return object::SectionedAddress{AdjustedAddress};
  }
  // Addresses which do not originate from the profiled text segment in the
  // binary are not adjusted. These will fail symbolization and be filtered out
  // during processing.
  return object::SectionedAddress{VirtualAddress};
}

Error RawMemProfReader::readNextRecord(
    GuidMemProfRecordPair &GuidRecord,
    std::function<const Frame(const FrameId)> Callback) {
  // Create a new callback for the RawMemProfRecord iterator so that we can
  // provide the symbol name if the reader was initialized with KeepSymbolName =
  // true. This is useful for debugging and testing.
  auto IdToFrameCallback = [this](const FrameId Id) {
    Frame F = this->idToFrame(Id);
    if (!this->KeepSymbolName)
      return F;
    auto Iter = this->GuidToSymbolName.find(F.Function);
    assert(Iter != this->GuidToSymbolName.end());
    F.SymbolName = Iter->getSecond();
    return F;
  };
  return MemProfReader::readNextRecord(GuidRecord, IdToFrameCallback);
}
} // namespace memprof
} // namespace llvm
