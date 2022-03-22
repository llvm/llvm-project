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
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableModule.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableObjectFile.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/ProfileData/RawMemProfReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "memprof"

namespace llvm {
namespace memprof {
namespace {

struct Summary {
  uint64_t Version;
  uint64_t TotalSizeBytes;
  uint64_t NumSegments;
  uint64_t NumMIBInfo;
  uint64_t NumStackOffsets;
};

template <class T = uint64_t> inline T alignedRead(const char *Ptr) {
  static_assert(std::is_pod<T>::value, "Not a pod type.");
  assert(reinterpret_cast<size_t>(Ptr) % sizeof(T) == 0 && "Unaligned Read");
  return *reinterpret_cast<const T *>(Ptr);
}

Summary computeSummary(const char *Start) {
  auto *H = reinterpret_cast<const Header *>(Start);

  // Check alignment while reading the number of items in each section.
  return Summary{
      H->Version,
      H->TotalSize,
      alignedRead(Start + H->SegmentOffset),
      alignedRead(Start + H->MIBOffset),
      alignedRead(Start + H->StackOffset),
  };
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
      endian::readNext<uint64_t, little, unaligned>(Ptr);
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
      endian::readNext<uint64_t, little, unaligned>(Ptr);
  llvm::SmallVector<std::pair<uint64_t, MemInfoBlock>> Items;
  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    const uint64_t Id = endian::readNext<uint64_t, little, unaligned>(Ptr);
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
      endian::readNext<uint64_t, little, unaligned>(Ptr);
  CallStackMap Items;

  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    const uint64_t StackId = endian::readNext<uint64_t, little, unaligned>(Ptr);
    const uint64_t NumPCs = endian::readNext<uint64_t, little, unaligned>(Ptr);

    SmallVector<uint64_t> CallStack;
    for (uint64_t J = 0; J < NumPCs; J++) {
      CallStack.push_back(endian::readNext<uint64_t, little, unaligned>(Ptr));
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
  return StringRef(llvm::sys::path::convert_to_slash(Path))
      .contains("memprof/memprof_");
}
} // namespace

Expected<std::unique_ptr<RawMemProfReader>>
RawMemProfReader::create(const Twine &Path, const StringRef ProfiledBinary) {
  auto BufferOr = MemoryBuffer::getFileOrSTDIN(Path);
  if (std::error_code EC = BufferOr.getError())
    return report(errorCodeToError(EC), Path.getSingleStringRef());

  std::unique_ptr<MemoryBuffer> Buffer(BufferOr.get().release());
  if (Error E = checkBuffer(*Buffer))
    return report(std::move(E), Path.getSingleStringRef());

  if (ProfiledBinary.empty())
    return report(
        errorCodeToError(make_error_code(std::errc::invalid_argument)),
        "Path to profiled binary is empty!");

  auto BinaryOr = llvm::object::createBinary(ProfiledBinary);
  if (!BinaryOr) {
    return report(BinaryOr.takeError(), ProfiledBinary);
  }

  std::unique_ptr<RawMemProfReader> Reader(
      new RawMemProfReader(std::move(Buffer), std::move(BinaryOr.get())));
  if (Error E = Reader->initialize()) {
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
  OS << "MemprofProfile:\n";
  // TODO: Update printSummaries to print out the data after the profile has
  // been symbolized and pruned. We can parse some raw profile characteristics
  // from the data buffer for additional information.
  printSummaries(OS);
  // Print out the merged contents of the profiles.
  OS << "  Records:\n";
  for (const auto &Entry : *this) {
    OS << "  -\n";
    OS << "    FunctionGUID: " << Entry.first << "\n";
    Entry.second.print(OS);
  }
}

void RawMemProfReader::printSummaries(raw_ostream &OS) const {
  const char *Next = DataBuffer->getBufferStart();
  while (Next < DataBuffer->getBufferEnd()) {
    auto Summary = computeSummary(Next);
    OS << "  -\n";
    OS << "  Header:\n";
    OS << "    Version: " << Summary.Version << "\n";
    OS << "    TotalSizeBytes: " << Summary.TotalSizeBytes << "\n";
    OS << "    NumSegments: " << Summary.NumSegments << "\n";
    OS << "    NumMibInfo: " << Summary.NumMIBInfo << "\n";
    OS << "    NumStackOffsets: " << Summary.NumStackOffsets << "\n";
    // TODO: Print the build ids once we can record them using the
    // sanitizer_procmaps library for linux.

    auto *H = reinterpret_cast<const Header *>(Next);
    Next += H->TotalSize;
  }
}

Error RawMemProfReader::initialize() {
  const StringRef FileName = Binary.getBinary()->getFileName();

  auto *ElfObject = dyn_cast<object::ELFObjectFileBase>(Binary.getBinary());
  if (!ElfObject) {
    return report(make_error<StringError>(Twine("Not an ELF file: "),
                                          inconvertibleErrorCode()),
                  FileName);
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

  if (Error E = readRawProfile())
    return E;

  if (Error E = symbolizeAndFilterStackFrames())
    return E;

  return mapRawProfileToRecords();
}

Error RawMemProfReader::mapRawProfileToRecords() {
  // Hold a mapping from function to each callsite location we encounter within
  // it that is part of some dynamic allocation context. The location is stored
  // as a pointer to a symbolized list of inline frames.
  using LocationPtr = const llvm::SmallVector<FrameId> *;
  llvm::DenseMap<GlobalValue::GUID, llvm::SetVector<LocationPtr>>
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
  for (auto I = PerFunctionCallSites.begin(), E = PerFunctionCallSites.end();
       I != E; I++) {
    const GlobalValue::GUID Id = I->first;
    // Some functions may have only callsite data and no allocation data. Here
    // we insert a new entry for callsite data if we need to.
    auto Result = FunctionProfileData.insert({Id, IndexedMemProfRecord()});
    IndexedMemProfRecord &Record = Result.first->second;
    for (LocationPtr Loc : I->getSecond()) {
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
        LLVM_DEBUG(
            // Print out the name to guid mapping for debugging.
            llvm::dbgs() << "FunctionName: " << DIFrame.FunctionName
                         << " GUID: "
                         << IndexedMemProfRecord::getGUID(DIFrame.FunctionName)
                         << "\n";);

        const Frame F(IndexedMemProfRecord::getGUID(DIFrame.FunctionName),
                      DIFrame.Line - DIFrame.StartLine, DIFrame.Column,
                      // Only the last entry is not an inlined location.
                      I != NumFrames - 1);

        const FrameId Id = F.hash();
        IdToFrame.insert({Id, F});
        SymbolizedFrame[VAddr].push_back(Id);
      }
    }

    auto &CallStack = Entry.getSecond();
    CallStack.erase(std::remove_if(CallStack.begin(), CallStack.end(),
                                   [&AllVAddrsToDiscard](const uint64_t A) {
                                     return AllVAddrsToDiscard.contains(A);
                                   }),
                    CallStack.end());
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

Error RawMemProfReader::readRawProfile() {
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
  LLVM_DEBUG({
  SegmentEntry *ContainingSegment = nullptr;
  for (auto &SE : SegmentInfo) {
    if (VirtualAddress > SE.Start && VirtualAddress <= SE.End) {
      ContainingSegment = &SE;
    }
  }

  // Ensure that the virtual address is valid.
  assert(ContainingSegment && "Could not find a segment entry");
  });

  // TODO: Compute the file offset based on the maps and program headers. For
  // now this only works for non PIE binaries.
  return object::SectionedAddress{VirtualAddress};
}

Error RawMemProfReader::readNextRecord(GuidMemProfRecordPair &GuidRecord) {
  if (FunctionProfileData.empty())
    return make_error<InstrProfError>(instrprof_error::empty_raw_profile);

  if (Iter == FunctionProfileData.end())
    return make_error<InstrProfError>(instrprof_error::eof);

  auto IdToFrameCallback = [this](const FrameId Id) {
    return this->idToFrame(Id);
  };
  const IndexedMemProfRecord &IndexedRecord = Iter->second;
  GuidRecord = {Iter->first, MemProfRecord(IndexedRecord, IdToFrameCallback)};
  Iter++;
  return Error::success();
}
} // namespace memprof
} // namespace llvm
