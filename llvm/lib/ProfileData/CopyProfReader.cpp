//===-- CopyProfReader.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file implements the CopyProf profile data reader, parsing raw binary
/// profile files and resolving stack frames via DWARF debug information.
///
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CopyProfReader.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableObjectFile.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/BuildID.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "copyprof"

namespace llvm::copyprof {
namespace {

uint64_t alignedRead(const char *Ptr) {
  assert(reinterpret_cast<size_t>(Ptr) % sizeof(uint64_t) == 0 &&
         "Unaligned Read");
  return *reinterpret_cast<const uint64_t *>(Ptr);
}

llvm::SmallVector<SegmentEntry> readSegmentEntries(const char *Ptr) {
  using namespace support;
  const uint64_t NumItemsToRead =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  llvm::SmallVector<SegmentEntry> Items;
  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    Items.push_back(*reinterpret_cast<const SegmentEntry *>(
        Ptr + I * sizeof(SegmentEntry)));
  }
  return Items;
}

llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t>>
readStackInfo(const char *Ptr) {
  using namespace support;
  const uint64_t NumItemsToRead =
      endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t>> Items;

  for (uint64_t I = 0; I < NumItemsToRead; I++) {
    const uint64_t StackId =
        endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    const uint64_t NumPCs =
        endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

    SmallVector<uint64_t> CallStack;
    CallStack.reserve(NumPCs);
    for (uint64_t J = 0; J < NumPCs; J++) {
      CallStack.push_back(
          endian::readNext<uint64_t, llvm::endianness::little>(Ptr));
    }

    Items[StackId] = CallStack;
  }
  return Items;
}
} // namespace

RawCopyProfReader::~RawCopyProfReader() = default;

bool RawCopyProfReader::hasFormat(const MemoryBuffer &Buffer) {
  if (Buffer.getBufferSize() < sizeof(uint64_t))
    return false;
  const uint64_t Magic = alignedRead(Buffer.getBufferStart());
  return Magic == RawMagic64;
}

bool RawCopyProfReader::hasFormat(const StringRef Path) {
  auto BufferOr = MemoryBuffer::getFileOrSTDIN(Path);
  if (!BufferOr)
    return false;
  return hasFormat(*BufferOr.get());
}

Expected<std::unique_ptr<RawCopyProfReader>>
RawCopyProfReader::create(const Twine &ProfilePath, StringRef ProfiledBinary) {
  auto ProfileBufferOr = MemoryBuffer::getFileOrSTDIN(ProfilePath);
  if (std::error_code EC = ProfileBufferOr.getError())
    return errorCodeToError(EC);

  auto BinaryOr = llvm::object::createBinary(ProfiledBinary);
  if (!BinaryOr)
    return BinaryOr.takeError();

  std::unique_ptr<RawCopyProfReader> Reader(new RawCopyProfReader(
      std::move(BinaryOr.get()), std::move(ProfileBufferOr.get())));

  if (Error E = Reader->initialize())
    return std::move(E);

  return std::move(Reader);
}

Error RawCopyProfReader::initialize() {
  if (Error E = readRawProfile())
    return E;

  if (Error E = setupForSymbolization())
    return E;

  return Error::success();
}

Error RawCopyProfReader::readRawProfile() {
  const char *Ptr = ProfileBuffer->getBufferStart();
  const auto *Hdr = reinterpret_cast<const Header *>(Ptr);

  if (Hdr->Magic != RawMagic64)
    return createStringError(inconvertibleErrorCode(), "Bad magic number");

  CopyProfRawVersion = Hdr->Version;

  SegmentInfo = readSegmentEntries(Ptr + Hdr->SegmentOffset);
  StackMap = readStackInfo(Ptr + Hdr->StackOffset);

  const char *ReportPtr = Ptr + Hdr->ReportOffset;
  size_t NumReports = Hdr->NumReports;

  for (size_t I = 0; I < NumReports; ++I) {
    Reports.push_back(
        *reinterpret_cast<const Report *>(ReportPtr + I * sizeof(Report)));
  }

  return Error::success();
}

Error RawCopyProfReader::setupForSymbolization() {
  auto *Object = cast<object::ObjectFile>(Binary.getBinary());
  object::BuildIDRef BinaryId = object::getBuildID(Object);
  if (BinaryId.empty())
    return createStringError(inconvertibleErrorCode(),
                             "No build id found in binary");

  int NumMatched = 0;
  for (const auto &Entry : SegmentInfo) {
    llvm::ArrayRef<uint8_t> SegmentId(Entry.BuildId, Entry.BuildIdSize);
    if (BinaryId == SegmentId) {
      if (++NumMatched > 1) {
        return createStringError(
            inconvertibleErrorCode(),
            "Expect only one executable segment in the profiled binary");
      }
      ProfiledTextSegmentStart = Entry.Start;
      ProfiledTextSegmentEnd = Entry.End;
    }
  }

  if (NumMatched == 0)
    return createStringError(inconvertibleErrorCode(),
                             "No matching executable segments found in binary");

  // Note that the current implementation only supports ELF files for
  // symbolization.
  auto *ElfObject = dyn_cast<object::ELFObjectFileBase>(Binary.getBinary());
  if (!ElfObject)
    return createStringError(inconvertibleErrorCode(), "Not an ELF file");

  auto *Elf64LEObject = llvm::cast<llvm::object::ELF64LEObjectFile>(ElfObject);
  const llvm::object::ELF64LEFile &ElfFile = Elf64LEObject->getELFFile();
  auto PHdrsOr = ElfFile.program_headers();
  if (!PHdrsOr)
    return PHdrsOr.takeError();

  for (const auto &Phdr : *PHdrsOr) {
    if (Phdr.p_type == ELF::PT_LOAD && (Phdr.p_flags & ELF::PF_X)) {
      PreferredTextSegmentAddress = Phdr.p_vaddr;
      break;
    }
  }

  return Error::success();
}

Error RawCopyProfReader::symbolizeAndFilterStackFrames(
    std::unique_ptr<llvm::symbolize::SymbolizableModule> Symbolizer) {
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
      if (SymbolizedFrame.count(VAddr) > 0 ||
          AllVAddrsToDiscard.contains(VAddr))
        continue;

      Expected<DIInliningInfo> DI = Symbolizer->symbolizeInlinedCode(
          getModuleOffset(VAddr), Specifier, /*UseSymbolTable=*/false);
      if (!DI)
        return DI.takeError();
      // Drop frames which we can't symbolize.
      if (DI->getFrame(0).FunctionName == DILineInfo::BadString) {
        AllVAddrsToDiscard.insert(VAddr);
        continue;
      }
      for (size_t I = 0, NumFrames = DI->getNumberOfFrames(); I < NumFrames;
           ++I) {
        const auto &DIFrame = DI->getFrame(I);
        const uint64_t Guid = memprof::getGUID(DIFrame.FunctionName);
        memprof::Frame F(Guid, DIFrame.Line - DIFrame.StartLine, DIFrame.Column,
                         I != NumFrames - 1);
        F.SymbolName = std::make_unique<std::string>(DIFrame.FunctionName);
        SymbolizedFrame[VAddr].push_back(F);
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
  for (const uint64_t Id : EntriesToErase)
    StackMap.erase(Id);
  if (StackMap.empty())
    return createStringError(inconvertibleErrorCode(),
                             "no entries in callstack map after symbolization");
  return Error::success();
}

object::SectionedAddress
RawCopyProfReader::getModuleOffset(const uint64_t VirtualAddress) {
  if (VirtualAddress > ProfiledTextSegmentStart &&
      VirtualAddress <= ProfiledTextSegmentEnd) {
    // Map the runtime virtual address back to the link-time address space
    // based on the preferred load address of the text segment. This is
    // necessary for correct symbolization when ASLR is active.
    const uint64_t AdjustedAddress =
        VirtualAddress + PreferredTextSegmentAddress - ProfiledTextSegmentStart;
    return object::SectionedAddress{AdjustedAddress};
  }
  return object::SectionedAddress{VirtualAddress};
}

Error RawCopyProfReader::readAndSymbolize() {
  auto *Object = cast<object::ObjectFile>(Binary.getBinary());
  auto ObjectFile = symbolize::SymbolizableObjectFile::create(
      Object,
      DWARFContext::create(*Object,
                           DWARFContext::ProcessDebugRelocations::Process),
      /*UntagAddresses=*/false);
  if (!ObjectFile)
    return ObjectFile.takeError();
  if (Error E = symbolizeAndFilterStackFrames(std::move(*ObjectFile)))
    return E;

  for (const Report &Report : Reports) {
    auto It = StackMap.find(Report.stack_trace_id);
    if (It == StackMap.end())
      continue;
    SymbolizedReport SymReport;
    SymReport.CopyNumBytes = Report.copy_num_bytes;
    for (uint64_t VAddr : It->getSecond()) {
      if (auto FrameIt = SymbolizedFrame.find(VAddr);
          FrameIt != SymbolizedFrame.end()) {
        for (const memprof::Frame &F : FrameIt->getSecond()) {
          SymReport.CallStack.push_back(F);
        }
      }
    }
    SymbolizedReports.push_back(SymReport);
  }

  return Error::success();
}

void RawCopyProfReader::printYAML(raw_ostream &OS, bool Demangle) {
  OS << "CopyProfProfile:\n";
  OS << "  Reports:\n";
  for (const SymbolizedReport &Report : SymbolizedReports) {
    OS << "  -\n";
    OS << "    CopyNumBytes: " << Report.CopyNumBytes << "\n";
    OS << "    CallStack:\n";
    for (const memprof::Frame &F : Report.CallStack) {
      OS << "    -\n";
      OS << "      Function: " << F.Function << "\n";
      std::string SymName = F.getSymbolNameOr("<None>");
      OS << "      SymbolName: "
         << (Demangle ? llvm::demangle(SymName) : SymName) << "\n";
      OS << "      LineOffset: " << F.LineOffset << "\n";
      OS << "      Column: " << F.Column << "\n";
      OS << "      Inline: " << F.IsInlineFrame << "\n";
    }
  }
}

} // namespace llvm::copyprof
