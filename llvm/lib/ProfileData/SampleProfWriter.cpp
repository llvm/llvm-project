//===- SampleProfWriter.cpp - Write LLVM sample profile data --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that writes LLVM sample profiles. It
// supports two file formats: text and binary. The textual representation
// is useful for debugging and testing purposes. The binary representation
// is more compact, resulting in smaller file sizes. However, they can
// both be used interchangeably.
//
// See lib/ProfileData/SampleProfReader.cpp for documentation on each of the
// supported formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <set>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;
using namespace sampleprof;

std::error_code SampleProfileWriter::writeFuncProfiles(
    const StringMap<FunctionSamples> &ProfileMap) {
  // Sort the ProfileMap by total samples.
  typedef std::pair<StringRef, const FunctionSamples *> NameFunctionSamples;
  std::vector<NameFunctionSamples> V;
  for (const auto &I : ProfileMap)
    V.push_back(std::make_pair(I.getKey(), &I.second));

  llvm::stable_sort(
      V, [](const NameFunctionSamples &A, const NameFunctionSamples &B) {
        if (A.second->getTotalSamples() == B.second->getTotalSamples())
          return A.first > B.first;
        return A.second->getTotalSamples() > B.second->getTotalSamples();
      });

  for (const auto &I : V) {
    if (std::error_code EC = writeSample(*I.second))
      return EC;
  }
  return sampleprof_error::success;
}

std::error_code
SampleProfileWriter::write(const StringMap<FunctionSamples> &ProfileMap) {
  if (std::error_code EC = writeHeader(ProfileMap))
    return EC;

  if (std::error_code EC = writeFuncProfiles(ProfileMap))
    return EC;

  return sampleprof_error::success;
}

SecHdrTableEntry &
SampleProfileWriterExtBinaryBase::getEntryInLayout(SecType Type) {
  auto SecIt = std::find_if(
      SectionHdrLayout.begin(), SectionHdrLayout.end(),
      [=](const auto &Entry) -> bool { return Entry.Type == Type; });
  return *SecIt;
}

/// Return the current position and prepare to use it as the start
/// position of a section.
uint64_t SampleProfileWriterExtBinaryBase::markSectionStart(SecType Type) {
  uint64_t SectionStart = OutputStream->tell();
  auto &Entry = getEntryInLayout(Type);
  // Use LocalBuf as a temporary output for writting data.
  if (hasSecFlag(Entry, SecFlagCompress))
    LocalBufStream.swap(OutputStream);
  return SectionStart;
}

std::error_code SampleProfileWriterExtBinaryBase::compressAndOutput() {
  if (!llvm::zlib::isAvailable())
    return sampleprof_error::zlib_unavailable;
  std::string &UncompressedStrings =
      static_cast<raw_string_ostream *>(LocalBufStream.get())->str();
  if (UncompressedStrings.size() == 0)
    return sampleprof_error::success;
  auto &OS = *OutputStream;
  SmallString<128> CompressedStrings;
  llvm::Error E = zlib::compress(UncompressedStrings, CompressedStrings,
                                 zlib::BestSizeCompression);
  if (E)
    return sampleprof_error::compress_failed;
  encodeULEB128(UncompressedStrings.size(), OS);
  encodeULEB128(CompressedStrings.size(), OS);
  OS << CompressedStrings.str();
  UncompressedStrings.clear();
  return sampleprof_error::success;
}

/// Add a new section into section header table.
std::error_code
SampleProfileWriterExtBinaryBase::addNewSection(SecType Type,
                                                uint64_t SectionStart) {
  auto Entry = getEntryInLayout(Type);
  if (hasSecFlag(Entry, SecFlagCompress)) {
    LocalBufStream.swap(OutputStream);
    if (std::error_code EC = compressAndOutput())
      return EC;
  }
  SecHdrTable.push_back({Type, Entry.Flags, SectionStart - FileStart,
                         OutputStream->tell() - SectionStart});
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterExtBinaryBase::write(
    const StringMap<FunctionSamples> &ProfileMap) {
  if (std::error_code EC = writeHeader(ProfileMap))
    return EC;

  std::string LocalBuf;
  LocalBufStream = std::make_unique<raw_string_ostream>(LocalBuf);
  if (std::error_code EC = writeSections(ProfileMap))
    return EC;

  if (std::error_code EC = writeSecHdrTable())
    return EC;

  return sampleprof_error::success;
}

std::error_code
SampleProfileWriterExtBinary::writeSample(const FunctionSamples &S) {
  uint64_t Offset = OutputStream->tell();
  StringRef Name = S.getName();
  FuncOffsetTable[Name] = Offset - SecLBRProfileStart;
  encodeULEB128(S.getHeadSamples(), *OutputStream);
  return writeBody(S);
}

std::error_code SampleProfileWriterExtBinary::writeFuncOffsetTable() {
  auto &OS = *OutputStream;

  // Write out the table size.
  encodeULEB128(FuncOffsetTable.size(), OS);

  // Write out FuncOffsetTable.
  for (auto entry : FuncOffsetTable) {
    writeNameIdx(entry.first);
    encodeULEB128(entry.second, OS);
  }
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterExtBinary::writeSections(
    const StringMap<FunctionSamples> &ProfileMap) {
  uint64_t SectionStart = markSectionStart(SecProfSummary);
  computeSummary(ProfileMap);
  if (auto EC = writeSummary())
    return EC;
  if (std::error_code EC = addNewSection(SecProfSummary, SectionStart))
    return EC;

  // Generate the name table for all the functions referenced in the profile.
  SectionStart = markSectionStart(SecNameTable);
  for (const auto &I : ProfileMap) {
    addName(I.first());
    addNames(I.second);
  }
  writeNameTable();
  if (std::error_code EC = addNewSection(SecNameTable, SectionStart))
    return EC;

  SectionStart = markSectionStart(SecLBRProfile);
  SecLBRProfileStart = OutputStream->tell();
  if (std::error_code EC = writeFuncProfiles(ProfileMap))
    return EC;
  if (std::error_code EC = addNewSection(SecLBRProfile, SectionStart))
    return EC;

  if (ProfSymList && ProfSymList->toCompress())
    setToCompressSection(SecProfileSymbolList);

  SectionStart = markSectionStart(SecProfileSymbolList);
  if (ProfSymList && ProfSymList->size() > 0)
    if (std::error_code EC = ProfSymList->write(*OutputStream))
      return EC;
  if (std::error_code EC = addNewSection(SecProfileSymbolList, SectionStart))
    return EC;

  SectionStart = markSectionStart(SecFuncOffsetTable);
  if (std::error_code EC = writeFuncOffsetTable())
    return EC;
  if (std::error_code EC = addNewSection(SecFuncOffsetTable, SectionStart))
    return EC;

  return sampleprof_error::success;
}

std::error_code SampleProfileWriterCompactBinary::write(
    const StringMap<FunctionSamples> &ProfileMap) {
  if (std::error_code EC = SampleProfileWriter::write(ProfileMap))
    return EC;
  if (std::error_code EC = writeFuncOffsetTable())
    return EC;
  return sampleprof_error::success;
}

/// Write samples to a text file.
///
/// Note: it may be tempting to implement this in terms of
/// FunctionSamples::print().  Please don't.  The dump functionality is intended
/// for debugging and has no specified form.
///
/// The format used here is more structured and deliberate because
/// it needs to be parsed by the SampleProfileReaderText class.
std::error_code SampleProfileWriterText::writeSample(const FunctionSamples &S) {
  auto &OS = *OutputStream;
  OS << S.getName() << ":" << S.getTotalSamples();
  if (Indent == 0)
    OS << ":" << S.getHeadSamples();
  OS << "\n";

  SampleSorter<LineLocation, SampleRecord> SortedSamples(S.getBodySamples());
  for (const auto &I : SortedSamples.get()) {
    LineLocation Loc = I->first;
    const SampleRecord &Sample = I->second;
    OS.indent(Indent + 1);
    if (Loc.Discriminator == 0)
      OS << Loc.LineOffset << ": ";
    else
      OS << Loc.LineOffset << "." << Loc.Discriminator << ": ";

    OS << Sample.getSamples();

    for (const auto &J : Sample.getSortedCallTargets())
      OS << " " << J.first << ":" << J.second;
    OS << "\n";
  }

  SampleSorter<LineLocation, FunctionSamplesMap> SortedCallsiteSamples(
      S.getCallsiteSamples());
  Indent += 1;
  for (const auto &I : SortedCallsiteSamples.get())
    for (const auto &FS : I->second) {
      LineLocation Loc = I->first;
      const FunctionSamples &CalleeSamples = FS.second;
      OS.indent(Indent);
      if (Loc.Discriminator == 0)
        OS << Loc.LineOffset << ": ";
      else
        OS << Loc.LineOffset << "." << Loc.Discriminator << ": ";
      if (std::error_code EC = writeSample(CalleeSamples))
        return EC;
    }
  Indent -= 1;

  return sampleprof_error::success;
}

std::error_code SampleProfileWriterBinary::writeNameIdx(StringRef FName) {
  const auto &ret = NameTable.find(FName);
  if (ret == NameTable.end())
    return sampleprof_error::truncated_name_table;
  encodeULEB128(ret->second, *OutputStream);
  return sampleprof_error::success;
}

void SampleProfileWriterBinary::addName(StringRef FName) {
  NameTable.insert(std::make_pair(FName, 0));
}

void SampleProfileWriterBinary::addNames(const FunctionSamples &S) {
  // Add all the names in indirect call targets.
  for (const auto &I : S.getBodySamples()) {
    const SampleRecord &Sample = I.second;
    for (const auto &J : Sample.getCallTargets())
      addName(J.first());
  }

  // Recursively add all the names for inlined callsites.
  for (const auto &J : S.getCallsiteSamples())
    for (const auto &FS : J.second) {
      const FunctionSamples &CalleeSamples = FS.second;
      addName(CalleeSamples.getName());
      addNames(CalleeSamples);
    }
}

void SampleProfileWriterBinary::stablizeNameTable(std::set<StringRef> &V) {
  // Sort the names to make NameTable deterministic.
  for (const auto &I : NameTable)
    V.insert(I.first);
  int i = 0;
  for (const StringRef &N : V)
    NameTable[N] = i++;
}

std::error_code SampleProfileWriterBinary::writeNameTable() {
  auto &OS = *OutputStream;
  std::set<StringRef> V;
  stablizeNameTable(V);

  // Write out the name table.
  encodeULEB128(NameTable.size(), OS);
  for (auto N : V) {
    OS << N;
    encodeULEB128(0, OS);
  }
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterCompactBinary::writeFuncOffsetTable() {
  auto &OS = *OutputStream;

  // Fill the slot remembered by TableOffset with the offset of FuncOffsetTable.
  auto &OFS = static_cast<raw_fd_ostream &>(OS);
  uint64_t FuncOffsetTableStart = OS.tell();
  if (OFS.seek(TableOffset) == (uint64_t)-1)
    return sampleprof_error::ostream_seek_unsupported;
  support::endian::Writer Writer(*OutputStream, support::little);
  Writer.write(FuncOffsetTableStart);
  if (OFS.seek(FuncOffsetTableStart) == (uint64_t)-1)
    return sampleprof_error::ostream_seek_unsupported;

  // Write out the table size.
  encodeULEB128(FuncOffsetTable.size(), OS);

  // Write out FuncOffsetTable.
  for (auto entry : FuncOffsetTable) {
    writeNameIdx(entry.first);
    encodeULEB128(entry.second, OS);
  }
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterCompactBinary::writeNameTable() {
  auto &OS = *OutputStream;
  std::set<StringRef> V;
  stablizeNameTable(V);

  // Write out the name table.
  encodeULEB128(NameTable.size(), OS);
  for (auto N : V) {
    encodeULEB128(MD5Hash(N), OS);
  }
  return sampleprof_error::success;
}

std::error_code
SampleProfileWriterBinary::writeMagicIdent(SampleProfileFormat Format) {
  auto &OS = *OutputStream;
  // Write file magic identifier.
  encodeULEB128(SPMagic(Format), OS);
  encodeULEB128(SPVersion(), OS);
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterBinary::writeHeader(
    const StringMap<FunctionSamples> &ProfileMap) {
  writeMagicIdent(Format);

  computeSummary(ProfileMap);
  if (auto EC = writeSummary())
    return EC;

  // Generate the name table for all the functions referenced in the profile.
  for (const auto &I : ProfileMap) {
    addName(I.first());
    addNames(I.second);
  }

  writeNameTable();
  return sampleprof_error::success;
}

void SampleProfileWriterExtBinaryBase::setToCompressAllSections() {
  for (auto &Entry : SectionHdrLayout)
    addSecFlags(Entry, SecFlagCompress);
}

void SampleProfileWriterExtBinaryBase::setToCompressSection(SecType Type) {
  addSectionFlags(Type, SecFlagCompress);
}

void SampleProfileWriterExtBinaryBase::addSectionFlags(SecType Type,
                                                       SecFlags Flags) {
  for (auto &Entry : SectionHdrLayout) {
    if (Entry.Type == Type)
      addSecFlags(Entry, Flags);
  }
}

void SampleProfileWriterExtBinaryBase::allocSecHdrTable() {
  support::endian::Writer Writer(*OutputStream, support::little);

  Writer.write(static_cast<uint64_t>(SectionHdrLayout.size()));
  SecHdrTableOffset = OutputStream->tell();
  for (uint32_t i = 0; i < SectionHdrLayout.size(); i++) {
    Writer.write(static_cast<uint64_t>(-1));
    Writer.write(static_cast<uint64_t>(-1));
    Writer.write(static_cast<uint64_t>(-1));
    Writer.write(static_cast<uint64_t>(-1));
  }
}

std::error_code SampleProfileWriterExtBinaryBase::writeSecHdrTable() {
  auto &OFS = static_cast<raw_fd_ostream &>(*OutputStream);
  uint64_t Saved = OutputStream->tell();

  // Set OutputStream to the location saved in SecHdrTableOffset.
  if (OFS.seek(SecHdrTableOffset) == (uint64_t)-1)
    return sampleprof_error::ostream_seek_unsupported;
  support::endian::Writer Writer(*OutputStream, support::little);

  DenseMap<uint32_t, uint32_t> IndexMap;
  for (uint32_t i = 0; i < SecHdrTable.size(); i++) {
    IndexMap.insert({static_cast<uint32_t>(SecHdrTable[i].Type), i});
  }

  // Write the section header table in the order specified in
  // SectionHdrLayout. That is the sections order Reader will see.
  // Note that the sections order in which Reader expects to read
  // may be different from the order in which Writer is able to
  // write, so we need to adjust the order in SecHdrTable to be
  // consistent with SectionHdrLayout when we write SecHdrTable
  // to the memory.
  for (uint32_t i = 0; i < SectionHdrLayout.size(); i++) {
    uint32_t idx = IndexMap[static_cast<uint32_t>(SectionHdrLayout[i].Type)];
    Writer.write(static_cast<uint64_t>(SecHdrTable[idx].Type));
    Writer.write(static_cast<uint64_t>(SecHdrTable[idx].Flags));
    Writer.write(static_cast<uint64_t>(SecHdrTable[idx].Offset));
    Writer.write(static_cast<uint64_t>(SecHdrTable[idx].Size));
  }

  // Reset OutputStream.
  if (OFS.seek(Saved) == (uint64_t)-1)
    return sampleprof_error::ostream_seek_unsupported;

  return sampleprof_error::success;
}

std::error_code SampleProfileWriterExtBinaryBase::writeHeader(
    const StringMap<FunctionSamples> &ProfileMap) {
  auto &OS = *OutputStream;
  FileStart = OS.tell();
  writeMagicIdent(Format);

  allocSecHdrTable();
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterCompactBinary::writeHeader(
    const StringMap<FunctionSamples> &ProfileMap) {
  support::endian::Writer Writer(*OutputStream, support::little);
  if (auto EC = SampleProfileWriterBinary::writeHeader(ProfileMap))
    return EC;

  // Reserve a slot for the offset of function offset table. The slot will
  // be populated with the offset of FuncOffsetTable later.
  TableOffset = OutputStream->tell();
  Writer.write(static_cast<uint64_t>(-2));
  return sampleprof_error::success;
}

std::error_code SampleProfileWriterBinary::writeSummary() {
  auto &OS = *OutputStream;
  encodeULEB128(Summary->getTotalCount(), OS);
  encodeULEB128(Summary->getMaxCount(), OS);
  encodeULEB128(Summary->getMaxFunctionCount(), OS);
  encodeULEB128(Summary->getNumCounts(), OS);
  encodeULEB128(Summary->getNumFunctions(), OS);
  std::vector<ProfileSummaryEntry> &Entries = Summary->getDetailedSummary();
  encodeULEB128(Entries.size(), OS);
  for (auto Entry : Entries) {
    encodeULEB128(Entry.Cutoff, OS);
    encodeULEB128(Entry.MinCount, OS);
    encodeULEB128(Entry.NumCounts, OS);
  }
  return sampleprof_error::success;
}
std::error_code SampleProfileWriterBinary::writeBody(const FunctionSamples &S) {
  auto &OS = *OutputStream;

  if (std::error_code EC = writeNameIdx(S.getName()))
    return EC;

  encodeULEB128(S.getTotalSamples(), OS);

  // Emit all the body samples.
  encodeULEB128(S.getBodySamples().size(), OS);
  for (const auto &I : S.getBodySamples()) {
    LineLocation Loc = I.first;
    const SampleRecord &Sample = I.second;
    encodeULEB128(Loc.LineOffset, OS);
    encodeULEB128(Loc.Discriminator, OS);
    encodeULEB128(Sample.getSamples(), OS);
    encodeULEB128(Sample.getCallTargets().size(), OS);
    for (const auto &J : Sample.getSortedCallTargets()) {
      StringRef Callee = J.first;
      uint64_t CalleeSamples = J.second;
      if (std::error_code EC = writeNameIdx(Callee))
        return EC;
      encodeULEB128(CalleeSamples, OS);
    }
  }

  // Recursively emit all the callsite samples.
  uint64_t NumCallsites = 0;
  for (const auto &J : S.getCallsiteSamples())
    NumCallsites += J.second.size();
  encodeULEB128(NumCallsites, OS);
  for (const auto &J : S.getCallsiteSamples())
    for (const auto &FS : J.second) {
      LineLocation Loc = J.first;
      const FunctionSamples &CalleeSamples = FS.second;
      encodeULEB128(Loc.LineOffset, OS);
      encodeULEB128(Loc.Discriminator, OS);
      if (std::error_code EC = writeBody(CalleeSamples))
        return EC;
    }

  return sampleprof_error::success;
}

/// Write samples of a top-level function to a binary file.
///
/// \returns true if the samples were written successfully, false otherwise.
std::error_code
SampleProfileWriterBinary::writeSample(const FunctionSamples &S) {
  encodeULEB128(S.getHeadSamples(), *OutputStream);
  return writeBody(S);
}

std::error_code
SampleProfileWriterCompactBinary::writeSample(const FunctionSamples &S) {
  uint64_t Offset = OutputStream->tell();
  StringRef Name = S.getName();
  FuncOffsetTable[Name] = Offset;
  encodeULEB128(S.getHeadSamples(), *OutputStream);
  return writeBody(S);
}

/// Create a sample profile file writer based on the specified format.
///
/// \param Filename The file to create.
///
/// \param Format Encoding format for the profile file.
///
/// \returns an error code indicating the status of the created writer.
ErrorOr<std::unique_ptr<SampleProfileWriter>>
SampleProfileWriter::create(StringRef Filename, SampleProfileFormat Format) {
  std::error_code EC;
  std::unique_ptr<raw_ostream> OS;
  if (Format == SPF_Binary || Format == SPF_Ext_Binary ||
      Format == SPF_Compact_Binary)
    OS.reset(new raw_fd_ostream(Filename, EC, sys::fs::OF_None));
  else
    OS.reset(new raw_fd_ostream(Filename, EC, sys::fs::OF_Text));
  if (EC)
    return EC;

  return create(OS, Format);
}

/// Create a sample profile stream writer based on the specified format.
///
/// \param OS The output stream to store the profile data to.
///
/// \param Format Encoding format for the profile file.
///
/// \returns an error code indicating the status of the created writer.
ErrorOr<std::unique_ptr<SampleProfileWriter>>
SampleProfileWriter::create(std::unique_ptr<raw_ostream> &OS,
                            SampleProfileFormat Format) {
  std::error_code EC;
  std::unique_ptr<SampleProfileWriter> Writer;

  if (Format == SPF_Binary)
    Writer.reset(new SampleProfileWriterRawBinary(OS));
  else if (Format == SPF_Ext_Binary)
    Writer.reset(new SampleProfileWriterExtBinary(OS));
  else if (Format == SPF_Compact_Binary)
    Writer.reset(new SampleProfileWriterCompactBinary(OS));
  else if (Format == SPF_Text)
    Writer.reset(new SampleProfileWriterText(OS));
  else if (Format == SPF_GCC)
    EC = sampleprof_error::unsupported_writing_format;
  else
    EC = sampleprof_error::unrecognized_format;

  if (EC)
    return EC;

  Writer->Format = Format;
  return std::move(Writer);
}

void SampleProfileWriter::computeSummary(
    const StringMap<FunctionSamples> &ProfileMap) {
  SampleProfileSummaryBuilder Builder(ProfileSummaryBuilder::DefaultCutoffs);
  for (const auto &I : ProfileMap) {
    const FunctionSamples &Profile = I.second;
    Builder.addRecord(Profile);
  }
  Summary = Builder.getSummary();
}
