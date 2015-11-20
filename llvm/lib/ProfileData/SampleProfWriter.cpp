//===- SampleProfWriter.cpp - Write LLVM sample profile data --------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"

using namespace llvm::sampleprof;
using namespace llvm;

/// \brief Write samples to a text file.
///
/// Note: it may be tempting to implement this in terms of
/// FunctionSamples::print().  Please don't.  The dump functionality is intended
/// for debugging and has no specified form.
///
/// The format used here is more structured and deliberate because
/// it needs to be parsed by the SampleProfileReaderText class.
std::error_code SampleProfileWriterText::write(StringRef FName,
                                               const FunctionSamples &S) {
  OS << FName << ":" << S.getTotalSamples();
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

    for (const auto &J : Sample.getCallTargets())
      OS << " " << J.first() << ":" << J.second;
    OS << "\n";
  }

  SampleSorter<CallsiteLocation, FunctionSamples> SortedCallsiteSamples(
      S.getCallsiteSamples());
  Indent += 1;
  for (const auto &I : SortedCallsiteSamples.get()) {
    CallsiteLocation Loc = I->first;
    const FunctionSamples &CalleeSamples = I->second;
    OS.indent(Indent);
    if (Loc.Discriminator == 0)
      OS << Loc.LineOffset << ": ";
    else
      OS << Loc.LineOffset << "." << Loc.Discriminator << ": ";
    if (std::error_code EC = write(Loc.CalleeName, CalleeSamples))
      return EC;
  }
  Indent -= 1;

  return sampleprof_error::success;
}

std::error_code SampleProfileWriterBinary::writeNameIdx(StringRef FName) {
  const auto &ret = NameTable.find(FName);
  if (ret == NameTable.end())
    return sampleprof_error::truncated_name_table;
  encodeULEB128(ret->second, OS);
  return sampleprof_error::success;
}

void SampleProfileWriterBinary::addName(StringRef FName) {
  auto NextIdx = NameTable.size();
  NameTable.insert(std::make_pair(FName, NextIdx));
}

void SampleProfileWriterBinary::addNames(const FunctionSamples &S) {
  // Add all the names in indirect call targets.
  for (const auto &I : S.getBodySamples()) {
    const SampleRecord &Sample = I.second;
    for (const auto &J : Sample.getCallTargets())
      addName(J.first());
  }

  // Recursively add all the names for inlined callsites.
  for (const auto &J : S.getCallsiteSamples()) {
    CallsiteLocation Loc = J.first;
    const FunctionSamples &CalleeSamples = J.second;
    addName(Loc.CalleeName);
    addNames(CalleeSamples);
  }
}

std::error_code SampleProfileWriterBinary::writeHeader(
    const StringMap<FunctionSamples> &ProfileMap) {
  // Write file magic identifier.
  encodeULEB128(SPMagic(), OS);
  encodeULEB128(SPVersion(), OS);

  // Generate the name table for all the functions referenced in the profile.
  for (const auto &I : ProfileMap) {
    addName(I.first());
    addNames(I.second);
  }

  // Write out the name table.
  encodeULEB128(NameTable.size(), OS);
  for (auto N : NameTable) {
    OS << N.first;
    encodeULEB128(0, OS);
  }

  return sampleprof_error::success;
}

std::error_code SampleProfileWriterBinary::writeBody(StringRef FName,
                                                     const FunctionSamples &S) {
  if (std::error_code EC = writeNameIdx(FName))
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
    for (const auto &J : Sample.getCallTargets()) {
      StringRef Callee = J.first();
      uint64_t CalleeSamples = J.second;
      if (std::error_code EC = writeNameIdx(Callee))
        return EC;
      encodeULEB128(CalleeSamples, OS);
    }
  }

  // Recursively emit all the callsite samples.
  encodeULEB128(S.getCallsiteSamples().size(), OS);
  for (const auto &J : S.getCallsiteSamples()) {
    CallsiteLocation Loc = J.first;
    const FunctionSamples &CalleeSamples = J.second;
    encodeULEB128(Loc.LineOffset, OS);
    encodeULEB128(Loc.Discriminator, OS);
    if (std::error_code EC = writeBody(Loc.CalleeName, CalleeSamples))
      return EC;
  }

  return sampleprof_error::success;
}

/// \brief Write samples of a top-level function to a binary file.
///
/// \returns true if the samples were written successfully, false otherwise.
std::error_code SampleProfileWriterBinary::write(StringRef FName,
                                                 const FunctionSamples &S) {
  encodeULEB128(S.getHeadSamples(), OS);
  return writeBody(FName, S);
}

/// \brief Create a sample profile writer based on the specified format.
///
/// \param Filename The file to create.
///
/// \param Writer The writer to instantiate according to the specified format.
///
/// \param Format Encoding format for the profile file.
///
/// \returns an error code indicating the status of the created writer.
ErrorOr<std::unique_ptr<SampleProfileWriter>>
SampleProfileWriter::create(StringRef Filename, SampleProfileFormat Format) {
  std::error_code EC;
  std::unique_ptr<SampleProfileWriter> Writer;

  if (Format == SPF_Binary)
    Writer.reset(new SampleProfileWriterBinary(Filename, EC));
  else if (Format == SPF_Text)
    Writer.reset(new SampleProfileWriterText(Filename, EC));
  else if (Format == SPF_GCC)
    EC = sampleprof_error::unsupported_writing_format;
  else
    EC = sampleprof_error::unrecognized_format;

  if (EC)
    return EC;

  return std::move(Writer);
}
