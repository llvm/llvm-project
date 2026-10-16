//===- SourceMgrDiagnosticVerifier.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgrDiagnosticVerifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagKindStr(SourceMgr::DiagKind Kind) {
  switch (Kind) {
  case SourceMgr::DK_Note:
    return "note";
  case SourceMgr::DK_Warning:
    return "warning";
  case SourceMgr::DK_Error:
    return "error";
  case SourceMgr::DK_Remark:
    return "remark";
  }
  llvm_unreachable("Unknown SourceMgr::DiagKind");
}

bool SourceMgrDiagnosticVerifier::ExpectedDiag::emitError(raw_ostream &OS,
                                                          SourceMgr &Mgr,
                                                          const Twine &Msg) const {
  if (FileLoc.isValid()) {
    SMRange Range(FileLoc, SMLoc::getFromPointer(FileLoc.getPointer() +
                                                 Substring.size()));
    Mgr.PrintMessage(OS, FileLoc, SourceMgr::DK_Error, Msg, Range);
  } else {
    Mgr.PrintMessage(OS, FileLoc, SourceMgr::DK_Error, Msg);
  }
  return false;
}

bool SourceMgrDiagnosticVerifier::ExpectedDiag::match(StringRef Str) const {
  // If this isn't a regex diagnostic, we simply check if the string was
  // contained.
  if (SubstringRegex)
    return SubstringRegex->match(Str);
  return Str.contains(Substring);
}

bool SourceMgrDiagnosticVerifier::ExpectedDiag::computeRegex(raw_ostream &OS,
                                                             SourceMgr &Mgr) {
  std::string RegexStr;
  raw_string_ostream RegexOS(RegexStr);
  StringRef StrToProcess = Substring;
  while (!StrToProcess.empty()) {
    // Find the next regex block.
    size_t RegexIt = StrToProcess.find("{{");
    if (RegexIt == StringRef::npos) {
      RegexOS << Regex::escape(StrToProcess);
      break;
    }
    RegexOS << Regex::escape(StrToProcess.take_front(RegexIt));
    StrToProcess = StrToProcess.drop_front(RegexIt + 2);

    // Find the end of the regex block.
    size_t RegexEndIt = StrToProcess.find("}}");
    if (RegexEndIt == StringRef::npos)
      return emitError(OS, Mgr, "found start of regex with no end '}}'");
    StringRef RegexBlock = StrToProcess.take_front(RegexEndIt);

    // Validate that the regex is actually valid.
    std::string RegexError;
    if (!Regex(RegexBlock).isValid(RegexError))
      return emitError(OS, Mgr, "invalid regex: " + RegexError);

    RegexOS << '(' << RegexBlock << ')';
    StrToProcess = StrToProcess.drop_front(RegexEndIt + 2);
  }
  SubstringRegex = Regex(RegexStr);
  return true;
}

MutableArrayRef<SourceMgrDiagnosticVerifier::ExpectedDiag>
SourceMgrDiagnosticVerifier::computeExpectedDiags(raw_ostream &OS,
                                                  SourceMgr &Mgr,
                                                  const MemoryBuffer *Buf) {
  // If the buffer is invalid, return an empty list.
  if (!Buf)
    return {};
  // Idempotent: a buffer that's already been scanned is returned as-is,
  // rather than being rescanned and appending duplicate entries.
  if (auto MaybeDiags = getExpectedDiags(Buf->getBufferIdentifier()))
    return *MaybeDiags;
  auto &ExpectedDiags = ExpectedDiagsPerFile[Buf->getBufferIdentifier()];

  // The number of the last line that did not correlate to a designator.
  unsigned LastNonDesignatorLine = 0;

  // The indices of designators that apply to the next non designator line.
  SmallVector<unsigned, 1> DesignatorsForNextLine;

  // Scan the file for expected-* designators.
  SmallVector<StringRef, 100> Lines;
  Buf->getBuffer().split(Lines, '\n');
  for (unsigned LineNo = 0, E = Lines.size(); LineNo < E; ++LineNo) {
    SmallVector<StringRef, 4> Matches;
    if (!Expected.match(Lines[LineNo].rtrim(), &Matches)) {
      // Check for designators that apply to this line.
      if (!DesignatorsForNextLine.empty()) {
        for (unsigned DiagIndex : DesignatorsForNextLine)
          ExpectedDiags[DiagIndex].LineNo = LineNo + 1;
        DesignatorsForNextLine.clear();
      }
      LastNonDesignatorLine = LineNo;
      continue;
    }

    // Point to the start of expected-*.
    SMLoc ExpectedStart = SMLoc::getFromPointer(Matches[0].data());

    SourceMgr::DiagKind Kind;
    if (Matches[1] == "error")
      Kind = SourceMgr::DK_Error;
    else if (Matches[1] == "warning")
      Kind = SourceMgr::DK_Warning;
    else if (Matches[1] == "remark")
      Kind = SourceMgr::DK_Remark;
    else {
      assert(Matches[1] == "note");
      Kind = SourceMgr::DK_Note;
    }
    ExpectedDiag Record(Kind, LineNo + 1, ExpectedStart, Matches[5]);

    // Check to see if this is a regex match, i.e. it includes the `-re`.
    if (!Matches[2].empty() && !Record.computeRegex(OS, Mgr)) {
      OK = false;
      continue;
    }

    StringRef OffsetMatch = Matches[3];
    if (!OffsetMatch.empty()) {
      OffsetMatch = OffsetMatch.drop_front(1);

      // Get the integer value without the @ and +/- prefix.
      if (OffsetMatch[0] == '+' || OffsetMatch[0] == '-') {
        int Offset;
        // The regex only ever captures decimal digits here, so parse as
        // decimal explicitly: a leading zero (e.g. '@+08') would otherwise
        // be mis-parsed as octal by getAsInteger's radix auto-detection
        // (and '08' isn't valid octal, so that already-broken corner case
        // would silently discard the offset instead of reporting an error).
        if (OffsetMatch.drop_front().getAsInteger(10, Offset)) {
          OK = false;
          Record.emitError(OS, Mgr, "invalid line offset '" + OffsetMatch + "'");
          continue;
        }

        if (OffsetMatch.front() == '+') {
          Record.LineNo += Offset;
        } else if (static_cast<unsigned>(Offset) >= Record.LineNo) {
          // Line numbers are 1-based, so an offset that would take LineNo to
          // 0 or below is out of range.
          OK = false;
          Record.emitError(OS, Mgr,
                           "line offset '" + OffsetMatch +
                               "' is before the start of the file");
          continue;
        } else {
          Record.LineNo -= Offset;
        }
      } else if (OffsetMatch.consume_front("unknown")) {
        // This is matching unknown locations.
        Record.FileLoc = SMLoc();
        ExpectedUnknownLocDiags.emplace_back(std::move(Record));
        continue;
      } else if (OffsetMatch.consume_front("above")) {
        // If the designator applies 'above' we add it to the last non
        // designator line.
        Record.LineNo = LastNonDesignatorLine + 1;
      } else {
        // Otherwise, this is a 'below' designator and applies to the next
        // non-designator line.
        assert(OffsetMatch.consume_front("below"));
        DesignatorsForNextLine.push_back(ExpectedDiags.size());

        // Set the line number to the last in the case that this designator
        // ends up dangling.
        Record.LineNo = E;
      }
    }
    ExpectedDiags.emplace_back(std::move(Record));
  }
  return ExpectedDiags;
}

std::optional<MutableArrayRef<SourceMgrDiagnosticVerifier::ExpectedDiag>>
SourceMgrDiagnosticVerifier::getExpectedDiags(StringRef BufName) {
  auto ExpectedDiags = ExpectedDiagsPerFile.find(BufName);
  if (ExpectedDiags != ExpectedDiagsPerFile.end())
    return MutableArrayRef<ExpectedDiag>(ExpectedDiags->second);
  return std::nullopt;
}

SourceMgrDiagnosticVerifier::MatchResult SourceMgrDiagnosticVerifier::process(
    raw_ostream &OS, SourceMgr &Mgr, SourceMgr::DiagKind Kind, bool HasLoc,
    const MemoryBuffer *Buf, unsigned LineNo, StringRef Message,
    bool ReportUnexpected) {
  MutableArrayRef<ExpectedDiag> Diags;
  if (HasLoc) {
    // If the buffer couldn't be resolved, `Diags` stays empty: a diagnostic
    // with a location in an unknown file can never match anything.
    if (Buf)
      Diags = computeExpectedDiags(OS, Mgr, Buf);
  } else {
    Diags = ExpectedUnknownLocDiags;
  }

  // Search for a matching expected diagnostic.
  // If we find something that is close then emit a more specific error.
  ExpectedDiag *NearMiss = nullptr;

  // If this was an expected error, remember that we saw it and return.
  for (auto &E : Diags) {
    // File line must match (unless it's an unknown location).
    if (HasLoc && E.LineNo != LineNo)
      continue;
    if (E.match(Message)) {
      if (E.Kind == Kind) {
        E.Matched = true;
        return MatchResult::Matched;
      }

      // If this only differs based on the diagnostic kind, then consider it
      // to be a near miss.
      NearMiss = &E;
    }
  }

  if (!ReportUnexpected)
    return MatchResult::Ignored;

  OK = false;

  // Otherwise, emit an error for the near miss.
  if (NearMiss) {
    Mgr.PrintMessage(OS, NearMiss->FileLoc, SourceMgr::DK_Error,
                     "'" + getDiagKindStr(Kind) +
                         "' diagnostic emitted when expecting a '" +
                         getDiagKindStr(NearMiss->Kind) + "'");
    return MatchResult::NearMiss;
  }
  return MatchResult::Unexpected;
}

bool SourceMgrDiagnosticVerifier::verify(raw_ostream &OS, SourceMgr &Mgr) {
  // Scan any buffers that process() never had a reason to look at (e.g. an
  // included file where the diagnostic its 'expected-*' comment names never
  // actually fired): otherwise that expectation would never be recorded, and
  // so never reported as missing below, letting verification silently
  // pass when it shouldn't. computeExpectedDiags is idempotent, so
  // re-scanning an already-known buffer here is harmless.
  for (unsigned I = 0, E = Mgr.getNumBuffers(); I != E; ++I)
    (void)computeExpectedDiags(OS, Mgr, Mgr.getMemoryBuffer(I + 1));

  // Verify that all expected errors were seen.
  auto CheckExpectedDiags = [&](ExpectedDiag &Diag) {
    if (!Diag.Matched) {
      Diag.emitError(OS, Mgr,
                     "expected " + getDiagKindStr(Diag.Kind) + " \"" +
                         Diag.Substring + "\" was not produced");
      OK = false;
    }
  };
  for (auto &ExpectedDiagsPair : ExpectedDiagsPerFile)
    for (auto &Diag : ExpectedDiagsPair.second)
      CheckExpectedDiags(Diag);
  for (auto &Diag : ExpectedUnknownLocDiags)
    CheckExpectedDiags(Diag);
  ExpectedDiagsPerFile.clear();
  return OK;
}
