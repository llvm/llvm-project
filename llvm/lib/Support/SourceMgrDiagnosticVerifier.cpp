//===- SourceMgrDiagnosticVerifier.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgrDiagnosticVerifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// Given a diagnostic kind, return a human readable string for it.
static StringRef getDiagKindStr(SourceMgr::DiagKind kind) {
  switch (kind) {
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

bool SourceMgrDiagnosticVerifier::ExpectedDiag::emitError(
    raw_ostream &os, SourceMgr &mgr, const Twine &msg) const {
  if (fileLoc.isValid()) {
    SMRange range(fileLoc, SMLoc::getFromPointer(fileLoc.getPointer() +
                                                 substring.size()));
    mgr.PrintMessage(os, fileLoc, SourceMgr::DK_Error, msg, range);
  } else {
    mgr.PrintMessage(os, fileLoc, SourceMgr::DK_Error, msg);
  }
  return false;
}

bool SourceMgrDiagnosticVerifier::ExpectedDiag::match(StringRef str) const {
  // If this isn't a regex diagnostic, we simply check if the string was
  // contained.
  if (substringRegex)
    return substringRegex->match(str);
  return str.contains(substring);
}

bool SourceMgrDiagnosticVerifier::ExpectedDiag::computeRegex(raw_ostream &os,
                                                             SourceMgr &mgr) {
  std::string regexStr;
  raw_string_ostream regexOS(regexStr);
  StringRef strToProcess = substring;
  while (!strToProcess.empty()) {
    // Find the next regex block.
    size_t regexIt = strToProcess.find("{{");
    if (regexIt == StringRef::npos) {
      regexOS << Regex::escape(strToProcess);
      break;
    }
    regexOS << Regex::escape(strToProcess.take_front(regexIt));
    strToProcess = strToProcess.drop_front(regexIt + 2);

    // Find the end of the regex block.
    size_t regexEndIt = strToProcess.find("}}");
    if (regexEndIt == StringRef::npos)
      return emitError(os, mgr, "found start of regex with no end '}}'");
    StringRef regexBlock = strToProcess.take_front(regexEndIt);

    // Validate that the regex is actually valid.
    std::string regexError;
    if (!Regex(regexBlock).isValid(regexError))
      return emitError(os, mgr, "invalid regex: " + regexError);

    regexOS << '(' << regexBlock << ')';
    strToProcess = strToProcess.drop_front(regexEndIt + 2);
  }
  substringRegex = Regex(regexStr);
  return true;
}

static Regex buildExpectedRegex(ArrayRef<StringRef> prefixes) {
  assert(!prefixes.empty() && "Diagnostic verifier needs at least one prefix");
  std::string pattern = "(^|[^[:alnum:]_-])(";
  llvm::interleave(
      prefixes,
      [&](StringRef prefix) {
        assert(!prefix.empty() && "Diagnostic verifier prefix cannot be empty");
        pattern += Regex::escape(prefix);
      },
      [&] { pattern += '|'; });
  pattern += ")-(error|note|remark|warning)(-re)?"
             "( *@([+-][0-9]+|above|below|unknown))?"
             "( +([0-9]+))? *{{(.*)}}$";
  return Regex(pattern);
}

SourceMgrDiagnosticVerifier::SourceMgrDiagnosticVerifier(
    ArrayRef<StringRef> prefixes)
    : expected(buildExpectedRegex(prefixes)) {}

MutableArrayRef<SourceMgrDiagnosticVerifier::ExpectedDiag>
SourceMgrDiagnosticVerifier::computeExpectedDiags(raw_ostream &os,
                                                  SourceMgr &mgr,
                                                  const MemoryBuffer *buf) {
  // If the buffer is invalid, return an empty list.
  if (!buf)
    return {};
  auto &expectedDiags = expectedDiagsPerFile[buf->getBufferIdentifier()];

  // The number of the last line that did not correlate to a designator.
  unsigned lastNonDesignatorLine = 0;

  // The indices of designators that apply to the next non designator line.
  SmallVector<unsigned, 1> designatorsForNextLine;

  // Scan the file for expected-* designators.
  SmallVector<StringRef, 100> lines;
  buf->getBuffer().split(lines, '\n');
  for (unsigned lineNo = 0, e = lines.size(); lineNo < e; ++lineNo) {
    SmallVector<StringRef, 10> matches;
    if (!expected.match(lines[lineNo].rtrim(), &matches)) {
      // Check for designators that apply to this line.
      if (!designatorsForNextLine.empty()) {
        for (unsigned diagIndex : designatorsForNextLine)
          expectedDiags[diagIndex].lineNo = lineNo + 1;
        designatorsForNextLine.clear();
      }
      lastNonDesignatorLine = lineNo;
      continue;
    }

    // Point to the start of <prefix>-<kind>.
    SMLoc expectedStart = SMLoc::getFromPointer(matches[2].data());

    SourceMgr::DiagKind kind;
    if (matches[3] == "error")
      kind = SourceMgr::DK_Error;
    else if (matches[3] == "warning")
      kind = SourceMgr::DK_Warning;
    else if (matches[3] == "remark")
      kind = SourceMgr::DK_Remark;
    else {
      assert(matches[3] == "note");
      kind = SourceMgr::DK_Note;
    }
    ExpectedDiag record(kind, lineNo + 1, expectedStart, matches[9]);

    // Check to see if this is a regex match, i.e. it includes the `-re`.
    if (!matches[4].empty() && !record.computeRegex(os, mgr)) {
      ok = false;
      continue;
    }

    if (!matches[8].empty() &&
        (matches[8].getAsInteger(10, record.numExpected) ||
         record.numExpected == 0)) {
      record.emitError(os, mgr, "invalid count '" + matches[8] + "'");
      ok = false;
      continue;
    }

    StringRef offsetMatch = matches[6];
    if (!offsetMatch.empty()) {
      // Get the integer value without the @ and +/- prefix.
      if (offsetMatch[0] == '+' || offsetMatch[0] == '-') {
        // Parse as decimal so that e.g. '@+08' is not treated as octal.
        unsigned offset;
        if (offsetMatch.drop_front().getAsInteger(10, offset)) {
          record.emitError(os, mgr,
                           "invalid line offset '" + offsetMatch + "'");
          ok = false;
          continue;
        }

        if (offsetMatch.front() == '+') {
          if (offset > e - record.lineNo) {
            record.emitError(os, mgr,
                             "line offset '" + offsetMatch +
                                 "' is past the end of the file");
            ok = false;
            continue;
          }
          record.lineNo += offset;
        } else {
          // Line 0 is allowed since MLIR uses it for the implicit top-level
          // module.
          if (offset > record.lineNo) {
            record.emitError(os, mgr,
                             "line offset '" + offsetMatch +
                                 "' is before the start of the file");
            ok = false;
            continue;
          }
          record.lineNo -= offset;
        }
      } else if (offsetMatch.consume_front("unknown")) {
        // This is matching unknown locations.
        record.fileLoc = SMLoc();
        expectedUnknownLocDiags.emplace_back(std::move(record));
        continue;
      } else if (offsetMatch.consume_front("above")) {
        // If the designator applies 'above' we add it to the last non
        // designator line.
        record.lineNo = lastNonDesignatorLine + 1;
      } else {
        // Otherwise, this is a 'below' designator and applies to the next
        // non-designator line.
        assert(offsetMatch.consume_front("below"));
        designatorsForNextLine.push_back(expectedDiags.size());

        // Set the line number to the last in the case that this designator
        // ends up dangling.
        record.lineNo = e;
      }
    }
    expectedDiags.emplace_back(std::move(record));
  }
  return expectedDiags;
}

std::optional<MutableArrayRef<SourceMgrDiagnosticVerifier::ExpectedDiag>>
SourceMgrDiagnosticVerifier::getExpectedDiags(StringRef bufName) {
  auto expectedDiags = expectedDiagsPerFile.find(bufName);
  if (expectedDiags != expectedDiagsPerFile.end())
    return MutableArrayRef<ExpectedDiag>(expectedDiags->second);
  return std::nullopt;
}

SourceMgrDiagnosticVerifier::MatchResult
SourceMgrDiagnosticVerifier::process(raw_ostream &os, SourceMgr &mgr,
                                     SourceMgr::DiagKind kind, bool hasLoc,
                                     const MemoryBuffer *buf, unsigned lineNo,
                                     StringRef message, bool reportUnexpected) {
  MutableArrayRef<ExpectedDiag> diags;
  if (hasLoc) {
    // If the buffer couldn't be resolved, `diags` stays empty: a diagnostic
    // with a location in an unknown file can never match anything.
    if (buf) {
      if (auto maybeDiags = getExpectedDiags(buf->getBufferIdentifier()))
        diags = *maybeDiags;
      else
        diags = computeExpectedDiags(os, mgr, buf);
    }
  } else {
    diags = expectedUnknownLocDiags;
  }

  // Search for a matching expected diagnostic.
  // If we find something that is close then emit a more specific error.
  ExpectedDiag *nearMiss = nullptr;

  for (auto &e : diags) {
    // File line must match (unless it's an unknown location).
    if (hasLoc && e.lineNo != lineNo)
      continue;
    if (e.numSeen == e.numExpected || !e.match(message))
      continue;
    if (e.kind == kind) {
      ++e.numSeen;
      return MatchResult::Matched;
    }
    // If this only differs based on the diagnostic kind, then consider it
    // to be a near miss.
    nearMiss = &e;
  }

  if (!reportUnexpected)
    return MatchResult::Ignored;

  ok = false;

  // Otherwise, emit an error for the near miss.
  if (nearMiss) {
    mgr.PrintMessage(os, nearMiss->fileLoc, SourceMgr::DK_Error,
                     "'" + getDiagKindStr(kind) +
                         "' diagnostic emitted when expecting a '" +
                         getDiagKindStr(nearMiss->kind) + "'");
    return MatchResult::NearMiss;
  }
  return MatchResult::Unexpected;
}

SourceMgrDiagnosticVerifier::MatchResult
SourceMgrDiagnosticVerifier::process(raw_ostream &os, SourceMgr &mgr,
                                     const SMDiagnostic &diag,
                                     bool reportUnexpected) {
  // Don't pass the verifier's own errors back to the verifier.
  SourceMgr::DiagHandlerTy handler = mgr.getDiagHandler();
  void *handlerCtx = mgr.getDiagContext();
  mgr.setDiagHandler(nullptr);

  bool hasLoc = diag.getLoc().isValid();
  const MemoryBuffer *buf = nullptr;
  if (hasLoc)
    if (unsigned id = mgr.FindBufferContainingLoc(diag.getLoc()))
      buf = mgr.getMemoryBuffer(id);
  MatchResult result =
      process(os, mgr, diag.getKind(), hasLoc, buf, diag.getLineNo(),
              diag.getMessage(), reportUnexpected);
  if (result == MatchResult::Unexpected)
    mgr.PrintMessage(
        os, SMDiagnostic(
                mgr, diag.getLoc(), diag.getFilename(), diag.getLineNo(),
                diag.getColumnNo(), SourceMgr::DK_Error,
                "unexpected " + getDiagKindStr(diag.getKind()).str() + ": " +
                    diag.getMessage().str(),
                diag.getLineContents(), diag.getRanges(), diag.getFixIts()));
  mgr.setDiagHandler(handler, handlerCtx);
  return result;
}

bool SourceMgrDiagnosticVerifier::verify(raw_ostream &os, SourceMgr &mgr) {
  // Verify that all expected errors were seen.
  auto checkExpectedDiags = [&](ExpectedDiag &diag) {
    if (diag.numSeen == diag.numExpected)
      return;
    std::string expectedMsg = ("expected " + getDiagKindStr(diag.kind) + " \"" +
                               diag.substring + "\"")
                                  .str();
    if (diag.numExpected == 1)
      diag.emitError(os, mgr, expectedMsg + " was not produced");
    else
      diag.emitError(os, mgr,
                     expectedMsg + " was produced " + Twine(diag.numSeen) +
                         " of " + Twine(diag.numExpected) + " times");
    ok = false;
  };
  for (auto &expectedDiagsPair : expectedDiagsPerFile)
    for (auto &diag : expectedDiagsPair.second)
      checkExpectedDiags(diag);
  for (auto &diag : expectedUnknownLocDiags)
    checkExpectedDiags(diag);
  expectedDiagsPerFile.clear();
  expectedUnknownLocDiags.clear();
  return ok;
}
