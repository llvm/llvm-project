//===------------------- SanitizerIngestor.cpp - LLVM Advisor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of SanitizerIngestor in Runtime
//
//===----------------------------------------------------------------------===//
#include "Runtime/SanitizerIngestor.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

static StringRef sanitizeSeverity(StringRef Check) {
  if (Check.contains_insensitive("heap-use-after-free") ||
      Check.contains_insensitive("stack-use-after-return") ||
      Check.contains_insensitive("data-race"))
    return "critical";
  if (Check.contains_insensitive("undefined") ||
      Check.contains_insensitive("leak"))
    return "high";
  return "medium";
}

Expected<json::Value> SanitizerIngestor::load(StringRef Path) {
  if (Path.empty())
    return createStringError(inconvertibleErrorCode(), "empty sanitizer path");
  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = MemoryBuffer::getFile(Path);
  if (!Buffer)
    return createStringError(Buffer.getError(),
                             "cannot read sanitizer report '%s'",
                             Path.data());

  json::Array Findings;
  json::Array CurrentStack;
  std::string CurrentCheck;
  std::string CurrentMessage;
  uint64_t Reports = 0;

  for (line_iterator Line(**Buffer, false); !Line.is_at_eof(); ++Line) {
    StringRef Text = (*Line).trim();
    if (Text.empty())
      continue;

    if (Text.starts_with("SUMMARY:")) {
      if (!CurrentCheck.empty()) {
        Findings.push_back(json::Object{
            {"kind", "sanitizer.report"},
            {"check_type", CurrentCheck},
            {"severity", sanitizeSeverity(CurrentCheck)},
            {"message", CurrentMessage.empty() ? Text : CurrentMessage},
            {"stack_trace", std::move(CurrentStack)}});
        CurrentStack = json::Array();
        CurrentCheck.clear();
        CurrentMessage.clear();
        ++Reports;
      }
      continue;
    }

    size_t ErrorPos = Text.find("ERROR:");
    if (ErrorPos != StringRef::npos) {
      StringRef Rest = Text.substr(ErrorPos + sizeof("ERROR:") - 1).trim();
      std::pair<StringRef, StringRef> Split = Rest.split(':');
      CurrentCheck = Split.first.trim().str();
      CurrentMessage = Rest.str();
      continue;
    }

    if (Text.starts_with("#") && !CurrentCheck.empty())
      CurrentStack.push_back(Text);
  }

  if (!CurrentCheck.empty()) {
    Findings.push_back(
        json::Object{{"kind", "sanitizer.report"},
                     {"check_type", CurrentCheck},
                     {"severity", sanitizeSeverity(CurrentCheck)},
                     {"message", CurrentMessage},
                     {"stack_trace", std::move(CurrentStack)}});
    ++Reports;
  }

  return json::Object{{"kind", "sanitizer-report"},
                      {"format", "sanitizer-text"},
                      {"path", Path},
                      {"report_count", static_cast<int64_t>(Reports)},
                      {"findings", std::move(Findings)}};
}
