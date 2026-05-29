//===--- RemarksAnalysisUtils.cpp - LLVM Advisor -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/RemarksAnalysisUtils.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace llvm::advisor;

StringRef llvm::advisor::remarkTypeKey(remarks::Type T) {
  switch (T) {
  case remarks::Type::Unknown:          return "unknown";
  case remarks::Type::Passed:           return "passed";
  case remarks::Type::Missed:           return "missed";
  case remarks::Type::Analysis:         return "analysis";
  case remarks::Type::AnalysisFPCommute: return "analysis-fp-commute";
  case remarks::Type::AnalysisAliasing: return "analysis-aliasing";
  case remarks::Type::Failure:          return "failure";
  }
  return "unknown";
}

ArrayRef<StringRef> llvm::advisor::allRemarkTypeKeys() {
  static constexpr StringRef Keys[] = {
      "unknown", "passed", "missed", "analysis",
      "analysis-fp-commute", "analysis-aliasing", "failure",
  };
  return Keys;
}

static Error maybeUpgradeVersionError(Error E, StringRef Path) {
  std::string Msg = toString(std::move(E));
  if (Msg.find("Unsupported remark container version") != std::string::npos ||
      Msg.find("Unsupported remark version in container") != std::string::npos) {
    std::string Full;
    raw_string_ostream OS(Full);
    OS << "Remarks file '" << Path
       << "' was produced by an incompatible LLVM toolchain. " << Msg
       << "  Please rebuild the project with a matching LLVM version, or "
       << "regenerate the remarks with the same LLVM used by llvm-advisor.";
    OS.flush();
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             Full);
  }
  std::string Full;
  raw_string_ostream OS(Full);
  OS << "Error parsing remarks from '" << Path << "': " << Msg;
  OS.flush();
  return createStringError(std::make_error_code(std::errc::invalid_argument),
                           Full);
}

Error llvm::advisor::foreachRemark(StringRef Path, RemarkVisitor Visitor) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return createStringError(MB.getError(),
                             Twine("cannot read remarks: ") + Path);

  Expected<std::unique_ptr<remarks::RemarkParser>> Parser =
      remarks::createRemarkParser(remarks::Format::Auto,
                                   MB.get()->getBuffer());
  if (!Parser)
    return maybeUpgradeVersionError(Parser.takeError(), Path);

  while (true) {
    Expected<std::unique_ptr<remarks::Remark>> Next = (*Parser)->next();
    if (!Next) {
      Error E = Next.takeError();
      if (E.isA<remarks::EndOfFileError>()) {
        consumeError(std::move(E));
        break;
      }
      return maybeUpgradeVersionError(std::move(E), Path);
    }
    if (Error E = Visitor(**Next))
      return E;
  }
  return Error::success();
}
