//===- Diagnostic.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Diagnostics.h"
#include "mlir/Query/Matcher/ErrorBuilder.h"

namespace mlir::query::matcher::internal {

Diagnostics::ArgStream &
Diagnostics::ArgStream::operator<<(const llvm::Twine &arg) {
  out->push_back(arg.str());
  return *this;
}

Diagnostics::ArgStream Diagnostics::addError(SourceRange range,
                                             ErrorType error) {
  errorValues.emplace_back();
  ErrorContent &last = errorValues.back();
  last.contextStack = contextStack;
  last.messages.emplace_back();
  last.messages.back().range = range;
  last.messages.back().type = error;
  return ArgStream(&last.messages.back().args);
}

static llvm::StringRef errorTypeToFormatString(ErrorType type) {
  switch (type) {
  case ErrorType::RegistryMatcherNotFound:
    return "Matcher not found: $0";
  case ErrorType::RegistryWrongArgCount:
    return "Incorrect argument count. (Expected = $0) != (Actual = $1)";
  case ErrorType::RegistryWrongArgType:
    return "Incorrect type for arg $0. (Expected = $1) != (Actual = $2)";
  case ErrorType::RegistryValueNotFound:
    return "Value not found: $0";

  case ErrorType::ParserStringError:
    return "Error parsing string token: <$0>";
  case ErrorType::ParserNoOpenParen:
    return "Error parsing matcher. Found token <$0> while looking for '('.";
  case ErrorType::ParserNoCloseParen:
    return "Error parsing matcher. Found end-of-code while looking for ')'.";
  case ErrorType::ParserNoComma:
    return "Error parsing matcher. Found token <$0> while looking for ','.";
  case ErrorType::ParserNoCode:
    return "End of code found while looking for token.";
  case ErrorType::ParserNotAMatcher:
    return "Input value is not a matcher expression.";
  case ErrorType::ParserInvalidToken:
    return "Invalid token <$0> found when looking for a value.";
  case ErrorType::ParserTrailingCode:
    return "Unexpected end of code.";
  case ErrorType::ParserOverloadedType:
    return "Input value has unresolved overloaded type: $0";
  case ErrorType::ParserFailedToBuildMatcher:
    return "Failed to build matcher: $0.";

  case ErrorType::None:
    return "<N/A>";
  }
  llvm_unreachable("Unknown ErrorType value.");
}

static void formatErrorString(llvm::StringRef formatString,
                              llvm::ArrayRef<std::string> args,
                              llvm::raw_ostream &os) {
  while (!formatString.empty()) {
    std::pair<llvm::StringRef, llvm::StringRef> pieces =
        formatString.split("$");
    os << pieces.first.str();
    if (pieces.second.empty())
      break;

    const char next = pieces.second.front();
    formatString = pieces.second.drop_front();
    if (next >= '0' && next <= '9') {
      const unsigned index = next - '0';
      if (index < args.size()) {
        os << args[index];
      } else {
        os << "<Argument_Not_Provided>";
      }
    }
  }
}

static void maybeAddLineAndColumn(SourceRange range, llvm::raw_ostream &os) {
  if (range.start.line > 0 && range.start.column > 0) {
    os << range.start.line << ":" << range.start.column << ": ";
  }
}

void Diagnostics::printMessage(
    const Diagnostics::ErrorContent::Message &message, const llvm::Twine prefix,
    llvm::raw_ostream &os) const {
  maybeAddLineAndColumn(message.range, os);
  os << prefix;
  formatErrorString(errorTypeToFormatString(message.type), message.args, os);
}

void Diagnostics::printErrorContent(const Diagnostics::ErrorContent &content,
                                    llvm::raw_ostream &os) const {
  if (content.messages.size() == 1) {
    printMessage(content.messages[0], "", os);
  } else {
    for (size_t i = 0, e = content.messages.size(); i != e; ++i) {
      if (i != 0)
        os << "\n";
      printMessage(content.messages[i],
                   "Candidate " + llvm::Twine(i + 1) + ": ", os);
    }
  }
}

void Diagnostics::print(llvm::raw_ostream &os) const {
  for (const ErrorContent &error : errorValues) {
    if (&error != &errorValues.front())
      os << "\n";
    printErrorContent(error, os);
  }
}

} // namespace mlir::query::matcher::internal
