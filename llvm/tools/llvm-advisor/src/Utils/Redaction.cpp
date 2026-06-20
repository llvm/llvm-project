//===------------------- Redaction.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sensitive argument and path redaction for secure logging.
// Removes secrets from stored command lines.
//
//===----------------------------------------------------------------------===//
#include "Utils/Redaction.h"

using namespace llvm;
using namespace llvm::advisor;

/// Return true if Haystack contains Needle ignoring ASCII case.
static bool containsInsensitive(StringRef Haystack, StringRef Needle) {
  if (Needle.size() > Haystack.size())
    return false;
  for (size_t I = 0, E = Haystack.size() - Needle.size() + 1; I != E; ++I) {
    bool Match = true;
    for (size_t J = 0; J != Needle.size(); ++J) {
      if (std::tolower(static_cast<unsigned char>(Haystack[I + J])) !=
          std::tolower(static_cast<unsigned char>(Needle[J]))) {
        Match = false;
        break;
      }
    }
    if (Match)
      return true;
  }
  return false;
}

static bool isSensitive(StringRef Value) {
  static const StringRef Keywords[] = {"secret", "token", "password",
                                       "apikey", "api_key"};
  for (StringRef Keyword : Keywords)
    if (containsInsensitive(Value, Keyword))
      return true;
  return false;
}

std::string llvm::advisor::redactString(StringRef Value) {
  if (isSensitive(Value))
    return "<redacted>";
  return Value.str();
}

SmallVector<std::string, 16>
llvm::advisor::redactCommand(ArrayRef<std::string> Arguments) {
  SmallVector<std::string, 16> Out;
  for (const std::string &Arg : Arguments)
    Out.push_back(redactString(Arg));
  return Out;
}
