//===--- special-case-list-fuzzer.cpp - Fuzzer for special case lists -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/YAMLTraits.h"
#include <cassert>
#include <string>

llvm::Regex Infinity("^[-+]?(\\.inf|\\.Inf|\\.INF)$");
llvm::Regex Base8("^0o[0-7]+$");
llvm::Regex Base16("^0x[0-9a-fA-F]+$");
llvm::Regex Float("^[-+]?(\\.[0-9]+|[0-9]+(\\.[0-9]*)?)([eE][-+]?[0-9]+)?$");

inline bool isNumericRegex(llvm::StringRef S) {

  if (S.equals(".nan") || S.equals(".NaN") || S.equals(".NAN"))
    return true;

  if (Infinity.match(S))
    return true;

  if (Base8.match(S))
    return true;

  if (Base16.match(S))
    return true;

  if (Float.match(S))
    return true;

  return false;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  std::string Input(reinterpret_cast<const char *>(Data), Size);
  Input.erase(std::remove(Input.begin(), Input.end(), 0), Input.end());
  if (!Input.empty() && llvm::yaml::isNumeric(Input) != isNumericRegex(Input))
    LLVM_BUILTIN_TRAP;
  return 0;
}
