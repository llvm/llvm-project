//===---------------- ARMTargetParserCommon ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Code that is common to ARMTargetParser and AArch64TargetParser.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ARMTargetParserCommon.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;

StringRef ARM::getArchSynonym(StringRef Arch) {
  return StringSwitch<StringRef>(Arch)
      .Case("v5", "v5t")
      .Case("v5e", "v5te")
      .Case("v6j", "v6")
      .Case("v6hl", "v6k")
      .Cases("v6m", "v6sm", "v6s-m", "v6-m")
      .Cases("v6z", "v6zk", "v6kz")
      .Cases("v7", "v7a", "v7hl", "v7l", "v7-a")
      .Case("v7r", "v7-r")
      .Case("v7m", "v7-m")
      .Case("v7em", "v7e-m")
      .Cases("v8", "v8a", "v8l", "aarch64", "arm64", "v8-a")
      .Case("v8.1a", "v8.1-a")
      .Case("v8.2a", "v8.2-a")
      .Case("v8.3a", "v8.3-a")
      .Case("v8.4a", "v8.4-a")
      .Case("v8.5a", "v8.5-a")
      .Case("v8.6a", "v8.6-a")
      .Case("v8.7a", "v8.7-a")
      .Case("v8.8a", "v8.8-a")
      .Case("v8r", "v8-r")
      .Cases("v9", "v9a", "v9-a")
      .Case("v9.1a", "v9.1-a")
      .Case("v9.2a", "v9.2-a")
      .Case("v9.3a", "v9.3-a")
      .Case("v8m.base", "v8-m.base")
      .Case("v8m.main", "v8-m.main")
      .Case("v8.1m.main", "v8.1-m.main")
      .Default(Arch);
}

StringRef ARM::getCanonicalArchName(StringRef Arch) {
  size_t offset = StringRef::npos;
  StringRef A = Arch;
  StringRef Error = "";

  // Begins with "arm" / "thumb", move past it.
  if (A.startswith("arm64_32"))
    offset = 8;
  else if (A.startswith("arm64e"))
    offset = 6;
  else if (A.startswith("arm64"))
    offset = 5;
  else if (A.startswith("aarch64_32"))
    offset = 10;
  else if (A.startswith("arm"))
    offset = 3;
  else if (A.startswith("thumb"))
    offset = 5;
  else if (A.startswith("aarch64")) {
    offset = 7;
    // AArch64 uses "_be", not "eb" suffix.
    if (A.contains("eb"))
      return Error;
    if (A.substr(offset, 3) == "_be")
      offset += 3;
  }

  // Ex. "armebv7", move past the "eb".
  if (offset != StringRef::npos && A.substr(offset, 2) == "eb")
    offset += 2;
  // Or, if it ends with eb ("armv7eb"), chop it off.
  else if (A.endswith("eb"))
    A = A.substr(0, A.size() - 2);
  // Trim the head
  if (offset != StringRef::npos)
    A = A.substr(offset);

  // Empty string means offset reached the end, which means it's valid.
  if (A.empty())
    return Arch;

  // Only match non-marketing names
  if (offset != StringRef::npos) {
    // Must start with 'vN'.
    if (A.size() >= 2 && (A[0] != 'v' || !std::isdigit(A[1])))
      return Error;
    // Can't have an extra 'eb'.
    if (A.contains("eb"))
      return Error;
  }

  // Arch will either be a 'v' name (v7a) or a marketing name (xscale).
  return A;
}
