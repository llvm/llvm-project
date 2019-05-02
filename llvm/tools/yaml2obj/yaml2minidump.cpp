//===- yaml2minidump.cpp - Convert a YAML file to a minidump file ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ObjectYAML/MinidumpYAML.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int yaml2minidump(MinidumpYAML::Object &Doc, raw_ostream &Out) {
  writeAsBinary(Doc, Out);
  return 0;
}
