//===-- COFFDirectiveParser.cpp - JITLink coff directive parser --*- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// MSVC COFF directive parser
//
//===----------------------------------------------------------------------===//

#include "COFFDirectiveParser.h"

using namespace llvm;
using namespace jitlink;

#define DEBUG_TYPE "jitlink"

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "COFFOptions.inc"
#undef PREFIX

// Create table mapping all options defined in COFFOptions.td
static const opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X7, X8, X9, X10, X11, X12)      \
  {X1,                                                                         \
   X2,                                                                         \
   X10,                                                                        \
   X11,                                                                        \
   COFF_OPT_##ID,                                                              \
   opt::Option::KIND##Class,                                                   \
   X9,                                                                         \
   X8,                                                                         \
   COFF_OPT_##GROUP,                                                           \
   COFF_OPT_##ALIAS,                                                           \
   X7,                                                                         \
   X12},
#include "COFFOptions.inc"
#undef OPTION
};

class COFFOptTable : public opt::OptTable {
public:
  COFFOptTable() : OptTable(infoTable, true) {}
};

static COFFOptTable optTable;

Expected<std::unique_ptr<opt::InputArgList>>
COFFDirectiveParser::parse(StringRef Str) {
  SmallVector<StringRef, 16> Tokens;
  SmallVector<const char *, 16> Buffer;
  cl::TokenizeWindowsCommandLineNoCopy(Str, saver, Tokens);
  for (StringRef Tok : Tokens) {
    bool HasNul = Tok.end() != Str.end() && Tok.data()[Tok.size()] == '\0';
    Buffer.push_back(HasNul ? Tok.data() : saver.save(Tok).data());
  }

  unsigned missingIndex;
  unsigned missingCount;

  auto Result = std::make_unique<opt::InputArgList>(
      optTable.ParseArgs(Buffer, missingIndex, missingCount));

  if (missingCount)
    return make_error<JITLinkError>(Twine("COFF directive parsing failed: ") +
                                    Result->getArgString(missingIndex) +
                                    " missing argument");
  LLVM_DEBUG({
    for (auto *arg : Result->filtered(COFF_OPT_UNKNOWN))
      dbgs() << "Unknown coff option argument: " << arg->getAsString(*Result)
             << "\n";
  });
  return std::move(Result);
}
