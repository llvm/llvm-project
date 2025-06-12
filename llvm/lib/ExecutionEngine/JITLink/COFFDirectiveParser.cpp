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

#include <array>

using namespace llvm;
using namespace jitlink;

#define DEBUG_TYPE "jitlink"

#define OPTTABLE_STR_TABLE_CODE
#include "COFFOptions.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "COFFOptions.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

#define OPTTABLE_PREFIXES_UNION_CODE
#include "COFFOptions.inc"
#undef OPTTABLE_PREFIXES_UNION_CODE

// Create table mapping all options defined in COFFOptions.td
using namespace llvm::opt;
static constexpr opt::OptTable::Info infoTable[] = {
#define OPTION(...)                                                            \
  LLVM_CONSTRUCT_OPT_INFO_WITH_ID_PREFIX(COFF_OPT_, __VA_ARGS__),
#include "COFFOptions.inc"
#undef OPTION
};

class COFFOptTable : public opt::PrecomputedOptTable {
public:
  COFFOptTable()
      : PrecomputedOptTable(OptionStrTable, OptionPrefixesTable, infoTable,
                            OptionPrefixesUnion, true) {}
};

static COFFOptTable optTable;

Expected<opt::InputArgList> COFFDirectiveParser::parse(StringRef Str) {
  SmallVector<StringRef, 16> Tokens;
  SmallVector<const char *, 16> Buffer;
  cl::TokenizeWindowsCommandLineNoCopy(Str, saver, Tokens);
  for (StringRef Tok : Tokens) {
    bool HasNul = Tok.end() != Str.end() && Tok.data()[Tok.size()] == '\0';
    Buffer.push_back(HasNul ? Tok.data() : saver.save(Tok).data());
  }

  unsigned missingIndex;
  unsigned missingCount;

  auto Result = optTable.ParseArgs(Buffer, missingIndex, missingCount);

  if (missingCount)
    return make_error<JITLinkError>(Twine("COFF directive parsing failed: ") +
                                    Result.getArgString(missingIndex) +
                                    " missing argument");
  LLVM_DEBUG({
    for (auto *arg : Result.filtered(COFF_OPT_UNKNOWN))
      dbgs() << "Unknown coff option argument: " << arg->getAsString(Result)
             << "\n";
  });
  return std::move(Result);
}
