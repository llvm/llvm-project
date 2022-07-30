//===--- COFFDirectiveParser.h - JITLink coff directive parser --*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_JITLINK_COFFDIRECTIVEPARSER_H
#define LLVM_EXECUTIONENGINE_JITLINK_COFFDIRECTIVEPARSER_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/StringSaver.h"

namespace llvm {
namespace jitlink {

enum {
  COFF_OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) COFF_OPT_##ID,
#include "COFFOptions.inc"
#undef OPTION
};

/// Parser for the MSVC specific preprocessor directives.
/// https://docs.microsoft.com/en-us/cpp/preprocessor/comment-c-cpp?view=msvc-160
class COFFDirectiveParser {
public:
  Expected<std::unique_ptr<opt::InputArgList>> parse(StringRef Str);

private:
  llvm::BumpPtrAllocator bAlloc;
  llvm::StringSaver saver{bAlloc};
};

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_COFFDIRECTIVEPARSER_H