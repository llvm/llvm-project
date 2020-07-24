//===- extract.cpp - Input splitting utility ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Split input into multipe parts separated by regex '^(.|//)--- ' and extract
// the specified part.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include <string>

using namespace llvm;

static cl::OptionCategory cat("extract Options");

static cl::opt<std::string> part(cl::Positional, cl::desc("part"),
                                 cl::cat(cat));

static cl::opt<std::string> input(cl::Positional, cl::desc("filename"),
                                  cl::cat(cat));

static cl::opt<std::string> output("o", cl::desc("Output filename"),
                                   cl::value_desc("filename"), cl::init("-"),
                                   cl::cat(cat));

static cl::opt<bool> noLeadingLines("no-leading-lines",
                                    cl::desc("Don't preserve line numbers"),
                                    cl::cat(cat));

static StringRef toolName;

LLVM_ATTRIBUTE_NORETURN static void error(StringRef filename,
                                          const Twine &message) {
  if (filename.empty())
    WithColor::error(errs(), toolName) << message << '\n';
  else
    WithColor::error(errs(), toolName) << filename << ": " << message << '\n';
  exit(1);
}

static void handle(MemoryBuffer &inputBuf, StringRef input) {
  const char *partBegin = nullptr, *partEnd = nullptr;
  int numEmptyLines = 0;
  StringRef separator;
  for (line_iterator i(inputBuf, /*SkipBlanks=*/false, '\0'); !i.is_at_eof();) {
    StringRef line = *i++;
    size_t markerLen = line.startswith("//") ? 6 : 5;
    if (!(line.size() > markerLen &&
          line.substr(markerLen - 4).startswith("--- ")))
      continue;
    separator = line.substr(0, markerLen);
    StringRef cur = line.substr(markerLen);
    if (cur == part) {
      if (partBegin)
        error(input, "'" + separator + cur + "' occurs more than once");
      if (!noLeadingLines)
        numEmptyLines = i.line_number() - 1;
      if (i.is_at_eof())
        break;
      partBegin = i->data();
    } else if (partBegin && !partEnd) {
      partEnd = line.data();
    }
  }
  if (!partBegin)
    error(input, "'" + separator + part + "' was not found");
  if (!partEnd)
    partEnd = inputBuf.getBufferEnd();

  Expected<std::unique_ptr<FileOutputBuffer>> outputBuf =
      FileOutputBuffer::create(output, numEmptyLines + (partEnd - partBegin));
  if (!outputBuf)
    error(input, toString(outputBuf.takeError()));
  uint8_t *buf = (*outputBuf)->getBufferStart();

  // If --no-leading-lines is not specified, numEmptyLines is 0. Append newlines
  // so that the extracted part preserves line numbers.
  std::fill_n(buf, numEmptyLines, '\n');
  std::copy(partBegin, partEnd, buf + numEmptyLines);
  if (Error e = (*outputBuf)->commit())
    error(input, toString(std::move(e)));
}

int main(int argc, const char **argv) {
  toolName = sys::path::stem(argv[0]);
  cl::HideUnrelatedOptions({&cat});
  cl::ParseCommandLineOptions(
      argc, argv,
      "Split input into multiple parts separated by regex '^(.|//)--- ' and "
      "extract the part specified by '^(.|//)--- <part>'\n",
      nullptr,
      /*EnvVar=*/nullptr,
      /*LongOptionsUseDoubleDash=*/true);

  if (input.empty())
    error("", "input filename is not specified");
  ErrorOr<std::unique_ptr<MemoryBuffer>> bufferOrErr =
      MemoryBuffer::getFileOrSTDIN(input);
  if (std::error_code ec = bufferOrErr.getError())
    error(input, ec.message());
  handle(**bufferOrErr, input);
}
