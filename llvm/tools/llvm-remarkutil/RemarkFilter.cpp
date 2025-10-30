//===- RemarkFilter.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic tool to filter remarks
//
//===----------------------------------------------------------------------===//

#include "RemarkUtilHelpers.h"
#include "RemarkUtilRegistry.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"

using namespace llvm;
using namespace remarks;
using namespace llvm::remarkutil;

// Note: Avoid using the identifier "filter" in this file, as it is prone to
// namespace collision with headers that might get included e.g.
// curses.h.

static cl::SubCommand FilterSub("filter",
                                "Filter remarks based on specified criteria.");

INPUT_FORMAT_COMMAND_LINE_OPTIONS(FilterSub)
OUTPUT_FORMAT_COMMAND_LINE_OPTIONS(FilterSub)
INPUT_OUTPUT_COMMAND_LINE_OPTIONS(FilterSub)
REMARK_FILTER_COMMAND_LINE_OPTIONS(FilterSub)

REMARK_FILTER_SETUP_FUNC()

static Error tryFilter() {
  auto MaybeFilter = getRemarkFilters();
  if (!MaybeFilter)
    return MaybeFilter.takeError();
  Filters &Filter = *MaybeFilter;

  auto MaybeBuf = getInputMemoryBuffer(InputFileName);
  if (!MaybeBuf)
    return MaybeBuf.takeError();
  auto MaybeParser = createRemarkParser(InputFormat, (*MaybeBuf)->getBuffer());
  if (!MaybeParser)
    return MaybeParser.takeError();
  auto &Parser = **MaybeParser;

  Format SerializerFormat = OutputFormat;
  if (SerializerFormat == Format::Auto) {
    SerializerFormat = Parser.ParserFormat;
    if (OutputFileName.empty() || OutputFileName == "-")
      SerializerFormat = Format::YAML;
  }

  auto MaybeOF = getOutputFileForRemarks(OutputFileName, SerializerFormat);
  if (!MaybeOF)
    return MaybeOF.takeError();
  auto OF = std::move(*MaybeOF);

  auto MaybeSerializer = createRemarkSerializer(SerializerFormat, OF->os());
  if (!MaybeSerializer)
    return MaybeSerializer.takeError();
  auto &Serializer = **MaybeSerializer;

  auto MaybeRemark = Parser.next();
  for (; MaybeRemark; MaybeRemark = Parser.next()) {
    Remark &Remark = **MaybeRemark;
    if (!Filter.filterRemark(Remark))
      continue;
    Serializer.emit(Remark);
  }

  auto E = MaybeRemark.takeError();
  if (!E.isA<EndOfFileError>())
    return E;
  consumeError(std::move(E));
  OF->keep();
  return Error::success();
}

static CommandRegistration FilterReg(&FilterSub, tryFilter);
