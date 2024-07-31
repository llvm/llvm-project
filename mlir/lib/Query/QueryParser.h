//===--- QueryParser.h - ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H
#define MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H

#include "Matcher/Parser.h"
#include "mlir/Query/Query.h"
#include "mlir/Query/QuerySession.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/LineEditor/LineEditor.h"

namespace mlir::query {

class QuerySession;

class QueryParser {
public:
  // Parse line as a query and return a QueryRef representing the query, which
  // may be an InvalidQuery.
  static QueryRef parse(llvm::StringRef line, const QuerySession &qs);

  static std::vector<llvm::LineEditor::Completion>
  complete(llvm::StringRef line, size_t pos, const QuerySession &qs);

private:
  QueryParser(llvm::StringRef line, const QuerySession &qs)
      : line(line), completionPos(nullptr), qs(qs) {}

  llvm::StringRef lexWord();

  template <typename T>
  struct LexOrCompleteWord;

  QueryRef completeMatcherExpression();

  QueryRef endQuery(QueryRef queryRef);

  // Parse [begin, end) and returns a reference to the parsed query object,
  // which may be an InvalidQuery if a parse error occurs.
  QueryRef doParse();

  llvm::StringRef line;

  const char *completionPos;
  std::vector<llvm::LineEditor::Completion> completions;

  const QuerySession &qs;
};

} // namespace mlir::query

#endif // MLIR_TOOLS_MLIRQUERY_QUERYPARSER_H
