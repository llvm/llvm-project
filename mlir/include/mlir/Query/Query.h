//===--- Query.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_QUERY_H
#define MLIR_TOOLS_MLIRQUERY_QUERY_H

#include "Matcher/VariantValue.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/LineEditor/LineEditor.h"
#include <string>

namespace mlir::query {

enum class QueryKind { Invalid, NoOp, Help, Match, Quit };

class QuerySession;

struct Query : llvm::RefCountedBase<Query> {
  Query(QueryKind kind) : kind(kind) {}
  virtual ~Query();

  // Perform the query on qs and print output to os.
  virtual mlir::LogicalResult run(llvm::raw_ostream &os,
                                  QuerySession &qs) const = 0;

  llvm::StringRef remainingContent;
  const QueryKind kind;
};

typedef llvm::IntrusiveRefCntPtr<Query> QueryRef;

QueryRef parse(llvm::StringRef line, const QuerySession &qs);

std::vector<llvm::LineEditor::Completion>
complete(llvm::StringRef line, size_t pos, const QuerySession &qs);

// Any query which resulted in a parse error. The error message is in ErrStr.
struct InvalidQuery : Query {
  InvalidQuery(const llvm::Twine &errStr)
      : Query(QueryKind::Invalid), errStr(errStr.str()) {}
  mlir::LogicalResult run(llvm::raw_ostream &os,
                          QuerySession &qs) const override;

  std::string errStr;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Invalid;
  }
};

// No-op query (i.e. a blank line).
struct NoOpQuery : Query {
  NoOpQuery() : Query(QueryKind::NoOp) {}
  mlir::LogicalResult run(llvm::raw_ostream &os,
                          QuerySession &qs) const override;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::NoOp;
  }
};

// Query for "help".
struct HelpQuery : Query {
  HelpQuery() : Query(QueryKind::Help) {}
  mlir::LogicalResult run(llvm::raw_ostream &os,
                          QuerySession &qs) const override;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Help;
  }
};

// Query for "quit".
struct QuitQuery : Query {
  QuitQuery() : Query(QueryKind::Quit) {}
  mlir::LogicalResult run(llvm::raw_ostream &os,
                          QuerySession &qs) const override;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Quit;
  }
};

// Query for "match MATCHER".
struct MatchQuery : Query {
  MatchQuery(llvm::StringRef source, const matcher::DynMatcher &matcher)
      : Query(QueryKind::Match), matcher(matcher), source(source) {}
  mlir::LogicalResult run(llvm::raw_ostream &os,
                          QuerySession &qs) const override;

  const matcher::DynMatcher matcher;

  llvm::StringRef source;

  static bool classof(const Query *query) {
    return query->kind == QueryKind::Match;
  }
};

} // namespace mlir::query

#endif
