//===---- QueryParser.cpp - mlir-query command parser ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::query {

// Lex any amount of whitespace followed by a "word" (any sequence of
// non-whitespace characters) from the start of region [begin,end).  If no word
// is found before end, return StringRef(). begin is adjusted to exclude the
// lexed region.
llvm::StringRef QueryParser::lexWord() {
  // Don't trim newlines.
  line = line.ltrim(" \t\v\f\r");

  if (line.empty())
    // Even though the line is empty, it contains a pointer and
    // a (zero) length. The pointer is used in the LexOrCompleteWord
    // code completion.
    return line;

  llvm::StringRef word;
  if (line.front() == '#') {
    word = line.substr(0, 1);
  } else {
    word = line.take_until([](char c) {
      // Don't trim newlines.
      return llvm::StringRef(" \t\v\f\r").contains(c);
    });
  }

  line = line.drop_front(word.size());
  return word;
}

// This is the StringSwitch-alike used by LexOrCompleteWord below. See that
// function for details.
template <typename T>
struct QueryParser::LexOrCompleteWord {
  llvm::StringRef word;
  llvm::StringSwitch<T> stringSwitch;

  QueryParser *queryParser;
  // Set to the completion point offset in word, or StringRef::npos if
  // completion point not in word.
  size_t wordCompletionPos;

  // Lexes a word and stores it in word. Returns a LexOrCompleteword<T> object
  // that can be used like a llvm::StringSwitch<T>, but adds cases as possible
  // completions if the lexed word contains the completion point.
  LexOrCompleteWord(QueryParser *queryParser, llvm::StringRef &outWord)
      : word(queryParser->lexWord()), stringSwitch(word),
        queryParser(queryParser), wordCompletionPos(llvm::StringRef::npos) {
    outWord = word;
    if (queryParser->completionPos &&
        queryParser->completionPos <= word.data() + word.size()) {
      if (queryParser->completionPos < word.data())
        wordCompletionPos = 0;
      else
        wordCompletionPos = queryParser->completionPos - word.data();
    }
  }

  LexOrCompleteWord &Case(llvm::StringLiteral caseStr, const T &value,
                          bool isCompletion = true) {

    if (wordCompletionPos == llvm::StringRef::npos)
      stringSwitch.Case(caseStr, value);
    else if (!caseStr.empty() && isCompletion &&
             wordCompletionPos <= caseStr.size() &&
             caseStr.substr(0, wordCompletionPos) ==
                 word.substr(0, wordCompletionPos)) {

      queryParser->completions.emplace_back(
          (caseStr.substr(wordCompletionPos) + " ").str(),
          std::string(caseStr));
    }
    return *this;
  }

  T Default(T value) { return stringSwitch.Default(value); }
};

QueryRef QueryParser::endQuery(QueryRef queryRef) {
  llvm::StringRef extra = line;
  llvm::StringRef extraTrimmed = extra.ltrim(" \t\v\f\r");

  if (extraTrimmed.starts_with('\n') || extraTrimmed.starts_with("\r\n"))
    queryRef->remainingContent = extra;
  else {
    llvm::StringRef trailingWord = lexWord();
    if (trailingWord.starts_with('#')) {
      line = line.drop_until([](char c) { return c == '\n'; });
      line = line.drop_while([](char c) { return c == '\n'; });
      return endQuery(queryRef);
    }
    if (!trailingWord.empty()) {
      return new InvalidQuery("unexpected extra input: '" + extra + "'");
    }
  }
  return queryRef;
}

namespace {

enum class ParsedQueryKind {
  Invalid,
  Comment,
  NoOp,
  Help,
  Match,
  Quit,
};

QueryRef
makeInvalidQueryFromDiagnostics(const matcher::internal::Diagnostics &diag) {
  std::string errStr;
  llvm::raw_string_ostream os(errStr);
  diag.print(os);
  return new InvalidQuery(errStr);
}
} // namespace

QueryRef QueryParser::completeMatcherExpression() {
  std::vector<matcher::MatcherCompletion> comps =
      matcher::internal::Parser::completeExpression(
          line, completionPos - line.begin(), qs.getRegistryData(),
          &qs.namedValues);
  for (const auto &comp : comps) {
    completions.emplace_back(comp.typedText, comp.matcherDecl);
  }
  return QueryRef();
}

QueryRef QueryParser::doParse() {

  llvm::StringRef commandStr;
  ParsedQueryKind qKind =
      LexOrCompleteWord<ParsedQueryKind>(this, commandStr)
          .Case("", ParsedQueryKind::NoOp)
          .Case("#", ParsedQueryKind::Comment, /*isCompletion=*/false)
          .Case("help", ParsedQueryKind::Help)
          .Case("m", ParsedQueryKind::Match, /*isCompletion=*/false)
          .Case("match", ParsedQueryKind::Match)
          .Case("q", ParsedQueryKind::Quit, /*IsCompletion=*/false)
          .Case("quit", ParsedQueryKind::Quit)
          .Default(ParsedQueryKind::Invalid);

  switch (qKind) {
  case ParsedQueryKind::Comment:
  case ParsedQueryKind::NoOp:
    line = line.drop_until([](char c) { return c == '\n'; });
    line = line.drop_while([](char c) { return c == '\n'; });
    if (line.empty())
      return new NoOpQuery;
    return doParse();

  case ParsedQueryKind::Help:
    return endQuery(new HelpQuery);

  case ParsedQueryKind::Quit:
    return endQuery(new QuitQuery);

  case ParsedQueryKind::Match: {
    if (completionPos) {
      return completeMatcherExpression();
    }

    matcher::internal::Diagnostics diag;
    auto matcherSource = line.ltrim();
    auto origMatcherSource = matcherSource;
    std::optional<matcher::DynMatcher> matcher =
        matcher::internal::Parser::parseMatcherExpression(
            matcherSource, qs.getRegistryData(), &qs.namedValues, &diag);
    if (!matcher) {
      return makeInvalidQueryFromDiagnostics(diag);
    }
    auto actualSource = origMatcherSource.slice(0, origMatcherSource.size() -
                                                       matcherSource.size());
    QueryRef query = new MatchQuery(actualSource, *matcher);
    query->remainingContent = matcherSource;
    return query;
  }

  case ParsedQueryKind::Invalid:
    return new InvalidQuery("unknown command: " + commandStr);
  }

  llvm_unreachable("Invalid query kind");
}

QueryRef QueryParser::parse(llvm::StringRef line, const QuerySession &qs) {
  return QueryParser(line, qs).doParse();
}

std::vector<llvm::LineEditor::Completion>
QueryParser::complete(llvm::StringRef line, size_t pos,
                      const QuerySession &qs) {
  QueryParser queryParser(line, qs);
  queryParser.completionPos = line.data() + pos;

  queryParser.doParse();
  return queryParser.completions;
}

} // namespace mlir::query
