/*
 * kmp_traits.cpp -- Handle OpenMP context traits
 *
 * OpenMP 6.0 specifies the following trait sets:
 * - construct
 * - device
 * - target device
 * - implementation
 * - extension
 * - dynamic
 * Currently, the implementation in this file supports traits from the (target)
 * device and implementation trait sets that are relevant for implementing the
 * OMP_DEFAULT_DEVICE and OMP_AVAILABLE_DEVICES environment variables.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kmp_traits.h"
#include "kmp_i18n.h"

using namespace kmp_traits;

// OpenMP trait grammar (in EBNF), currently used for parsing the
// OMP_DEFAULT_DEVICE/OMP_AVAILABLE_DEVICES environment variables
//
// Notes about the grammar:
// - Device traits are going to be translated into device numbers (aka integers)
// later in the runtime. The parser handles device numbers as device traits that
// have already been translated.
// - "*" is also not a trait, strictly speaking. But it's also supported by the
// parser and converted into a "match any" wildcard trait.
// - OpenMP 6.0 explicitly excludes "&&" and "||" from appearing in the same
// grouping level.
// - This grammar currently only supports plain integers for array subsripts /
// sections, no expressions.
// - TODO:
//   - Add support for more traits
//
// TODOs regarding the implementation (not the grammar):
// - Implement array subscript/section parsing
// - Implement grammar TODOs after they have been incorporated into the grammar
//
// list = [clause {',' clause}]
// clause =
//       device_number
//     | "*" [index_expr]
//     | trait_expr_group
//     | trait_expr index_expr
// device_number = ["-"] integer0
// trait_expr_group =
//       trait_expr
//     | trait_expr {"&&" trait_expr}
//     | trait_expr {"||" trait_expr}
// trait_expr = ["!"] (trait | trait_expr_group_paren)
// trait_expr_group_paren = "(" trait_expr_group ")"
// trait =
//       "uid" "(" uid_value ")"
// uid_value = (letter | digit0 | symbol) {letter | digit0 | symbol}
//
// index_expr = "[" integer0 "]" | "[" array_section "]"
// array_section =
//       lower_bound ":" length ":" stride
//     | lower_bound ":" length ":"
//     | lower_bound ":" length
//     | lower_bound "::" stride
//     | lower_bound "::"
//     | lower_bound ":"
//     | ":" length ":" stride
//     | ":" length ":"
//     | ":" length
//     | "::" stride
//     | "::"
//     | ":"
// lower_bound = integer0
// length = integer0
// stride = integer
//
// integer0 = 0 | integer
// integer = digit {digit0}
//
// letter =
//       "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L"
//     | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X"
//     | "Y" | "Z" | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j"
//     | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v"
//     | "w" | "x" | "y" | "z"
// digit0 = "0" | digit
// digit = "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
// symbol = "-" | "_"

// A character that can appear in a word (uid keyword / uid_value / integer),
// i.e. a letter, a digit, or one of the symbols "-" / "_".
static bool is_word_char(char c) {
  return isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
}

static bool is_digit(char c) {
  return static_cast<bool>(isdigit(static_cast<unsigned char>(c)));
}

namespace lexer {

token kmp_lexer::lex() {
  scan.skip_space();

  const char *start = scan.begin();
  if (scan.empty())
    return {token_kind::END, kmp_str_ref(start, 0)};

  // Two-character operators.
  if (scan.consume_front("&&"))
    return {token_kind::AND, kmp_str_ref(start, 2)};
  if (scan.consume_front("||"))
    return {token_kind::OR, kmp_str_ref(start, 2)};

  // Word characters form a single WORD.
  kmp_str_ref word = scan.take_while(is_word_char);
  if (!word.empty()) {
    scan.drop_front(word.length());
    return {token_kind::WORD, word};
  }

  // Single-character tokens.
  token_kind kind;
  switch (*start) {
  case ',':
    kind = token_kind::COMMA;
    break;
  case '*':
    kind = token_kind::STAR;
    break;
  case '!':
    kind = token_kind::NOT;
    break;
  case '(':
    kind = token_kind::L_PAREN;
    break;
  case ')':
    kind = token_kind::R_PAREN;
    break;
  case '[':
    kind = token_kind::L_BRACKET;
    break;
  case ']':
    kind = token_kind::R_BRACKET;
    break;
  case ':':
    kind = token_kind::COLON;
    break;
  default:
    kind = token_kind::UNKNOWN;
    break;
  }
  scan.drop_front(1);
  return {kind, kmp_str_ref(start, 1)};
}

} // namespace lexer

namespace parser {

constexpr int MAX_RECURSION_DEPTH = 64;

using namespace kmp_traits;
using namespace lexer;

// Check whether a token is a WORD whose text equals the given keyword.
static bool word_is(const token &tok, kmp_str_ref keyword) {
  kmp_str_ref text = tok.text;
  return tok.kind == token_kind::WORD && text.consume_front(keyword) &&
         text.empty();
}

// Check whether a token is a WORD that has the shape of a device number, i.e.
// device_number = ["-"] integer0 (an optional "-" followed by digits only).
// The lexer emits every word-character run as a WORD, so it is the parser that
// tells a device number apart from a name or uid_value (which may also contain
// "-" and digits).
static bool word_is_number(const token &tok) {
  if (tok.kind != token_kind::WORD)
    return false;
  kmp_str_ref text = tok.text;
  text.consume_front("-");
  return !text.empty() && text.find_if_not(is_digit) == kmp_str_ref::npos;
}

// uid_value = (letter | digit0 | symbol) {letter | digit0 | symbol}
// Consumes and returns a uid value.
// An invalid uid value is a hard error.
static kmp_str_ref consume_uid_value(kmp_lexer &lex, const char *dbg_name) {
  const token &tok = lex.peek();
  if (tok.kind != token_kind::WORD)
    KMP_FATAL(TraitParserInvalidTraitValue, dbg_name, "uid", tok.text.copy());
  kmp_str_ref uid = tok.text;
  lex.next(); // consume the uid_value
  return uid;
}

// trait = "uid" "(" uid_value ")"
// (more traits will be added as needed in the future)
// Returns false without consuming anything if the next token is not a
// recognized trait name.
// Once a trait name is consumed we are committed to a trait expression, so a
// missing "(" or ")" or an invalid trait value is a hard error.
static bool consume_trait(kmp_trait_expr_single &expr, kmp_lexer &lex,
                          const char *dbg_name) {
  if (!word_is(lex.peek(), "uid"))
    return false;
  // Add more traits as needed in the future.

  lex.next(); // consume trait name
  if (lex.peek().kind != token_kind::L_PAREN)
    KMP_FATAL(TraitParserError, dbg_name, "expected '(' after trait name");
  lex.next(); // consume "("
  kmp_str_ref uid = consume_uid_value(lex, dbg_name);
  if (lex.peek().kind != token_kind::R_PAREN)
    KMP_FATAL(TraitParserError, dbg_name, "expected ')' after trait value");
  lex.next(); // consume ")"
  expr.set_trait(new kmp_uid_trait(uid));
  return true;
}

// forward declaration
static bool consume_trait_expr_group(kmp_trait_expr_group &group,
                                     kmp_lexer &lex, int max_recursion,
                                     const char *dbg_name);

// trait_expr_group_paren = "(" trait_expr_group ")"
// Returns false without consuming anything if the next token is not "(", so the
// caller can try other alternatives.
// Once "(" is consumed we are committed to a parenthesized group, so a missing
// group or ")" is a hard error.
static bool consume_trait_expr_group_paren(kmp_trait_expr_group &group,
                                           bool negated, kmp_lexer &lex,
                                           int max_recursion,
                                           const char *dbg_name) {
  if (lex.peek().kind != token_kind::L_PAREN)
    return false;
  group.set_negated(negated);
  lex.next(); // consume "("
  if (!consume_trait_expr_group(group, lex, max_recursion, dbg_name))
    KMP_FATAL(TraitParserError, dbg_name,
              "expected trait expression after '('");
  if (lex.peek().kind != token_kind::R_PAREN)
    KMP_FATAL(TraitParserError, dbg_name,
              "expected ')' after trait expression group");
  lex.next(); // consume ")"
  return true;
}

// trait_expr = ["!"] (trait | trait_expr_group_paren)
// Returns false without consuming anything if neither a parenthesized group nor
// a single trait can be consumed.
// Once an optional leading "!" has been consumed we are committed to a trait
// expression, so a missing trait or parenthesized group is a hard error.
static bool consume_trait_expr(kmp_trait_expr *&expr, kmp_lexer &lex,
                               int max_recursion, const char *dbg_name) {
  if (max_recursion-- <= 0)
    KMP_FATAL(TraitParserMaxRecursion, dbg_name, MAX_RECURSION_DEPTH);

  // Consume the optional leading "!"; it applies to whatever follows.
  bool negated = lex.peek().kind == token_kind::NOT;
  if (negated)
    lex.next();

  // Try a parenthesized group (starts with "(") ...
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  if (consume_trait_expr_group_paren(*group, negated, lex, max_recursion,
                                     dbg_name)) {
    expr = group;
    return true;
  }
  delete group;

  // ... otherwise it must be a single trait.
  kmp_trait_expr_single *single = new kmp_trait_expr_single();
  single->set_negated(negated);
  if (consume_trait(*single, lex, dbg_name)) {
    expr = single;
    return true;
  }
  delete single;

  // A leading "!" has already committed us to a trait expression, so its
  // absence is an error; without it, nothing was consumed and the caller can
  // recover/report.
  if (negated)
    KMP_FATAL(TraitParserError, dbg_name,
              "expected trait expression after '!'");
  return false;
}

// trait_expr_group =
//       trait_expr
//     | trait_expr {"&&" trait_expr}
//     | trait_expr {"||" trait_expr}
// Returns false without consuming anything if no trait expression can be
// consumed.
// Any other missing or invalid tokens are a hard error.
static bool consume_trait_expr_group(kmp_trait_expr_group &group,
                                     kmp_lexer &lex, int max_recursion,
                                     const char *dbg_name) {
  if (max_recursion-- <= 0)
    KMP_FATAL(TraitParserMaxRecursion, dbg_name, MAX_RECURSION_DEPTH);

  kmp_trait_expr *expr = nullptr;
  if (!consume_trait_expr(expr, lex, max_recursion, dbg_name))
    return false;
  group.add_expr(expr);

  token_kind op;
  if (lex.peek().kind == token_kind::OR) {
    group.set_group_type(kmp_trait_expr_group::OR);
    op = token_kind::OR;
  } else if (lex.peek().kind == token_kind::AND) {
    group.set_group_type(kmp_trait_expr_group::AND);
    op = token_kind::AND;
  } else {
    return true; // single trait expression, no operator
  }
  lex.next(); // consume the operator

  // Having consumed an operator, we are committed: at least one more trait
  // expression must follow, so its absence is a hard error.
  while (true) {
    if (!consume_trait_expr(expr, lex, max_recursion, dbg_name))
      KMP_FATAL(TraitParserError, dbg_name,
                "expected trait expression after operator");
    group.add_expr(expr);
    if (lex.peek().kind != op)
      break;
    lex.next(); // consume the operator
  }

  return true;
}

// device_number = ["-"] integer0
// Returns false without consuming anything if the next token is not shaped like
// a device number. A WORD that has the shape of a device number is treated as a
// device number, so once it is recognized it is committed: a value that is not
// a valid device index (negative or too large to fit an int) is a hard error.
static bool consume_device_number(kmp_trait_clause &clause, kmp_lexer &lex,
                                  const char *dbg_name) {
  if (!word_is_number(lex.peek()))
    return false;
  kmp_str_ref number = lex.peek().text;
  int value;
  if (!number.consume_integer(value))
    KMP_FATAL(TraitParserError, dbg_name, "device number out of range");
  lex.next();
  clause.set_expr(new kmp_literal_trait(value));
  return true;
}

// clause =
//       device_number
//     | "*" [index_expr]
//     | trait_expr_group
//     | trait_expr index_expr
// Returns false without consuming anything if no clause can be consumed.
// Any other missing or invalid tokens are a hard error.
static bool consume_clause(kmp_trait_clause &clause, kmp_lexer &lex,
                           const char *dbg_name) {
  // Parse wildcard "trait"
  if (lex.peek().kind == token_kind::STAR) {
    lex.next();
    clause.set_expr(new kmp_wildcard_trait());
    return true;
  }

  // Parse a literal device number. A WORD that is not shaped like a number is
  // not a device number and starts a trait expression group instead.
  if (consume_device_number(clause, lex, dbg_name))
    return true;

  // Parse a trait expression group
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  if (consume_trait_expr_group(*group, lex, MAX_RECURSION_DEPTH, dbg_name)) {
    clause.set_expr(group);
    return true;
  }
  delete group;

  return false;
}

// list = [clause {',' clause}]
static void consume_list(kmp_trait_context &context, kmp_lexer &lex,
                         const char *dbg_name) {
  kmp_str_ref lex_pos = lex.remaining();

  while (lex.peek().kind != token_kind::END) {
    kmp_trait_clause *clause = new kmp_trait_clause();
    if (!consume_clause(*clause, lex, dbg_name)) {
      delete clause;
      KMP_FATAL(TraitParserFailed, dbg_name, lex_pos.copy());
    }
    context.add_clause(clause);

    lex_pos = lex.remaining();
    if (lex.peek().kind == token_kind::COMMA)
      lex.next();
    else if (lex.peek().kind != token_kind::END)
      KMP_FATAL(TraitParserFailed, dbg_name, lex_pos.copy());
  }
}

} // namespace parser

kmp_trait_context *kmp_trait_context::parse_from_spec(kmp_str_ref spec,
                                                      const char *dbg_name) {
  kmp_trait_context *context = new kmp_trait_context();
  lexer::kmp_lexer lex(spec);
  parser::consume_list(*context, lex, dbg_name);
  return context;
}
