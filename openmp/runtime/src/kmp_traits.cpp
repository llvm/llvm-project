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
// trait_expr =
//       trait_expr_single
//     | trait_expr_group_paren
// trait_expr_single = ["!"] trait
// trait_expr_group_paren = ["!"] "(" trait_expr_group ")"
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

namespace parser {

#define MAX_RECURSION_DEPTH 64

using namespace kmp_traits;

static kmp_str_ref consume_uid_value(kmp_str_ref &scan, const char *dbg_name) {
  scan.skip_space();
  kmp_str_ref uid = scan.take_while([](char c) {
    return isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
  });
  scan.drop_front(uid.length());
  if (uid.empty() || !scan.consume_front(")"))
    KMP_FATAL(TraitParserInvalidUID, dbg_name, uid.copy());
  return uid;
}

static bool consume_trait(kmp_trait_expr_single &expr, kmp_str_ref &scan,
                          const char *dbg_name) {
  scan.skip_space();
  if (!scan.consume_front("uid("))
    return false;
  kmp_str_ref uid = consume_uid_value(scan, dbg_name);
  expr.set_trait(new kmp_uid_trait(uid));
  return true;
}

static bool consume_trait_expr_single(kmp_trait_expr_single &expr,
                                      kmp_str_ref &scan, const char *dbg_name) {
  kmp_str_ref orig_scan = scan;

  scan.skip_space();
  if (scan.consume_front("!"))
    expr.set_negated();
  if (consume_trait(expr, scan, dbg_name))
    return true;
  scan = orig_scan;
  return false;
}

// forward declaration
static bool consume_trait_expr_group(kmp_trait_expr_group &group,
                                     kmp_str_ref &scan, int max_recursion,
                                     const char *dbg_name);

static bool consume_trait_expr_group_paren(kmp_trait_expr_group &group,
                                           kmp_str_ref &scan, int max_recursion,
                                           const char *dbg_name) {
  if (max_recursion-- <= 0)
    KMP_FATAL(TraitParserMaxRecursion, dbg_name, MAX_RECURSION_DEPTH);
  kmp_str_ref orig_scan = scan;

  scan.skip_space();
  if (scan.consume_front("!"))
    group.set_negated();

  scan.skip_space();
  if (!scan.consume_front("(") ||
      !consume_trait_expr_group(group, scan, max_recursion, dbg_name)) {
    scan = orig_scan;
    return false;
  }

  scan.skip_space();
  if (!scan.consume_front(")")) {
    scan = orig_scan;
    return false;
  }
  return true;
}

static bool consume_trait_expr(kmp_trait_expr *&expr, kmp_str_ref &scan,
                               int max_recursion, const char *dbg_name) {
  if (max_recursion-- <= 0)
    KMP_FATAL(TraitParserMaxRecursion, dbg_name, MAX_RECURSION_DEPTH);

  // Parse a single trait expression
  kmp_trait_expr_single *single_expr = new kmp_trait_expr_single();
  if (consume_trait_expr_single(*single_expr, scan, dbg_name)) {
    expr = single_expr;
    return true;
  }
  delete single_expr;

  // Parse a parenthesized group trait expression
  kmp_trait_expr_group *group_expr = new kmp_trait_expr_group();
  if (consume_trait_expr_group_paren(*group_expr, scan, max_recursion,
                                     dbg_name)) {
    expr = group_expr;
    return true;
  }
  delete group_expr;

  return false;
}

static bool consume_trait_expr_group(kmp_trait_expr_group &group,
                                     kmp_str_ref &scan, int max_recursion,
                                     const char *dbg_name) {
  if (max_recursion-- <= 0)
    KMP_FATAL(TraitParserMaxRecursion, dbg_name, MAX_RECURSION_DEPTH);

  kmp_trait_expr *expr = nullptr;
  if (!consume_trait_expr(expr, scan, max_recursion, dbg_name))
    return false;

  group.add_expr(expr);
  const char *op = nullptr;

  scan.skip_space();
  if (scan.consume_front("||")) {
    group.set_group_type(kmp_trait_expr_group::OR);
    op = "||";
  } else if (scan.consume_front("&&")) {
    group.set_group_type(kmp_trait_expr_group::AND);
    op = "&&";
  } else {
    return true; // single trait expression, no group
  }

  // Now that we got an operator, we need at least one more trait expr.
  do {
    if (!consume_trait_expr(expr, scan, max_recursion, dbg_name))
      return false;
    group.add_expr(expr);
    scan.skip_space();
  } while (scan.consume_front(op));

  return true;
}

static bool consume_clause(kmp_trait_clause &clause, kmp_str_ref &scan,
                           const char *dbg_name) {
  kmp_str_ref orig_scan = scan;
  scan.skip_space();

  // Parse wildcard "trait"
  if (scan.consume_front("*")) {
    clause.set_expr(new kmp_wildcard_trait());
    return true;
  }

  // Parse a literal device number
  int value;
  if (scan.consume_integer(value)) {
    clause.set_expr(new kmp_literal_trait(value));
    return true;
  }

  // Parse a trait expression group
  kmp_trait_expr_group *group = new kmp_trait_expr_group();
  if (consume_trait_expr_group(*group, scan, MAX_RECURSION_DEPTH, dbg_name)) {
    clause.set_expr(group);
    return true;
  }
  delete group;

  scan = orig_scan;
  return false;
}

static bool consume_list(kmp_trait_context &context, kmp_str_ref &scan,
                         const char *dbg_name) {
  kmp_str_ref orig_scan = scan;
  scan.skip_space();

  while (!scan.empty()) {
    kmp_trait_clause *clause = new kmp_trait_clause();
    if (!consume_clause(*clause, scan, dbg_name)) {
      delete clause;
      scan = orig_scan;
      return false;
    }
    context.add_clause(clause);
    orig_scan = scan;

    scan.skip_space();
    if (!scan.consume_front(",") && !scan.empty()) {
      scan = orig_scan;
      return false;
    }
  }

  return true;
}

} // namespace parser

kmp_trait_context *kmp_trait_context::parse_from_spec(kmp_str_ref spec,
                                                      const char *dbg_name) {
  kmp_trait_context *context = new kmp_trait_context();
  if (!parser::consume_list(*context, spec, dbg_name))
    KMP_FATAL(TraitParserFailed, dbg_name, spec.copy());
  return context;
}

int kmp_trait_context::parse_single_device(kmp_str_ref spec,
                                           int device_num_limit,
                                           const char *dbg_name) {
  int device_num;
  spec.skip_space();
  if (!spec.consume_integer(device_num))
    KMP_FATAL(TraitParserFailed, dbg_name, spec.copy());
  if (device_num > device_num_limit)
    KMP_FATAL(TraitParserValueTooLarge, dbg_name, device_num, device_num_limit);
  return device_num;
}
