//===-- DirectiveBase.td - Base directive definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the base definition file directives and clauses.
//
//===----------------------------------------------------------------------===//


// General information about the directive language.
class DirectiveLanguage {
  // Name of the directive language such as omp or acc.
  string name = ?;

  // The C++ namespace that code of this directive language should be placed
  // into. This namespace is nested in llvm namespace.
  //
  // By default, uses the name of the directive language as the only namespace.
  // To avoid placing in any namespace, use "". To specify nested namespaces,
  // use "::" as the delimiter, e.g., given "A::B", ops will be placed in
  // `namespace A { namespace B { <directives-clauses> } }`.
  string cppNamespace = name;

  // Optional prefix used for the generation of the enumerator in the Directive
  // enum.
  string directivePrefix = "";

  // Optional prefix used for the generation of the enumerator in the Clause
  // enum.
  string clausePrefix = "";

  // Make the enum values available in the namespace. This allows us to
  // write something like Enum_X if we have a `using namespace cppNamespace`.
  bit makeEnumAvailableInNamespace = 0;

  // Generate include and macro to enable LLVM BitmaskEnum.
  bit enableBitmaskEnumInNamespace = 0;

  // Header file included in the implementation code generated. Ususally the
  // output file of the declaration code generation. Can be left blank.
  string includeHeader = "";

  // EnumSet class name used for clauses to generated the allowed clauses map.
  string clauseEnumSetClass = "";

  // Class holding the clauses in the flang parse-tree.
  string flangClauseBaseClass = "";
}

// Information about values accepted by enum-like clauses
class ClauseVal<string n, int v, bit uv> {
  // Name of the clause value.
  string name = n;

  // Integer value of the clause.
  int value = v;

  // Can user specify this value?
  bit isUserValue = uv;

  // Set clause value used by default when unknown.
  bit isDefault = 0;
}

// Information about a specific clause.
class Clause<string c> {
  // Name of the clause.
  string name = c;

  // Define an alternative name return in get<LanguageName>ClauseName function.
  string alternativeName = "";

  // Optional class holding value of the clause in clang AST.
  string clangClass = "";

  // Optional class holding the clause in flang AST. If left blank, the class
  // is assumed to be the name of the clause with capitalized word and
  // underscores removed.
  // ex: async -> Async
  //     num_threads -> NumThreads
  string flangClass = "";

  // Optional class holding value of the clause in flang AST.
  string flangClassValue = "";

  // If set to 1, value is optional. Not optional by default.
  bit isValueOptional = 0;

  // Name of enum when there is a list of allowed clause values.
  string enumClauseValue = "";

  // List of allowed clause values
  list<ClauseVal> allowedClauseValues = [];
  // If set to 1, value class is part of a list. Single class by default.
  bit isValueList = 0;

  // Define a default value such as "*".
  string defaultValue = "";

  // Is clause implicit? If clause is set as implicit, the default kind will
  // be return in get<LanguageName>ClauseKind instead of their own kind.
  bit isImplicit = 0;

  // Set clause used by default when unknown. Function returning the kind
  // of enumeration will use this clause as the default.
  bit isDefault = 0;
}

// Hold information about clause validity by version.
class VersionedClause<Clause c, int min = 1, int max = 0x7FFFFFFF> {
  // Actual clause.
  Clause clause = c;

  // Mininum version number where this clause is valid.
  int minVersion = min;

  // Maximum version number where this clause is valid.
  int maxVersion = max;
}

// Information about a specific directive.
class Directive<string d> {
  // Name of the directive. Can be composite directive sepearted by whitespace.
  string name = d;

  // Define an alternative name return in get<LanguageName>DirectiveName
  // function.
  string alternativeName = "";

  // List of allowed clauses for the directive.
  list<VersionedClause> allowedClauses = [];

  // List of clauses that are allowed to appear only once.
  list<VersionedClause> allowedOnceClauses = [];

  // List of clauses that are allowed but mutually exclusive.
  list<VersionedClause> allowedExclusiveClauses = [];

  // List of clauses that are required.
  list<VersionedClause> requiredClauses = [];

  // Set directive used by default when unknown.
  bit isDefault = 0;
}
