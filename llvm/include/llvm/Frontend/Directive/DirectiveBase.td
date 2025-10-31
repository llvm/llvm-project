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
  bit makeEnumAvailableInNamespace = false;

  // Generate include and macro to enable LLVM BitmaskEnum.
  bit enableBitmaskEnumInNamespace = false;

  // Header file included in the implementation code generated. Ususally the
  // output file of the declaration code generation. Can be left blank.
  string includeHeader = "";

  // EnumSet class name used for clauses to generated the allowed clauses map.
  string clauseEnumSetClass = "";

  // Class holding the clauses in the flang parse-tree.
  string flangClauseBaseClass = "";
}

// Base class for versioned entities.
class Versioned<int min = 0, int max = 0x7FFFFFFF> {
  // Mininum version number where this object is valid.
  int minVersion = min;

  // Maximum version number where this object is valid.
  int maxVersion = max;
}

class Spelling<string s, int min = 0, int max = 0x7FFFFFFF>
    : Versioned<min, max> {
  string spelling = s;
}

// Some clauses take an argument from a predefined list of allowed keyword
// values. For example, assume a clause "someclause" with an argument from
// the list "foo", "bar", "baz". In the user source code this would look
// like "someclause(foo)", whereas in the compiler the values would be
// represented as
// enum someclause.enumClauseValue {
//   Xyz_foo = v_foo,
//   Xyz_bar = v_bar,
//   Xyz_baz = v_baz,
// }
// The "Xyz_..." are the _record_ names of EnumVal's:
// def Xyz_foo = EnumVal<"foo", v_foo>;
// def Xyz_bar = EnumVal<"bar", v_bar>;
// def Xyz_baz = EnumVal<"baz", v_baz>;
//
class EnumVal<string n, int v, bit uv> {
  // Spelling of the value.
  string name = n;

  // Integer value of the clause.
  int value = v;

  // Can user specify this value?
  bit isUserValue = uv;

  // Set clause value used by default when unknown.
  bit isDefault = false;
}

// Information about a specific clause.
class Clause<list<Spelling> ss> {
  // Spellings of the clause.
  list<Spelling> spellings = ss;

  // Optional class holding value of the clause in clang AST.
  string clangClass = "";

  // Optional class holding value of the clause in flang AST.
  string flangClass = "";

  // If set to true, don't emit flang Unparser.
  bit skipFlangUnparser = false;

  // If set to true, value is optional. Not optional by default.
  bit isValueOptional = false;

  // Name of enum when there is a list of allowed clause values.
  string enumClauseValue = "";

  // List of allowed clause values
  list<EnumVal> allowedClauseValues = [];

  // If set to true, value class is part of a list. Single class by default.
  bit isValueList = false;

  // Define a default value such as "*".
  string defaultValue = "";

  // Is clause implicit? If clause is set as implicit, the default kind will
  // be return in get<LanguageName>ClauseKind instead of their own kind.
  bit isImplicit = false;

  // Set clause used by default when unknown. Function returning the kind
  // of enumeration will use this clause as the default.
  bit isDefault = false;

  // Prefix before the actual value. Used in the parser generation.
  // `clause(prefix: value)`
  string prefix = "";

  // Set the prefix as optional.
  // `clause([prefix]: value)`
  bit isPrefixOptional = true;

  // When necessary because it isn't covered by rules, the name used in the
  // clause in the clang::OpenACCClauseKind enum.
  string clangAccSpelling = "";
}

// Hold information about clause validity by version.
class VersionedClause<Clause c, int min = 1, int max = 0x7FFFFFFF>
    : Versioned<min, max> {
  Clause clause = c;
}

// Kinds of directive associations.
class Association<string n> {
  string name = n;  // Name of the enum value in enum class Association.
}
// All of the AS_Xyz names are recognized by TableGen in order to calculate
// the association in the AS_FromLeaves case.
def AS_None : Association<"None"> {}              // No association
def AS_Block : Association<"Block"> {}            // Block (incl. single
                                                  // statement)
def AS_Declaration : Association<"Declaration"> {}   // Declaration
def AS_Delimited : Association<"Delimited"> {}    // Region delimited with
                                                  // begin/end
def AS_Loop : Association<"Loop"> {}              // Loop
def AS_Separating : Association<"Separating"> {}  // Separates parts of a
                                                  // construct

def AS_FromLeaves : Association<"FromLeaves"> {}    // See below
// AS_FromLeaves can be used for combined/composite directives, and the actual
// association will be computed based on associations of the leaf constructs:
//   (x + y) + z = x + (y + z)
//   x + y = y + x
//   x + x = x
//   AS_None + x = x
//   AS_Block + AS_Loop = AS_Loop
// Other combinations are not allowed.
// This association is not valid for leaf constructs.
// The name "AS_FromLeaves" is recognized by TableGen, and there is no enum
// generated for it.

// Kinds of directive categories.
class Category<string n> {
  string name = n;  // Name of the enum value in enum class Category.
}

def CA_Declarative: Category<"Declarative"> {}
def CA_Executable: Category<"Executable"> {}
def CA_Informational: Category<"Informational"> {}
def CA_Meta: Category<"Meta"> {}
def CA_Subsidiary: Category<"Subsidiary"> {}
def CA_Utility: Category<"Utility"> {}

class SourceLanguage<string n> {
  string name = n;  // Name of the enum value in enum class Association.
}

// The C language also implies C++ until there is a reason to add C++
// separately.
def L_C : SourceLanguage<"C"> {}
def L_Fortran : SourceLanguage<"Fortran"> {}

// Information about a specific directive.
class Directive<list<Spelling> ss> {
  // Spellings of the directive.
  list<Spelling> spellings = ss;

  // Clauses cannot appear twice in the three allowed lists below. Also, since
  // required implies allowed, the same clause cannot appear in both the
  // allowedClauses and requiredClauses lists.

  // List of allowed clauses for the directive.
  list<VersionedClause>  allowedClauses = [];

  // List of clauses that are allowed to appear only once.
  list<VersionedClause> allowedOnceClauses = [];

  // List of clauses that are allowed but mutually exclusive.
  list<VersionedClause> allowedExclusiveClauses = [];

  // List of clauses that are required.
  list<VersionedClause> requiredClauses = [];

  // List of leaf constituent directives in the order in which they appear
  // in the combined/composite directive.
  list<Directive> leafConstructs = [];

  // Set directive used by default when unknown.
  bit isDefault = false;

  // What the directive is associated with.
  Association association = AS_FromLeaves;

  // The category of the directive.
  Category category = ?;

  // The languages that allow this directive. Default: all languages.
  list<SourceLanguage> languages = [L_C, L_Fortran];
}
