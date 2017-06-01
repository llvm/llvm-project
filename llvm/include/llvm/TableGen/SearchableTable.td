//===- SearchableTable.td ----------------------------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the key top-level classes needed to produce a reasonably
// generic table that can be binary-searched via int and string entries.
//
// Each table must instantiate "Mappingkind", listing the fields that should be
// included and fields that shoould be searchable. Only two kinds of fields are
// searchable at the moment: "strings" (which are compared case-insensitively),
// and "bits".
//
// For each "MappingKind" the generated header will create GET_MAPPINGKIND_DECL
// and GET_MAPPINGKIND_IMPL guards.
//
// Inside the DECL guard will be a set of function declarations:
// "lookup{InstanceClass}By{SearchableField}", returning "const {InstanceClass}
// *" and accepting either a StringRef or a uintN_t. Additionally, if
// EnumNameField is still defined, there will be an "enum {InstanceClass}Values"
// allowing C++ code to reference either the primary data table's entries (if
// EnumValueField is not defined) or some other field (e.g. encoding) if it is.
//
// Inside the IMPL guard will be a primary data table "{InstanceClass}sList" and
// as many searchable indexes as requested
// ("{InstanceClass}sBy{SearchableField}"). Additionally implementations of the
// lookup function will be provided.
//
// See AArch64SystemOperands.td and its generated header for example uses.
//
//===----------------------------------------------------------------------===//

class SearchableTable {
  list<string> SearchableFields;
  string EnumNameField = "Name";
  string EnumValueField;
}
