//===- DirectiveEmitter.h - Directive Language Emitter ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DirectiveEmitter uses the descriptions of directives and clauses to construct
// common code declarations to be used in Frontends.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TABLEGEN_DIRECTIVEEMITTER_H
#define LLVM_TABLEGEN_DIRECTIVEEMITTER_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/Directive/Spelling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TableGen/Record.h"
#include <string>
#include <vector>

namespace llvm {

// Wrapper class that contains DirectiveLanguage's information defined in
// DirectiveBase.td and provides helper methods for accessing it.
class DirectiveLanguage {
public:
  explicit DirectiveLanguage(const RecordKeeper &Records) : Records(Records) {
    const auto &DirectiveLanguages = getDirectiveLanguages();
    Def = DirectiveLanguages[0];
  }

  StringRef getName() const { return Def->getValueAsString("name"); }

  StringRef getCppNamespace() const {
    return Def->getValueAsString("cppNamespace");
  }

  StringRef getDirectivePrefix() const {
    return Def->getValueAsString("directivePrefix");
  }

  StringRef getClausePrefix() const {
    return Def->getValueAsString("clausePrefix");
  }

  StringRef getClauseEnumSetClass() const {
    return Def->getValueAsString("clauseEnumSetClass");
  }

  StringRef getFlangClauseBaseClass() const {
    return Def->getValueAsString("flangClauseBaseClass");
  }

  bool hasMakeEnumAvailableInNamespace() const {
    return Def->getValueAsBit("makeEnumAvailableInNamespace");
  }

  bool hasEnableBitmaskEnumInNamespace() const {
    return Def->getValueAsBit("enableBitmaskEnumInNamespace");
  }

  ArrayRef<const Record *> getAssociations() const {
    return Records.getAllDerivedDefinitions("Association");
  }

  ArrayRef<const Record *> getCategories() const {
    return Records.getAllDerivedDefinitions("Category");
  }

  ArrayRef<const Record *> getSourceLanguages() const {
    return Records.getAllDerivedDefinitions("SourceLanguage");
  }

  ArrayRef<const Record *> getDirectives() const {
    return Records.getAllDerivedDefinitions("Directive");
  }

  ArrayRef<const Record *> getClauses() const {
    return Records.getAllDerivedDefinitions("Clause");
  }

  bool HasValidityErrors() const;

private:
  const Record *Def;
  const RecordKeeper &Records;

  ArrayRef<const Record *> getDirectiveLanguages() const {
    return Records.getAllDerivedDefinitions("DirectiveLanguage");
  }
};

class Versioned {
public:
  int getMinVersion(const Record *R) const {
    int64_t Min = R->getValueAsInt("minVersion");
    assert(llvm::isInt<IntWidth>(Min) && "Value out of range of 'int'");
    return Min;
  }

  int getMaxVersion(const Record *R) const {
    int64_t Max = R->getValueAsInt("maxVersion");
    assert(llvm::isInt<IntWidth>(Max) && "Value out of range of 'int'");
    return Max;
  }

private:
  constexpr static int IntWidth = 8 * sizeof(int);
};

class Spelling : public Versioned {
public:
  using Value = directive::Spelling;

  Spelling(const Record *Def) : Def(Def) {}

  StringRef getText() const { return Def->getValueAsString("spelling"); }
  llvm::directive::VersionRange getVersions() const {
    return llvm::directive::VersionRange{getMinVersion(Def),
                                         getMaxVersion(Def)};
  }

  Value get() const { return Value{getText(), getVersions()}; }

private:
  const Record *Def;
};

// Note: In all the classes below, allow implicit construction from Record *,
// to allow writing code like:
//  for (const Directive D : getDirectives()) {
//
//  instead of:
//
//  for (const Record *R : getDirectives()) {
//    Directive D(R);

// Base record class used for Directive and Clause class defined in
// DirectiveBase.td.
class BaseRecord {
public:
  BaseRecord(const Record *Def) : Def(Def) {}

  std::vector<Spelling::Value> getSpellings() const {
    std::vector<Spelling::Value> List;
    llvm::transform(Def->getValueAsListOfDefs("spellings"),
                    std::back_inserter(List),
                    [](const Record *R) { return Spelling(R).get(); });
    return List;
  }

  StringRef getSpellingForIdentifier() const {
    // From all spellings, pick the first one with the minimum version
    // (i.e. pick the first from all the oldest ones). This guarantees
    // that given several equivalent (in terms of versions) names, the
    // first one is used, e.g. given
    //   Clause<[Spelling<"foo">, Spelling<"bar">]> ...
    // "foo" will be the selected spelling.
    //
    // This is a suitable spelling for generating an identifier name,
    // since it will remain unchanged when any potential new spellings
    // are added.
    Spelling::Value Oldest{"not found", {/*Min=*/INT_MAX, 0}};
    for (auto V : getSpellings())
      if (V.Versions.Min < Oldest.Versions.Min)
        Oldest = V;
    return Oldest.Name;
  }

  // Returns the name of the directive formatted for output. Whitespace are
  // replaced with underscores.
  static std::string getSnakeName(StringRef Name) {
    std::string N = Name.str();
    llvm::replace(N, ' ', '_');
    return N;
  }

  // Take a string Name with sub-words separated with characters from Sep,
  // and return a string with each of the sub-words capitalized, and the
  // separators removed, e.g.
  //   Name = "some_directive^name", Sep = "_^"  ->  "SomeDirectiveName".
  static std::string getUpperCamelName(StringRef Name, StringRef Sep) {
    std::string Camel = Name.str();
    // Convert to uppercase
    bool Cap = true;
    llvm::transform(Camel, Camel.begin(), [&](unsigned char C) {
      if (Sep.contains(C)) {
        assert(!Cap && "No initial or repeated separators");
        Cap = true;
      } else if (Cap) {
        C = llvm::toUpper(C);
        Cap = false;
      }
      return C;
    });
    size_t Out = 0;
    // Remove separators
    for (size_t In = 0, End = Camel.size(); In != End; ++In) {
      unsigned char C = Camel[In];
      if (!Sep.contains(C))
        Camel[Out++] = C;
    }
    Camel.resize(Out);
    return Camel;
  }

  std::string getFormattedName() const {
    if (auto maybeName = Def->getValueAsOptionalString("name"))
      return getSnakeName(*maybeName);
    return getSnakeName(getSpellingForIdentifier());
  }

  bool isDefault() const { return Def->getValueAsBit("isDefault"); }

  // Returns the record name.
  StringRef getRecordName() const { return Def->getName(); }

  const Record *getRecord() const { return Def; }

protected:
  const Record *Def;
};

// Wrapper class that contains a Directive's information defined in
// DirectiveBase.td and provides helper methods for accessing it.
class Directive : public BaseRecord {
public:
  Directive(const Record *Def) : BaseRecord(Def) {}

  std::vector<const Record *> getAllowedClauses() const {
    return Def->getValueAsListOfDefs("allowedClauses");
  }

  std::vector<const Record *> getAllowedOnceClauses() const {
    return Def->getValueAsListOfDefs("allowedOnceClauses");
  }

  std::vector<const Record *> getAllowedExclusiveClauses() const {
    return Def->getValueAsListOfDefs("allowedExclusiveClauses");
  }

  std::vector<const Record *> getRequiredClauses() const {
    return Def->getValueAsListOfDefs("requiredClauses");
  }

  std::vector<const Record *> getLeafConstructs() const {
    return Def->getValueAsListOfDefs("leafConstructs");
  }

  const Record *getAssociation() const {
    return Def->getValueAsDef("association");
  }

  const Record *getCategory() const { return Def->getValueAsDef("category"); }

  std::vector<const Record *> getSourceLanguages() const {
    return Def->getValueAsListOfDefs("languages");
  }

  // Clang uses a different format for names of its directives enum.
  std::string getClangAccSpelling() const {
    StringRef Name = getSpellingForIdentifier();

    // Clang calls the 'unknown' value 'invalid'.
    if (Name == "unknown")
      return "Invalid";

    return BaseRecord::getUpperCamelName(Name, " _");
  }
};

// Wrapper class that contains Clause's information defined in DirectiveBase.td
// and provides helper methods for accessing it.
class Clause : public BaseRecord {
public:
  Clause(const Record *Def) : BaseRecord(Def) {}

  // Optional field.
  StringRef getClangClass() const {
    return Def->getValueAsString("clangClass");
  }

  // Optional field.
  StringRef getFlangClass() const {
    return Def->getValueAsString("flangClass");
  }

  // Get the formatted name for Flang parser class. The generic formatted class
  // name is constructed from the name were the first letter of each word is
  // captitalized and the underscores are removed.
  // ex: async -> Async
  //     num_threads -> NumThreads
  std::string getFormattedParserClassName() const {
    StringRef Name = getSpellingForIdentifier();
    return BaseRecord::getUpperCamelName(Name, "_");
  }

  // Clang uses a different format for names of its clause enum, which can be
  // overwritten with the `clangSpelling` value. So get the proper spelling
  // here.
  std::string getClangAccSpelling() const {
    if (StringRef ClangSpelling = Def->getValueAsString("clangAccSpelling");
        !ClangSpelling.empty())
      return ClangSpelling.str();

    StringRef Name = getSpellingForIdentifier();
    return BaseRecord::getUpperCamelName(Name, "_");
  }

  // Optional field.
  StringRef getEnumName() const {
    return Def->getValueAsString("enumClauseValue");
  }

  std::vector<const Record *> getClauseVals() const {
    return Def->getValueAsListOfDefs("allowedClauseValues");
  }

  bool skipFlangUnparser() const {
    return Def->getValueAsBit("skipFlangUnparser");
  }

  bool isValueOptional() const { return Def->getValueAsBit("isValueOptional"); }

  bool isValueList() const { return Def->getValueAsBit("isValueList"); }

  StringRef getDefaultValue() const {
    return Def->getValueAsString("defaultValue");
  }

  bool isImplicit() const { return Def->getValueAsBit("isImplicit"); }

  std::vector<StringRef> getAliases() const {
    return Def->getValueAsListOfStrings("aliases");
  }

  StringRef getPrefix() const { return Def->getValueAsString("prefix"); }

  bool isPrefixOptional() const {
    return Def->getValueAsBit("isPrefixOptional");
  }
};

// Wrapper class that contains VersionedClause's information defined in
// DirectiveBase.td and provides helper methods for accessing it.
class VersionedClause {
public:
  VersionedClause(const Record *Def) : Def(Def) {}

  // Return the specific clause record wrapped in the Clause class.
  Clause getClause() const { return Clause(Def->getValueAsDef("clause")); }

  int64_t getMinVersion() const { return Def->getValueAsInt("minVersion"); }

  int64_t getMaxVersion() const { return Def->getValueAsInt("maxVersion"); }

private:
  const Record *Def;
};

class EnumVal : public BaseRecord {
public:
  EnumVal(const Record *Def) : BaseRecord(Def) {}

  int getValue() const { return Def->getValueAsInt("value"); }

  bool isUserVisible() const { return Def->getValueAsBit("isUserValue"); }
};

} // namespace llvm

#endif // LLVM_TABLEGEN_DIRECTIVEEMITTER_H
