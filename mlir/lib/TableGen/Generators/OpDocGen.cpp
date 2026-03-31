//===- OpDocGen.cpp - MLIR operation documentation generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDocGen uses the description of operations to generate documentation for the
// operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Generators/OpDocGen.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/Generators/DocGenUtilities.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

#include <set>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using mlir::tblgen::Operator;

//===----------------------------------------------------------------------===//
// Operation Documentation
//===----------------------------------------------------------------------===//

/// Emit the assembly format of an operation.
static void emitAssemblyFormat(StringRef opName, StringRef format,
                               raw_ostream &os) {
  if (format.empty())
    return;
  os << "\nSyntax:\n\n```\noperation ::= `" << opName << "` ";

  // Print the assembly format aligned.
  unsigned indent = strlen("operation ::= ");
  std::pair<StringRef, StringRef> split = format.split('\n');
  os << split.first.trim() << "\n";
  do {
    split = split.second.split('\n');
    StringRef formatChunk = split.first.trim();
    if (!formatChunk.empty())
      os.indent(indent) << formatChunk << "\n";
  } while (!split.second.empty());
  os << "```\n";
}

/// Place `text` between backticks so that the Markdown processor renders it as
/// inline code.
static std::string backticks(const std::string &text) {
  return '`' + text + '`';
}

static void emitOpTraitsDoc(const Operator &op, raw_ostream &os) {
  // TODO: We should link to the trait/documentation of it. That also means we
  // should add descriptions to traits that can be queried.
  // Collect using set to sort effects, interfaces & traits.
  std::set<std::string> effects, interfaces, traits;
  for (auto &trait : op.getTraits()) {
    if (isa<PredTrait>(&trait))
      continue;

    std::string name = trait.getDef().getName().str();
    StringRef ref = name;
    StringRef traitName = trait.getDef().getValueAsString("trait");
    traitName.consume_back("::Trait");
    traitName.consume_back("::Impl");
    if (ref.starts_with("anonymous_"))
      name = traitName.str();
    if (isa<InterfaceTrait>(&trait)) {
      if (trait.getDef().isSubClassOf("SideEffectsTraitBase")) {
        auto effectName = trait.getDef().getValueAsString("baseEffectName");
        effectName.consume_front("::");
        effectName.consume_front("mlir::");
        std::string effectStr;
        raw_string_ostream os(effectStr);
        os << effectName << "{";
        auto list = trait.getDef().getValueAsListOfDefs("effects");
        interleaveComma(list, os, [&](const Record *rec) {
          StringRef effect = rec->getValueAsString("effect");
          effect.consume_front("::");
          effect.consume_front("mlir::");
          os << effect << " on " << rec->getValueAsString("resource");
        });
        os << "}";
        effects.insert(backticks(effectStr));
        name.append(formatv(" ({0})", traitName).str());
      }
      interfaces.insert(backticks(name));
      continue;
    }

    traits.insert(backticks(name));
  }
  if (!traits.empty()) {
    interleaveComma(traits, os << "\nTraits: ");
    os << "\n";
  }
  if (!interfaces.empty()) {
    interleaveComma(interfaces, os << "\nInterfaces: ");
    os << "\n";
  }
  if (!effects.empty()) {
    interleaveComma(effects, os << "\nEffects: ");
    os << "\n";
  }
}

static StringRef resolveAttrDescription(const Attribute &attr) {
  StringRef description = attr.getDescription();
  if (description.empty())
    return attr.getBaseAttr().getDescription();
  return description;
}

/// Emit the given named constraint.
template <typename T>
static void emitNamedConstraint(const T &it, raw_ostream &os) {
  if (!it.name.empty())
    os << "| `" << it.name << "`";
  else
    os << "| &laquo;unnamed&raquo;";
  os << " | " << it.constraint.getSummary() << " |\n";
}

void mlir::tblgen::emitOpDoc(const Operator &op, StringRef stripPrefix,
                             bool allowHugoSpecificFeatures, raw_ostream &os) {
  std::string classNameStr = op.getQualCppClassName();
  StringRef className = classNameStr;
  (void)className.consume_front(stripPrefix);
  os << formatv("\n### `{0}` ({1})\n", op.getOperationName(), className);

  // Emit the summary, syntax, and description if present.
  if (op.hasSummary())
    emitSummary(op.getSummary(), os);
  if (op.hasAssemblyFormat())
    emitAssemblyFormat(op.getOperationName(), op.getAssemblyFormat().trim(),
                       os);
  if (op.hasDescription())
    mlir::tblgen::emitDescription(op.getDescription(), os);

  emitOpTraitsDoc(op, os);

  // Emit attributes.
  if (op.getNumAttributes() != 0) {
    os << "\n#### Attributes:\n\n";
    // Note: This table is HTML rather than markdown so the attribute's
    // description can appear in an expandable region. The description may be
    // multiple lines, which is not supported in a markdown table cell.
    os << "<table>\n";
    // Header.
    os << "<tr><th>Attribute</th><th>MLIR Type</th><th>Description</th></tr>\n";
    for (const auto &it : op.getAttributes()) {
      StringRef storageType = it.attr.getStorageType();
      // Name and storage type.
      os << "<tr>";
      os << "<td><code>" << it.name << "</code></td><td>" << storageType
         << "</td><td>";
      StringRef description = resolveAttrDescription(it.attr);
      if (allowHugoSpecificFeatures && !description.empty()) {
        // Expandable description.
        // This appears as just the summary, but when clicked shows the full
        // description.
        os << "<details>" << "<summary>" << it.attr.getSummary() << "</summary>"
           << "{{% markdown %}}" << description << "{{% /markdown %}}"
           << "</details>";
      } else {
        // Fallback: Single-line summary.
        os << it.attr.getSummary();
      }
      os << "</td></tr>\n";
    }
    os << "</table>\n";
  }

  // Emit each of the operands.
  if (op.getNumOperands() != 0) {
    os << "\n#### Operands:\n\n";
    os << "| Operand | Description |\n"
       << "| :-----: | ----------- |\n";
    for (const auto &it : op.getOperands())
      emitNamedConstraint(it, os);
  }

  // Emit results.
  if (op.getNumResults() != 0) {
    os << "\n#### Results:\n\n";
    os << "| Result | Description |\n"
       << "| :----: | ----------- |\n";
    for (const auto &it : op.getResults())
      emitNamedConstraint(it, os);
  }

  // Emit successors.
  if (op.getNumSuccessors() != 0) {
    os << "\n#### Successors:\n\n";
    os << "| Successor | Description |\n"
       << "| :-------: | ----------- |\n";
    for (const auto &it : op.getSuccessors())
      emitNamedConstraint(it, os);
  }

  os << "\n";
}

static void emitSourceLink(StringRef inputFilename, raw_ostream &os) {
  size_t pathBegin = inputFilename.find("mlir/include/mlir/");
  if (pathBegin == StringRef::npos)
    return;

  StringRef inputFromMlirInclude = inputFilename.substr(pathBegin);

  os << "\n[source](https://github.com/llvm/llvm-project/blob/main/"
     << inputFromMlirInclude << ")\n";
}

static void maybeNest(bool nest, llvm::function_ref<void(raw_ostream &os)> fn,
                      raw_ostream &os) {
  std::string str;
  raw_string_ostream ss(str);
  fn(ss);
  for (StringRef x : llvm::split(str, "\n")) {
    if (nest && x.starts_with("#"))
      os << "#";
    os << x << "\n";
  }
}

static void emitOpDocGroup(const OpDocGroup &grouping, StringRef stripPrefix,
                           bool allowHugoSpecificFeatures, raw_ostream &os) {
  bool nested = !grouping.summary.empty();
  maybeNest(
      nested,
      [&](raw_ostream &os) {
        if (nested) {
          os << "\n## " << StringRef(grouping.summary).trim() << "\n";
          emitDescription(grouping.description, os);
          os << "\n";
        }
        for (const Operator &op : grouping.ops)
          mlir::tblgen::emitOpDoc(op, stripPrefix, allowHugoSpecificFeatures,
                                  os);
      },
      os);
}

bool mlir::tblgen::emitOpDoc(const DialectRecords &records,
                             StringRef stripPrefix,
                             bool allowHugoSpecificFeatures, raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  emitSourceLink(records.inputFilename, os);
  for (const OpDocGroup &grouping : records.ops)
    emitOpDocGroup(grouping, stripPrefix, allowHugoSpecificFeatures, os);
  return false;
}

//===----------------------------------------------------------------------===//
// Attribute Documentation
//===----------------------------------------------------------------------===//

static void emitAttrDoc(const Attribute &attr, raw_ostream &os) {
  os << "\n### " << attr.getSummary() << "\n";
  emitDescription(attr.getDescription(), os);
  os << "\n";
}

//===----------------------------------------------------------------------===//
// Type Documentation
//===----------------------------------------------------------------------===//

static void emitTypeDoc(const Type &type, raw_ostream &os) {
  os << "\n### " << type.getSummary() << "\n";
  emitDescription(type.getDescription(), os);
  os << "\n";
}

//===----------------------------------------------------------------------===//
// TypeDef/AttrDef Documentation
//===----------------------------------------------------------------------===//

static void emitAttrOrTypeDefAssemblyFormat(const AttrOrTypeDef &def,
                                            raw_ostream &os) {
  ArrayRef<AttrOrTypeParameter> parameters = def.getParameters();
  char prefix = isa<AttrDef>(def) ? '#' : '!';
  if (parameters.empty()) {
    os << "\nSyntax: `" << prefix << def.getDialect().getName() << "."
       << def.getMnemonic() << "`\n";
    return;
  }

  os << "\nSyntax:\n\n```\n"
     << prefix << def.getDialect().getName() << "." << def.getMnemonic()
     << "<\n";
  for (const auto &it : llvm::enumerate(parameters)) {
    const AttrOrTypeParameter &param = it.value();
    os << "  " << param.getSyntax();
    if (it.index() < (parameters.size() - 1))
      os << ",";
    os << "   # " << param.getName() << "\n";
  }
  os << ">\n```\n";
}

static void emitAttrOrTypeDefDoc(const AttrOrTypeDef &def, raw_ostream &os) {
  os << formatv("\n### {0}\n", def.getCppClassName());

  // Emit the summary if present.
  if (def.hasSummary())
    emitSummary(def.getSummary(), os);

  // Emit the syntax if present.
  if (def.getMnemonic() && !def.hasCustomAssemblyFormat())
    emitAttrOrTypeDefAssemblyFormat(def, os);

  // Emit the description if present.
  if (def.hasDescription())
    mlir::tblgen::emitDescription(def.getDescription(), os);

  // Emit parameter documentation.
  ArrayRef<AttrOrTypeParameter> parameters = def.getParameters();
  if (!parameters.empty()) {
    os << "\n#### Parameters:\n\n";
    os << "| Parameter | C++ type | Description |\n"
       << "| :-------: | :-------: | ----------- |";
    for (const auto &it : parameters) {
      auto desc = it.getSummary();
      os << "\n| " << it.getName() << " | `" << it.getCppType() << "` | "
         << (desc ? *desc : "") << " |";
    }
  }

  os << "\n";
}

bool mlir::tblgen::emitAttrDefDoc(const DialectRecords &records,
                                  raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  for (const AttrDef &def : records.attrDefs)
    emitAttrOrTypeDefDoc(def, os);
  return false;
}

bool mlir::tblgen::emitTypeDefDoc(const DialectRecords &records,
                                  raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  for (const TypeDef &def : records.typeDefs)
    emitAttrOrTypeDefDoc(def, os);
  return false;
}

//===----------------------------------------------------------------------===//
// Enum Documentation
//===----------------------------------------------------------------------===//

static void emitSingleEnumDoc(const EnumInfo &def, raw_ostream &os) {
  os << formatv("\n### {0}\n", def.getEnumClassName());

  // Emit the summary if present.
  emitSummary(def.getSummary(), os);

  // Emit case documentation.
  std::vector<EnumCase> cases = def.getAllCases();
  os << "\n#### Cases:\n\n";
  os << "| Symbol | Value | String |\n"
     << "| :----: | :---: | ------ |";
  for (const auto &it : cases) {
    os << "\n| " << it.getSymbol() << " | `" << it.getValue() << "` | "
       << it.getStr() << " |";
  }

  os << "\n";
}

bool mlir::tblgen::emitEnumDoc(const DialectRecords &records, raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  for (const EnumInfo &def : records.enums)
    emitSingleEnumDoc(def, os);
  return false;
}

//===----------------------------------------------------------------------===//
// Dialect Documentation
//===----------------------------------------------------------------------===//

static void emitBlock(const DialectRecords &records, StringRef stripPrefix,
                      bool allowHugoSpecificFeatures, raw_ostream &os) {
  if (!records.ops.empty()) {
    os << "\n## Operations\n";
    emitSourceLink(records.inputFilename, os);
    for (const OpDocGroup &grouping : records.ops)
      emitOpDocGroup(grouping, stripPrefix, allowHugoSpecificFeatures, os);
  }

  if (!records.attributes.empty()) {
    os << "\n## Attribute constraints\n";
    for (const Attribute &attr : records.attributes)
      emitAttrDoc(attr, os);
  }

  if (!records.attrDefs.empty()) {
    os << "\n## Attributes\n";
    for (const AttrDef &def : records.attrDefs)
      emitAttrOrTypeDefDoc(def, os);
  }

  // TODO: Add link between use and def for types
  if (!records.types.empty()) {
    os << "\n## Type constraints\n";
    for (const Type &type : records.types)
      emitTypeDoc(type, os);
  }

  if (!records.typeDefs.empty()) {
    os << "\n## Types\n";
    for (const TypeDef &def : records.typeDefs)
      emitAttrOrTypeDefDoc(def, os);
  }

  if (!records.enums.empty()) {
    os << "\n## Enums\n";
    for (const EnumInfo &def : records.enums)
      emitSingleEnumDoc(def, os);
  }
}

bool mlir::tblgen::emitDialectDoc(const DialectRecords &records,
                                  StringRef stripPrefix,
                                  bool allowHugoSpecificFeatures,
                                  raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  os << "\n# '" << records.dialect.getName() << "' Dialect\n";
  emitSummary(records.dialect.getSummary(), os);
  emitDescription(records.dialect.getDescription(), os);

  // Generate a TOC marker except if description already contains one.
  Regex r("^[[:space:]]*\\[TOC\\]$", Regex::RegexFlags::Newline);
  if (!r.match(records.dialect.getDescription()))
    os << "\n[TOC]\n";

  emitBlock(records, stripPrefix, allowHugoSpecificFeatures, os);
  return false;
}

//===----------------------------------------------------------------------===//
// Record Collection
//===----------------------------------------------------------------------===//

std::optional<DialectRecords>
mlir::tblgen::collectRecords(const RecordKeeper &records,
                             ArrayRef<const Record *> opDefs,
                             const Dialect &dialect, bool keepOpSourceOrder) {
  auto attrDefs = records.getAllDerivedDefinitionsIfDefined("DialectAttr");
  auto typeDefs = records.getAllDerivedDefinitionsIfDefined("DialectType");
  auto typeDefDefs = records.getAllDerivedDefinitionsIfDefined("TypeDef");
  auto attrDefDefs = records.getAllDerivedDefinitionsIfDefined("AttrDef");
  auto enumDefs = records.getAllDerivedDefinitionsIfDefined("EnumInfo");

  DialectRecords result(dialect, records.getInputFilename());
  SmallDenseSet<const Record *> seen;
  auto addIfNotSeen = [&](const Record *record, const auto &def, auto &vec) {
    if (seen.insert(record).second) {
      vec.push_back(def);
      return true;
    }
    return false;
  };
  auto addIfInDialect = [&](const Record *record, const auto &def, auto &vec) {
    return def.getDialect() == dialect && addIfNotSeen(record, def, vec);
  };

  SmallDenseMap<const Record *, OpDocGroup> opDocGroup;

  for (const Record *def : attrDefDefs)
    addIfInDialect(def, AttrDef(def), result.attrDefs);
  for (const Record *def : attrDefs)
    addIfInDialect(def, Attribute(def), result.attributes);
  for (const Record *def : opDefs) {
    if (const Record *group = def->getValueAsOptionalDef("opDocGroup")) {
      OpDocGroup &op = opDocGroup[group];
      addIfInDialect(def, Operator(def), op.ops);
    } else {
      OpDocGroup op;
      op.ops.emplace_back(def);
      addIfInDialect(def, op, result.ops);
    }
  }
  for (const Record *rec :
       records.getAllDerivedDefinitionsIfDefined("OpDocGroup")) {
    if (opDocGroup[rec].ops.empty())
      continue;
    opDocGroup[rec].summary = rec->getValueAsString("summary");
    opDocGroup[rec].description = rec->getValueAsString("description");
    result.ops.push_back(opDocGroup[rec]);
  }
  for (const Record *def : typeDefDefs)
    addIfInDialect(def, TypeDef(def), result.typeDefs);
  for (const Record *def : typeDefs)
    addIfInDialect(def, Type(def), result.types);
  result.enums.reserve(enumDefs.size());
  for (const Record *def : enumDefs)
    addIfNotSeen(def, EnumInfo(def), result.enums);

  // Sort alphabetically ignoring dialect for ops and section name for sections.
  // TODO: The sorting order could be revised, currently attempting to sort of
  // keep in alphabetical order.
  if (keepOpSourceOrder)
    return result;
  llvm::sort(result.ops, [](const OpDocGroup &lhs, const OpDocGroup &rhs) {
    auto getDesc = [](const OpDocGroup &arg) -> StringRef {
      if (!arg.summary.empty())
        return arg.summary;
      return arg.ops.front().getDef().getValueAsString("opName");
    };
    return getDesc(lhs).compare_insensitive(getDesc(rhs)) < 0;
  });

  return result;
}
