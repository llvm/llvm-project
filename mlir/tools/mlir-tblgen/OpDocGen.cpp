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

#include "DialectGenUtilities.h"
#include "DocGenUtilities.h"
#include "OpGenHelpers.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <set>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;
using mlir::tblgen::Operator;

//===----------------------------------------------------------------------===//
// Commandline Options
//===----------------------------------------------------------------------===//
static cl::OptionCategory
    docCat("Options for -gen-(attrdef|typedef|enum|op|dialect)-doc");
cl::opt<std::string>
    stripPrefix("strip-prefix",
                cl::desc("Strip prefix of the fully qualified names"),
                cl::init("::mlir::"), cl::cat(docCat));
cl::opt<bool> allowHugoSpecificFeatures(
    "allow-hugo-specific-features",
    cl::desc("Allows using features specific to Hugo"), cl::init(false),
    cl::cat(docCat));

void mlir::tblgen::emitSummary(StringRef summary, raw_ostream &os) {
  if (!summary.empty()) {
    StringRef trimmed = summary.trim();
    char first = std::toupper(trimmed.front());
    StringRef rest = trimmed.drop_front();
    os << "\n_" << first << rest << "_\n\n";
  }
}

// Emit the description by aligning the text to the left per line (e.g.,
// removing the minimum indentation across the block).
//
// This expects that the description in the tablegen file is already formatted
// in a way the user wanted but has some additional indenting due to being
// nested in the op definition.
void mlir::tblgen::emitDescription(StringRef description, raw_ostream &os) {
  raw_indented_ostream ros(os);
  ros.printReindented(description.rtrim(" \t"));
}

void mlir::tblgen::emitDescriptionComment(StringRef description,
                                          raw_ostream &os, StringRef prefix) {
  if (description.empty())
    return;
  raw_indented_ostream ros(os);
  StringRef trimmed = description.rtrim(" \t");
  ros.printReindented(trimmed, (Twine(prefix) + "/// ").str());
  if (!trimmed.ends_with("\n"))
    ros << "\n";
}

// Emits `str` with trailing newline if not empty.
static void emitIfNotEmpty(StringRef str, raw_ostream &os) {
  if (!str.empty()) {
    emitDescription(str, os);
    os << "\n";
  }
}

/// Emit the given named constraint.
template <typename T>
static void emitNamedConstraint(const T &it, raw_ostream &os) {
  if (!it.name.empty())
    os << "| `" << it.name << "`";
  else
    os << "&laquo;unnamed&raquo;";
  os << " | " << it.constraint.getSummary() << "\n";
}

//===----------------------------------------------------------------------===//
// Operation Documentation
//===----------------------------------------------------------------------===//

/// Emit the assembly format of an operation.
static void emitAssemblyFormat(StringRef opName, StringRef format,
                               raw_ostream &os) {
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
  os << "```\n\n";
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

static void emitOpDoc(const Operator &op, raw_ostream &os) {
  std::string classNameStr = op.getQualCppClassName();
  StringRef className = classNameStr;
  (void)className.consume_front(stripPrefix);
  os << formatv("### `{0}` ({1})\n", op.getOperationName(), className);

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

  os << "[source](https://github.com/llvm/llvm-project/blob/main/"
     << inputFromMlirInclude << ")\n\n";
}

static void emitOpDoc(const RecordKeeper &recordKeeper, raw_ostream &os) {
  auto opDefs = getRequestedOpDefinitions(recordKeeper);

  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  emitSourceLink(recordKeeper.getInputFilename(), os);
  for (const Record *opDef : opDefs)
    emitOpDoc(Operator(opDef), os);
}

//===----------------------------------------------------------------------===//
// Attribute Documentation
//===----------------------------------------------------------------------===//

static void emitAttrDoc(const Attribute &attr, raw_ostream &os) {
  os << "### " << attr.getSummary() << "\n\n";
  emitDescription(attr.getDescription(), os);
  os << "\n\n";
}

//===----------------------------------------------------------------------===//
// Type Documentation
//===----------------------------------------------------------------------===//

static void emitTypeDoc(const Type &type, raw_ostream &os) {
  os << "### " << type.getSummary() << "\n\n";
  emitDescription(type.getDescription(), os);
  os << "\n\n";
}

//===----------------------------------------------------------------------===//
// TypeDef Documentation
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
  os << formatv("### {0}\n", def.getCppClassName());

  // Emit the summary if present.
  if (def.hasSummary())
    os << "\n" << def.getSummary() << "\n";

  // Emit the syntax if present.
  if (def.getMnemonic() && !def.hasCustomAssemblyFormat())
    emitAttrOrTypeDefAssemblyFormat(def, os);

  // Emit the description if present.
  if (def.hasDescription()) {
    os << "\n";
    mlir::tblgen::emitDescription(def.getDescription(), os);
  }

  // Emit parameter documentation.
  ArrayRef<AttrOrTypeParameter> parameters = def.getParameters();
  if (!parameters.empty()) {
    os << "\n#### Parameters:\n\n";
    os << "| Parameter | C++ type | Description |\n"
       << "| :-------: | :-------: | ----------- |\n";
    for (const auto &it : parameters) {
      auto desc = it.getSummary();
      os << "| " << it.getName() << " | `" << it.getCppType() << "` | "
         << (desc ? *desc : "") << " |\n";
    }
  }

  os << "\n";
}

static void emitAttrOrTypeDefDoc(const RecordKeeper &recordKeeper,
                                 raw_ostream &os, StringRef recordTypeName) {
  auto defs = recordKeeper.getAllDerivedDefinitions(recordTypeName);

  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  for (const Record *def : defs)
    emitAttrOrTypeDefDoc(AttrOrTypeDef(def), os);
}

//===----------------------------------------------------------------------===//
// Enum Documentation
//===----------------------------------------------------------------------===//

static void emitEnumDoc(const EnumAttr &def, raw_ostream &os) {
  os << formatv("### {0}\n", def.getEnumClassName());

  // Emit the summary if present.
  if (!def.getSummary().empty())
    os << "\n" << def.getSummary() << "\n";

  // Emit case documentation.
  std::vector<EnumAttrCase> cases = def.getAllCases();
  os << "\n#### Cases:\n\n";
  os << "| Symbol | Value | String |\n"
     << "| :----: | :---: | ------ |\n";
  for (const auto &it : cases) {
    os << "| " << it.getSymbol() << " | `" << it.getValue() << "` | "
       << it.getStr() << " |\n";
  }

  os << "\n";
}

static void emitEnumDoc(const RecordKeeper &recordKeeper, raw_ostream &os) {
  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  for (const Record *def : recordKeeper.getAllDerivedDefinitions("EnumAttr"))
    emitEnumDoc(EnumAttr(def), os);
}

//===----------------------------------------------------------------------===//
// Dialect Documentation
//===----------------------------------------------------------------------===//

struct OpDocGroup {
  const Dialect &getDialect() const { return ops.front().getDialect(); }

  // Returns the summary description of the section.
  std::string summary = "";

  // Returns the description of the section.
  StringRef description = "";

  // Instances inside the section.
  std::vector<Operator> ops;
};

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

static void emitBlock(ArrayRef<Attribute> attributes, StringRef inputFilename,
                      ArrayRef<AttrDef> attrDefs, ArrayRef<OpDocGroup> ops,
                      ArrayRef<Type> types, ArrayRef<TypeDef> typeDefs,
                      ArrayRef<EnumAttr> enums, raw_ostream &os) {
  if (!ops.empty()) {
    os << "## Operations\n\n";
    emitSourceLink(inputFilename, os);
    for (const OpDocGroup &grouping : ops) {
      bool nested = !grouping.summary.empty();
      maybeNest(
          nested,
          [&](raw_ostream &os) {
            if (nested) {
              os << "## " << StringRef(grouping.summary).trim() << "\n\n";
              emitDescription(grouping.description, os);
              os << "\n\n";
            }
            for (const Operator &op : grouping.ops) {
              emitOpDoc(op, os);
            }
          },
          os);
    }
  }

  if (!attributes.empty()) {
    os << "## Attribute constraints\n\n";
    for (const Attribute &attr : attributes)
      emitAttrDoc(attr, os);
  }

  if (!attrDefs.empty()) {
    os << "## Attributes\n\n";
    for (const AttrDef &def : attrDefs)
      emitAttrOrTypeDefDoc(def, os);
  }

  // TODO: Add link between use and def for types
  if (!types.empty()) {
    os << "## Type constraints\n\n";
    for (const Type &type : types)
      emitTypeDoc(type, os);
  }

  if (!typeDefs.empty()) {
    os << "## Types\n\n";
    for (const TypeDef &def : typeDefs)
      emitAttrOrTypeDefDoc(def, os);
  }

  if (!enums.empty()) {
    os << "## Enums\n\n";
    for (const EnumAttr &def : enums)
      emitEnumDoc(def, os);
  }
}

static void emitDialectDoc(const Dialect &dialect, StringRef inputFilename,
                           ArrayRef<Attribute> attributes,
                           ArrayRef<AttrDef> attrDefs, ArrayRef<OpDocGroup> ops,
                           ArrayRef<Type> types, ArrayRef<TypeDef> typeDefs,
                           ArrayRef<EnumAttr> enums, raw_ostream &os) {
  os << "# '" << dialect.getName() << "' Dialect\n\n";
  emitIfNotEmpty(dialect.getSummary(), os);
  emitIfNotEmpty(dialect.getDescription(), os);

  // Generate a TOC marker except if description already contains one.
  Regex r("^[[:space:]]*\\[TOC\\]$", Regex::RegexFlags::Newline);
  if (!r.match(dialect.getDescription()))
    os << "[TOC]\n\n";

  emitBlock(attributes, inputFilename, attrDefs, ops, types, typeDefs, enums,
            os);
}

static bool emitDialectDoc(const RecordKeeper &recordKeeper, raw_ostream &os) {
  auto dialectDefs = recordKeeper.getAllDerivedDefinitionsIfDefined("Dialect");
  SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect = findDialectToGenerate(dialects);
  if (!dialect)
    return true;

  std::vector<const Record *> opDefs = getRequestedOpDefinitions(recordKeeper);
  auto attrDefs = recordKeeper.getAllDerivedDefinitionsIfDefined("DialectAttr");
  auto typeDefs = recordKeeper.getAllDerivedDefinitionsIfDefined("DialectType");
  auto typeDefDefs = recordKeeper.getAllDerivedDefinitionsIfDefined("TypeDef");
  auto attrDefDefs = recordKeeper.getAllDerivedDefinitionsIfDefined("AttrDef");
  auto enumDefs =
      recordKeeper.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");

  std::vector<Attribute> dialectAttrs;
  std::vector<AttrDef> dialectAttrDefs;
  std::vector<OpDocGroup> dialectOps;
  std::vector<Type> dialectTypes;
  std::vector<TypeDef> dialectTypeDefs;
  std::vector<EnumAttr> dialectEnums;

  SmallDenseSet<const Record *> seen;
  auto addIfNotSeen = [&](const Record *record, const auto &def, auto &vec) {
    if (seen.insert(record).second) {
      vec.push_back(def);
      return true;
    }
    return false;
  };
  auto addIfInDialect = [&](const Record *record, const auto &def, auto &vec) {
    return def.getDialect() == *dialect && addIfNotSeen(record, def, vec);
  };

  SmallDenseMap<const Record *, OpDocGroup> opDocGroup;

  for (const Record *def : attrDefDefs)
    addIfInDialect(def, AttrDef(def), dialectAttrDefs);
  for (const Record *def : attrDefs)
    addIfInDialect(def, Attribute(def), dialectAttrs);
  for (const Record *def : opDefs) {
    if (const Record *group = def->getValueAsOptionalDef("opDocGroup")) {
      OpDocGroup &op = opDocGroup[group];
      addIfInDialect(def, Operator(def), op.ops);
    } else {
      OpDocGroup op;
      op.ops.emplace_back(def);
      addIfInDialect(def, op, dialectOps);
    }
  }
  for (const Record *rec :
       recordKeeper.getAllDerivedDefinitionsIfDefined("OpDocGroup")) {
    if (opDocGroup[rec].ops.empty())
      continue;
    opDocGroup[rec].summary = rec->getValueAsString("summary");
    opDocGroup[rec].description = rec->getValueAsString("description");
    dialectOps.push_back(opDocGroup[rec]);
  }
  for (const Record *def : typeDefDefs)
    addIfInDialect(def, TypeDef(def), dialectTypeDefs);
  for (const Record *def : typeDefs)
    addIfInDialect(def, Type(def), dialectTypes);
  dialectEnums.reserve(enumDefs.size());
  for (const Record *def : enumDefs)
    addIfNotSeen(def, EnumAttr(def), dialectEnums);

  // Sort alphabetically ignorning dialect for ops and section name for
  // sections.
  // TODO: The sorting order could be revised, currently attempting to sort of
  // keep in alphabetical order.
  std::sort(dialectOps.begin(), dialectOps.end(),
            [](const OpDocGroup &lhs, const OpDocGroup &rhs) {
              auto getDesc = [](const OpDocGroup &arg) -> StringRef {
                if (!arg.summary.empty())
                  return arg.summary;
                return arg.ops.front().getDef().getValueAsString("opName");
              };
              return getDesc(lhs).compare_insensitive(getDesc(rhs)) < 0;
            });

  os << "<!-- Autogenerated by mlir-tblgen; don't manually edit -->\n";
  emitDialectDoc(*dialect, recordKeeper.getInputFilename(), dialectAttrs,
                 dialectAttrDefs, dialectOps, dialectTypes, dialectTypeDefs,
                 dialectEnums, os);
  return false;
}

//===----------------------------------------------------------------------===//
// Gen Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genAttrRegister("gen-attrdef-doc",
                    "Generate dialect attribute documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      emitAttrOrTypeDefDoc(records, os, "AttrDef");
                      return false;
                    });

static mlir::GenRegistration
    genOpRegister("gen-op-doc", "Generate dialect documentation",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    emitOpDoc(records, os);
                    return false;
                  });

static mlir::GenRegistration
    genTypeRegister("gen-typedef-doc", "Generate dialect type documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      emitAttrOrTypeDefDoc(records, os, "TypeDef");
                      return false;
                    });

static mlir::GenRegistration
    genEnumRegister("gen-enum-doc", "Generate dialect enum documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      emitEnumDoc(records, os);
                      return false;
                    });

static mlir::GenRegistration
    genRegister("gen-dialect-doc", "Generate dialect documentation",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitDialectDoc(records, os);
                });
