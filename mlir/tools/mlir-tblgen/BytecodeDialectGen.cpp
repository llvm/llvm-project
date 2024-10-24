//===- BytecodeDialectGen.cpp - Dialect bytecode read/writer gen  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <regex>

using namespace llvm;

static cl::OptionCategory dialectGenCat("Options for -gen-bytecode");
static cl::opt<std::string>
    selectedBcDialect("bytecode-dialect", cl::desc("The dialect to gen for"),
                      cl::cat(dialectGenCat), cl::CommaSeparated);

namespace {

/// Helper class to generate C++ bytecode parser helpers.
class Generator {
public:
  Generator(raw_ostream &output) : output(output) {}

  /// Returns whether successfully emitted attribute/type parsers.
  void emitParse(StringRef kind, const Record &x);

  /// Returns whether successfully emitted attribute/type printers.
  void emitPrint(StringRef kind, StringRef type,
                 ArrayRef<std::pair<int64_t, const Record *>> vec);

  /// Emits parse dispatch table.
  void emitParseDispatch(StringRef kind, ArrayRef<const Record *> vec);

  /// Emits print dispatch table.
  void emitPrintDispatch(StringRef kind, ArrayRef<std::string> vec);

private:
  /// Emits parse calls to construct given kind.
  void emitParseHelper(StringRef kind, StringRef returnType, StringRef builder,
                       ArrayRef<Init *> args, ArrayRef<std::string> argNames,
                       StringRef failure, mlir::raw_indented_ostream &ios);

  /// Emits print instructions.
  void emitPrintHelper(const Record *memberRec, StringRef kind,
                       StringRef parent, StringRef name,
                       mlir::raw_indented_ostream &ios);

  raw_ostream &output;
};
} // namespace

/// Helper to replace set of from strings to target in `s`.
/// Assumed: non-overlapping replacements.
static std::string format(StringRef templ,
                          std::map<std::string, std::string> &&map) {
  std::string s = templ.str();
  for (const auto &[from, to] : map)
    // All replacements start with $, don't treat as anchor.
    s = std::regex_replace(s, std::regex("\\" + from), to);
  return s;
}

/// Return string with first character capitalized.
static std::string capitalize(StringRef str) {
  return ((Twine)toUpper(str[0]) + str.drop_front()).str();
}

/// Return the C++ type for the given record.
static std::string getCType(const Record *def) {
  std::string format = "{0}";
  if (def->isSubClassOf("Array")) {
    def = def->getValueAsDef("elemT");
    format = "SmallVector<{0}>";
  }

  StringRef cType = def->getValueAsString("cType");
  if (cType.empty()) {
    if (def->isAnonymous())
      PrintFatalError(def->getLoc(), "Unable to determine cType");

    return formatv(format.c_str(), def->getName().str());
  }
  return formatv(format.c_str(), cType.str());
}

void Generator::emitParseDispatch(StringRef kind,
                                  ArrayRef<const Record *> vec) {
  mlir::raw_indented_ostream os(output);
  char const *head =
      R"(static {0} read{0}(MLIRContext* context, DialectBytecodeReader &reader))";
  os << formatv(head, capitalize(kind));
  auto funScope = os.scope(" {\n", "}\n\n");

  if (vec.empty()) {
    os << "return reader.emitError() << \"unknown attribute\", "
       << capitalize(kind) << "();\n";
    return;
  }

  os << "uint64_t kind;\n";
  os << "if (failed(reader.readVarInt(kind)))\n"
     << "  return " << capitalize(kind) << "();\n";
  os << "switch (kind) ";
  {
    auto switchScope = os.scope("{\n", "}\n");
    for (const auto &it : llvm::enumerate(vec)) {
      if (it.value()->getName() == "ReservedOrDead")
        continue;

      os << formatv("case {1}:\n  return read{0}(context, reader);\n",
                    it.value()->getName(), it.index());
    }
    os << "default:\n"
       << "  reader.emitError() << \"unknown attribute code: \" "
       << "<< kind;\n"
       << "  return " << capitalize(kind) << "();\n";
  }
  os << "return " << capitalize(kind) << "();\n";
}

void Generator::emitParse(StringRef kind, const Record &x) {
  if (x.getNameInitAsString() == "ReservedOrDead")
    return;

  char const *head =
      R"(static {0} read{1}(MLIRContext* context, DialectBytecodeReader &reader) )";
  mlir::raw_indented_ostream os(output);
  std::string returnType = getCType(&x);
  os << formatv(head, kind == "attribute" ? "::mlir::Attribute" : "::mlir::Type", x.getName());
  DagInit *members = x.getValueAsDag("members");
  SmallVector<std::string> argNames =
      llvm::to_vector(map_range(members->getArgNames(), [](StringInit *init) {
        return init->getAsUnquotedString();
      }));
  StringRef builder = x.getValueAsString("cBuilder").trim();
  emitParseHelper(kind, returnType, builder, members->getArgs(), argNames,
                  returnType + "()", os);
  os << "\n\n";
}

void printParseConditional(mlir::raw_indented_ostream &ios,
                           ArrayRef<Init *> args,
                           ArrayRef<std::string> argNames) {
  ios << "if ";
  auto parenScope = ios.scope("(", ") {");
  ios.indent();

  auto listHelperName = [](StringRef name) {
    return formatv("read{0}", capitalize(name));
  };

  auto parsedArgs =
      llvm::to_vector(make_filter_range(args, [](Init *const attr) {
        const Record *def = cast<DefInit>(attr)->getDef();
        if (def->isSubClassOf("Array"))
          return true;
        return !def->getValueAsString("cParser").empty();
      }));

  interleave(
      zip(parsedArgs, argNames),
      [&](std::tuple<llvm::Init *&, const std::string &> it) {
        const Record *attr = cast<DefInit>(std::get<0>(it))->getDef();
        std::string parser;
        if (auto optParser = attr->getValueAsOptionalString("cParser")) {
          parser = *optParser;
        } else if (attr->isSubClassOf("Array")) {
          const Record *def = attr->getValueAsDef("elemT");
          bool composite = def->isSubClassOf("CompositeBytecode");
          if (!composite && def->isSubClassOf("AttributeKind"))
            parser = "succeeded($_reader.readAttributes($_var))";
          else if (!composite && def->isSubClassOf("TypeKind"))
            parser = "succeeded($_reader.readTypes($_var))";
          else
            parser = ("succeeded($_reader.readList($_var, " +
                      listHelperName(std::get<1>(it)) + "))")
                         .str();
        } else {
          PrintFatalError(attr->getLoc(), "No parser specified");
        }
        std::string type = getCType(attr);
        ios << format(parser, {{"$_reader", "reader"},
                               {"$_resultType", type},
                               {"$_var", std::get<1>(it)}});
      },
      [&]() { ios << " &&\n"; });
}

void Generator::emitParseHelper(StringRef kind, StringRef returnType,
                                StringRef builder, ArrayRef<Init *> args,
                                ArrayRef<std::string> argNames,
                                StringRef failure,
                                mlir::raw_indented_ostream &ios) {
  auto funScope = ios.scope("{\n", "}");

  if (args.empty()) {
    ios << formatv("return get<{0}>(context);\n", returnType);
    return;
  }

  // Print decls.
  std::string lastCType = "";
  for (auto [arg, name] : zip(args, argNames)) {
    DefInit *first = dyn_cast<DefInit>(arg);
    if (!first)
      PrintFatalError("Unexpected type for " + name);
    const Record *def = first->getDef();

    // Create variable decls, if there are a block of same type then create
    // comma separated list of them.
    std::string cType = getCType(def);
    if (lastCType == cType) {
      ios << ", ";
    } else {
      if (!lastCType.empty())
        ios << ";\n";
      ios << cType << " ";
    }
    ios << name;
    lastCType = cType;
  }
  ios << ";\n";

  // Returns the name of the helper used in list parsing. E.g., the name of the
  // lambda passed to array parsing.
  auto listHelperName = [](StringRef name) {
    return formatv("read{0}", capitalize(name));
  };

  // Emit list helper functions.
  for (auto [arg, name] : zip(args, argNames)) {
    const Record *attr = cast<DefInit>(arg)->getDef();
    if (!attr->isSubClassOf("Array"))
      continue;

    // TODO: Dedupe readers.
    const Record *def = attr->getValueAsDef("elemT");
    if (!def->isSubClassOf("CompositeBytecode") &&
        (def->isSubClassOf("AttributeKind") || def->isSubClassOf("TypeKind")))
      continue;

    std::string returnType = getCType(def);
    ios << "auto " << listHelperName(name) << " = [&]() -> FailureOr<"
        << returnType << "> ";
    SmallVector<Init *> args;
    SmallVector<std::string> argNames;
    if (def->isSubClassOf("CompositeBytecode")) {
      DagInit *members = def->getValueAsDag("members");
      args = llvm::to_vector(members->getArgs());
      argNames = llvm::to_vector(
          map_range(members->getArgNames(), [](StringInit *init) {
            return init->getAsUnquotedString();
          }));
    } else {
      args = {def->getDefInit()};
      argNames = {"temp"};
    }
    StringRef builder = def->getValueAsString("cBuilder");
    emitParseHelper(kind, returnType, builder, args, argNames, "failure()",
                    ios);
    ios << ";\n";
  }

  // Print parse conditional.
  printParseConditional(ios, args, argNames);

  // Compute args to pass to create method.
  auto passedArgs = llvm::to_vector(make_filter_range(
      argNames, [](StringRef str) { return !str.starts_with("_"); }));
  std::string argStr;
  raw_string_ostream argStream(argStr);
  interleaveComma(passedArgs, argStream,
                  [&](const std::string &str) { argStream << str; });
  // Return the invoked constructor.
  ios << "\nreturn "
      << format(builder, {{"$_resultType", returnType.str()},
                          {"$_args", argStream.str()}})
      << ";\n";
  ios.unindent();

  // TODO: Emit error in debug.
  // This assumes the result types in error case can always be empty
  // constructed.
  ios << "}\nreturn " << failure << ";\n";
}

void Generator::emitPrint(StringRef kind, StringRef type,
                          ArrayRef<std::pair<int64_t, const Record *>> vec) {
  if (type == "ReservedOrDead")
    return;

  char const *head =
      R"(static void write({0} {1}, DialectBytecodeWriter &writer) )";
  mlir::raw_indented_ostream os(output);
  os << formatv(head, type, kind);
  auto funScope = os.scope("{\n", "}\n\n");

  // Check that predicates specified if multiple bytecode instances.
  for (const Record *rec : make_second_range(vec)) {
    StringRef pred = rec->getValueAsString("printerPredicate");
    if (vec.size() > 1 && pred.empty()) {
      for (auto [index, rec] : vec) {
        (void)index;
        StringRef pred = rec->getValueAsString("printerPredicate");
        if (vec.size() > 1 && pred.empty())
          PrintError(rec->getLoc(),
                     "Requires parsing predicate given common cType");
      }
      PrintFatalError("Unspecified for shared cType " + type);
    }
  }

  for (auto [index, rec] : vec) {
    StringRef pred = rec->getValueAsString("printerPredicate");
    if (!pred.empty()) {
      os << "if (" << format(pred, {{"$_val", kind.str()}}) << ") {\n";
      os.indent();
    }

    os << "writer.writeVarInt(/* " << rec->getName() << " */ " << index
       << ");\n";

    auto *members = rec->getValueAsDag("members");
    for (auto [arg, name] :
         llvm::zip(members->getArgs(), members->getArgNames())) {
      DefInit *def = dyn_cast<DefInit>(arg);
      assert(def);
      const Record *memberRec = def->getDef();
      emitPrintHelper(memberRec, kind, kind, name->getAsUnquotedString(), os);
    }

    if (!pred.empty()) {
      os.unindent();
      os << "}\n";
    }
  }
}

void Generator::emitPrintHelper(const Record *memberRec, StringRef kind,
                                StringRef parent, StringRef name,
                                mlir::raw_indented_ostream &ios) {
  std::string getter;
  if (auto cGetter = memberRec->getValueAsOptionalString("cGetter");
      cGetter && !cGetter->empty()) {
    getter = format(
        *cGetter,
        {{"$_attrType", parent.str()},
         {"$_member", name.str()},
         {"$_getMember", "get" + convertToCamelFromSnakeCase(name, true)}});
  } else {
    getter =
        formatv("{0}.get{1}()", parent, convertToCamelFromSnakeCase(name, true))
            .str();
  }

  if (memberRec->isSubClassOf("Array")) {
    const Record *def = memberRec->getValueAsDef("elemT");
    if (!def->isSubClassOf("CompositeBytecode")) {
      if (def->isSubClassOf("AttributeKind")) {
        ios << "writer.writeAttributes(" << getter << ");\n";
        return;
      }
      if (def->isSubClassOf("TypeKind")) {
        ios << "writer.writeTypes(" << getter << ");\n";
        return;
      }
    }
    std::string returnType = getCType(def);
    std::string nestedName = kind.str();
    ios << "writer.writeList(" << getter << ", [&](" << returnType << " "
        << nestedName << ") ";
    auto lambdaScope = ios.scope("{\n", "});\n");
    return emitPrintHelper(def, kind, nestedName, nestedName, ios);
  }
  if (memberRec->isSubClassOf("CompositeBytecode")) {
    auto *members = memberRec->getValueAsDag("members");
    for (auto [arg, argName] :
         zip(members->getArgs(), members->getArgNames())) {
      DefInit *def = dyn_cast<DefInit>(arg);
      assert(def);
      emitPrintHelper(def->getDef(), kind, parent,
                      argName->getAsUnquotedString(), ios);
    }
  }

  if (std::string printer = memberRec->getValueAsString("cPrinter").str();
      !printer.empty())
    ios << format(printer, {{"$_writer", "writer"},
                            {"$_name", kind.str()},
                            {"$_getter", getter}})
        << ";\n";
}

void Generator::emitPrintDispatch(StringRef kind, ArrayRef<std::string> vec) {
  mlir::raw_indented_ostream os(output);
  char const *head = R"(static LogicalResult write{0}({0} {1},
                                DialectBytecodeWriter &writer))";
  os << formatv(head, capitalize(kind), kind);
  auto funScope = os.scope(" {\n", "}\n\n");

  os << "return TypeSwitch<" << capitalize(kind) << ", LogicalResult>(" << kind
     << ")";
  auto switchScope = os.scope("", "");
  for (StringRef type : vec) {
    if (type == "ReservedOrDead")
      continue;

    os << "\n.Case([&](" << type << " t)";
    auto caseScope = os.scope(" {\n", "})");
    os << "return write(t, writer), success();\n";
  }
  os << "\n.Default([&](" << capitalize(kind) << ") { return failure(); });\n";
}

namespace {
/// Container of Attribute or Type for Dialect.
struct AttrOrType {
  std::vector<const Record *> attr, type;
};
} // namespace

static bool emitBCRW(const RecordKeeper &records, raw_ostream &os) {
  MapVector<StringRef, AttrOrType> dialectAttrOrType;
  for (const Record *it :
       records.getAllDerivedDefinitions("DialectAttributes")) {
    if (!selectedBcDialect.empty() &&
        it->getValueAsString("dialect") != selectedBcDialect)
      continue;
    dialectAttrOrType[it->getValueAsString("dialect")].attr =
        it->getValueAsListOfDefs("elems");
  }
  for (const Record *it : records.getAllDerivedDefinitions("DialectTypes")) {
    if (!selectedBcDialect.empty() &&
        it->getValueAsString("dialect") != selectedBcDialect)
      continue;
    dialectAttrOrType[it->getValueAsString("dialect")].type =
        it->getValueAsListOfDefs("elems");
  }

  if (dialectAttrOrType.size() != 1)
    PrintFatalError("Single dialect per invocation required (either only "
                    "one in input file or specified via dialect option)");

  auto it = dialectAttrOrType.front();
  Generator gen(os);

  SmallVector<std::vector<const Record *> *, 2> vecs;
  SmallVector<std::string, 2> kinds;
  vecs.push_back(&it.second.attr);
  kinds.push_back("attribute");
  vecs.push_back(&it.second.type);
  kinds.push_back("type");
  for (auto [vec, kind] : zip(vecs, kinds)) {
    // Handle Attribute/Type emission.
    std::map<std::string, std::vector<std::pair<int64_t, const Record *>>>
        perType;
    for (auto kt : llvm::enumerate(*vec))
      perType[getCType(kt.value())].emplace_back(kt.index(), kt.value());
    for (const auto &jt : perType) {
      for (auto kt : jt.second)
        gen.emitParse(kind, *std::get<1>(kt));
      gen.emitPrint(kind, jt.first, jt.second);
    }
    gen.emitParseDispatch(kind, *vec);

    SmallVector<std::string> types;
    for (const auto &it : perType) {
      types.push_back(it.first);
    }
    gen.emitPrintDispatch(kind, types);
  }

  return false;
}

static mlir::GenRegistration
    genBCRW("gen-bytecode", "Generate dialect bytecode readers/writers",
            [](const RecordKeeper &records, raw_ostream &os) {
              return emitBCRW(records, os);
            });
