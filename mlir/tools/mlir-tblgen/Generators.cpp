//===- Generators.cpp - Generator registrations for mlir-tblgen -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file registers all generators for mlir-tblgen by calling into the
// MLIRTableGenCppGen library. CLI options are read here and threaded as
// explicit parameters to the library functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Generators/AttrOrTypeDefGen.h"
#include "mlir/TableGen/Generators/BytecodeDialectGen.h"
#include "mlir/TableGen/Generators/DialectGen.h"
#include "mlir/TableGen/Generators/DialectInterfacesGen.h"
#include "mlir/TableGen/Generators/EnumPythonBindingGen.h"
#include "mlir/TableGen/Generators/EnumsGen.h"
#include "mlir/TableGen/Generators/FormatGen.h"
#include "mlir/TableGen/Generators/OpDefinitionsGen.h"
#include "mlir/TableGen/Generators/OpDocGen.h"
#include "mlir/TableGen/Generators/OpGenHelpers.h"
#include "mlir/TableGen/Generators/OpInterfacesGen.h"
#include "mlir/TableGen/Generators/OpPythonBindingGen.h"
#include "mlir/TableGen/Generators/PassCAPIGen.h"
#include "mlir/TableGen/Generators/PassDocGen.h"
#include "mlir/TableGen/Generators/PassGen.h"
#include "mlir/TableGen/Generators/RewriterGen.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Assembly format options (shared by AttrOrTypeDef and Op generators)
//===----------------------------------------------------------------------===//

static cl::opt<bool>
    formatErrorIsFatal("asmformat-error-is-fatal",
                       cl::desc("Emit a fatal error if format parsing fails"),
                       cl::init(true));

//===----------------------------------------------------------------------===//
// Op definition generator options (shared by op-def and op-doc generators)
//===----------------------------------------------------------------------===//

static cl::OptionCategory opDefGenCat("Options for op definition generators");
static cl::opt<std::string> opIncFilter(
    "op-include-regex",
    cl::desc("Regex of name of op's to include (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<std::string> opExcFilter(
    "op-exclude-regex",
    cl::desc("Regex of name of op's to exclude (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<unsigned> opShardCount(
    "op-shard-count",
    cl::desc("The number of shards into which the op classes will be divided"),
    cl::cat(opDefGenCat), cl::init(1));

static std::vector<const Record *>
getRequestedOpDefs(const RecordKeeper &records) {
  return getRequestedOpDefinitions(records, opIncFilter, opExcFilter);
}

static void shardOps(ArrayRef<const Record *> defs,
                     SmallVectorImpl<ArrayRef<const Record *>> &shardedDefs) {
  shardOpDefinitions(defs, shardedDefs, opShardCount);
}

//===----------------------------------------------------------------------===//
// AttrOrTypeDef generators
//===----------------------------------------------------------------------===//

static cl::OptionCategory attrdefGenCat("Options for -gen-attrdef-*");
static cl::opt<std::string>
    attrDialect("attrdefs-dialect",
                cl::desc("Generate attributes for this dialect"),
                cl::cat(attrdefGenCat), cl::CommaSeparated);

static GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  AttrDefGenerator generator(records, os, formatErrorIsFatal);
                  return generator.emitDefs(attrDialect);
                });
static GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   AttrDefGenerator generator(records, os, formatErrorIsFatal);
                   return generator.emitDecls(attrDialect);
                 });
static GenRegistration
    genAttrConstrDefs("gen-attr-constraint-defs",
                      "Generate attribute constraint definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        emitAttrConstraintDefs(records, os);
                        return false;
                      });
static GenRegistration
    genAttrConstrDecls("gen-attr-constraint-decls",
                       "Generate attribute constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitAttrConstraintDecls(records, os);
                         return false;
                       });

static cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static cl::opt<std::string>
    typeDialect("typedefs-dialect", cl::desc("Generate types for this dialect"),
                cl::cat(typedefGenCat), cl::CommaSeparated);

static GenRegistration
    genTypeDefs("gen-typedef-defs", "Generate TypeDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  TypeDefGenerator generator(records, os, formatErrorIsFatal);
                  return generator.emitDefs(typeDialect);
                });
static GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   TypeDefGenerator generator(records, os, formatErrorIsFatal);
                   return generator.emitDecls(typeDialect);
                 });
static GenRegistration genTypeConstrDefs("gen-type-constraint-defs",
                                         "Generate type constraint definitions",
                                         [](const RecordKeeper &records,
                                            raw_ostream &os) {
                                           emitTypeConstraintDefs(records, os);
                                           return false;
                                         });
static GenRegistration
    genTypeConstrDecls("gen-type-constraint-decls",
                       "Generate type constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitTypeConstraintDecls(records, os);
                         return false;
                       });

//===----------------------------------------------------------------------===//
// Bytecode dialect generator
//===----------------------------------------------------------------------===//

static cl::OptionCategory bytecodeGenCat("Options for -gen-bytecode");
static cl::opt<std::string>
    selectedBcDialect("bytecode-dialect", cl::desc("The dialect to gen for"),
                      cl::cat(bytecodeGenCat), cl::CommaSeparated);

static GenRegistration
    genBCRW("gen-bytecode", "Generate dialect bytecode readers/writers",
            [](const RecordKeeper &records, raw_ostream &os) {
              return emitBytecodeDialect(records, selectedBcDialect, os);
            });

//===----------------------------------------------------------------------===//
// Dialect generators
//===----------------------------------------------------------------------===//

static cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
static cl::opt<std::string> selectedDialect("dialect",
                                            cl::desc("The dialect to gen for"),
                                            cl::cat(dialectGenCat),
                                            cl::CommaSeparated);

static GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return emitDialectDecls(records, selectedDialect, os);
                    });
static GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitDialectDefs(records, selectedDialect, os);
                   });

//===----------------------------------------------------------------------===//
// Dialect interface generator
//===----------------------------------------------------------------------===//

static GenRegistration genDialectInterfaceDecls(
    "gen-dialect-interface-decls", "Generate dialect interface declarations.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return DialectInterfaceGenerator(records, os).emitInterfaceDecls();
    });

//===----------------------------------------------------------------------===//
// Enum generators
//===----------------------------------------------------------------------===//

static GenRegistration
    genEnumDecls("gen-enum-decls", "Generate enum utility declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitEnumDecls(records, os);
                 });
static GenRegistration
    genEnumDefs("gen-enum-defs", "Generate enum utility definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitEnumDefs(records, os);
                });

//===----------------------------------------------------------------------===//
// Op definition generators
//===----------------------------------------------------------------------===//

static GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 std::vector<const Record *> defs = getRequestedOpDefs(records);
                 SmallVector<ArrayRef<const Record *>> shardedDefs;
                 shardOps(defs, shardedDefs);
                 return emitOpDecls(records, defs, shardedDefs.size(), os,
                                    formatErrorIsFatal);
               });
static GenRegistration
    genOpDefs("gen-op-defs", "Generate op definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                std::vector<const Record *> defs = getRequestedOpDefs(records);
                SmallVector<ArrayRef<const Record *>> shardedDefs;
                shardOps(defs, shardedDefs);
                return emitOpDefs(records, defs, shardedDefs.size(), os,
                                  formatErrorIsFatal);
              });

//===----------------------------------------------------------------------===//
// Op documentation generators
//===----------------------------------------------------------------------===//

static cl::OptionCategory
    docCat("Options for -gen-(attrdef|typedef|enum|op|dialect)-doc");
static cl::opt<std::string>
    stripPrefix("strip-prefix",
                cl::desc("Strip prefix of the fully qualified names"),
                cl::init("::mlir::"), cl::cat(docCat));
static cl::opt<bool> allowHugoSpecificFeatures(
    "allow-hugo-specific-features",
    cl::desc("Allows using features specific to Hugo"), cl::init(false),
    cl::cat(docCat));
static cl::opt<bool>
    keepOpSourceOrder("keep-op-source-order",
                      cl::desc("Do not sort ops alphabetically"),
                      cl::init(false), cl::cat(docCat));

static bool
withDialectRecords(const RecordKeeper &records,
                   llvm::function_ref<bool(const DialectRecords &)> fn) {
  auto dialectDefs = records.getAllDerivedDefinitionsIfDefined("Dialect");
  SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect =
      findDialectToGenerate(dialects, selectedDialect.getNumOccurrences() > 0
                                          ? selectedDialect.getValue()
                                          : "");
  if (!dialect)
    return true;
  std::optional<DialectRecords> filtered = collectRecords(
      records, getRequestedOpDefs(records), *dialect, keepOpSourceOrder);
  if (!filtered)
    return true;
  return fn(*filtered);
}

static GenRegistration genAttrDocRegister(
    "gen-attrdef-doc", "Generate dialect attribute documentation",
    [](const RecordKeeper &records, raw_ostream &os) {
      return withDialectRecords(records, [&](const DialectRecords &r) {
        return emitAttrDefDoc(r, os);
      });
    });
static GenRegistration genOpDocRegister(
    "gen-op-doc", "Generate dialect documentation",
    [](const RecordKeeper &records, raw_ostream &os) {
      return withDialectRecords(records, [&](const DialectRecords &r) {
        return emitOpDoc(r, stripPrefix, allowHugoSpecificFeatures, os);
      });
    });
static GenRegistration genTypeDocRegister(
    "gen-typedef-doc", "Generate dialect type documentation",
    [](const RecordKeeper &records, raw_ostream &os) {
      return withDialectRecords(records, [&](const DialectRecords &r) {
        return emitTypeDefDoc(r, os);
      });
    });
static GenRegistration genEnumDocRegister(
    "gen-enum-doc", "Generate dialect enum documentation",
    [](const RecordKeeper &records, raw_ostream &os) {
      return withDialectRecords(
          records, [&](const DialectRecords &r) { return emitEnumDoc(r, os); });
    });
static GenRegistration genDialectDocRegister(
    "gen-dialect-doc", "Generate dialect documentation",
    [](const RecordKeeper &records, raw_ostream &os) {
      return withDialectRecords(records, [&](const DialectRecords &r) {
        return emitDialectDoc(r, stripPrefix, allowHugoSpecificFeatures, os);
      });
    });

//===----------------------------------------------------------------------===//
// Op interface generators
//===----------------------------------------------------------------------===//

namespace {
template <typename GeneratorT>
struct InterfaceGenRegistration {
  InterfaceGenRegistration(StringRef genArg, StringRef genDesc)
      : genDeclArg(("gen-" + genArg + "-interface-decls").str()),
        genDefArg(("gen-" + genArg + "-interface-defs").str()),
        genDocArg(("gen-" + genArg + "-interface-docs").str()),
        genDeclDesc(("Generate " + genDesc + " interface declarations").str()),
        genDefDesc(("Generate " + genDesc + " interface definitions").str()),
        genDocDesc(("Generate " + genDesc + " interface documentation").str()),
        genDecls(genDeclArg, genDeclDesc,
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return GeneratorT(records, os).emitInterfaceDecls();
                 }),
        genDefs(genDefArg, genDefDesc,
                [](const RecordKeeper &records, raw_ostream &os) {
                  return GeneratorT(records, os).emitInterfaceDefs();
                }),
        genDocs(genDocArg, genDocDesc,
                [](const RecordKeeper &records, raw_ostream &os) {
                  return GeneratorT(records, os).emitInterfaceDocs();
                }) {}

  std::string genDeclArg, genDefArg, genDocArg;
  std::string genDeclDesc, genDefDesc, genDocDesc;
  GenRegistration genDecls, genDefs, genDocs;
};
} // namespace

static InterfaceGenRegistration<AttrInterfaceGenerator> attrGen("attr",
                                                                "attribute");
static InterfaceGenRegistration<OpInterfaceGenerator> opGen("op", "op");
static InterfaceGenRegistration<TypeInterfaceGenerator> typeGen("type", "type");

//===----------------------------------------------------------------------===//
// Python binding generators
//===----------------------------------------------------------------------===//

static cl::OptionCategory
    clOpPythonBindingCat("Options for -gen-python-op-bindings");

// dialectNameStorage is shared between gen-python-op-bindings and
// gen-python-enum-bindings via the -bind-dialect option.
static std::string dialectNameStorage;

static cl::opt<std::string, /*ExternalStorage=*/true> clDialectName(
    "bind-dialect", cl::desc("The dialect to run the generator for"),
    cl::location(dialectNameStorage), cl::cat(clOpPythonBindingCat));

static cl::opt<std::string>
    clDialectExtensionName("dialect-extension",
                           cl::desc("The prefix of the dialect extension"),
                           cl::init(""), cl::cat(clOpPythonBindingCat));

static GenRegistration genPythonEnumBindings(
    "gen-python-enum-bindings", "Generate Python bindings for enum attributes",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitPythonEnums(records, dialectNameStorage, os);
    });
static GenRegistration
    genPythonBindings("gen-python-op-bindings",
                      "Generate Python bindings for MLIR Ops",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitPythonOpBindings(records, dialectNameStorage,
                                                    clDialectExtensionName, os);
                      });

//===----------------------------------------------------------------------===//
// Pass generators
//===----------------------------------------------------------------------===//

static cl::OptionCategory
    passCAPIGenCat("Options for -gen-pass-capi-header and -gen-pass-capi-impl");
static cl::opt<std::string> passCAPIGroupName(
    "prefix",
    cl::desc("The prefix to use for this group of passes. The form will be "
             "mlirCreate<prefix><passname>, the prefix can avoid conflicts "
             "across libraries."),
    cl::cat(passCAPIGenCat));

static GenRegistration
    genCAPIHeader("gen-pass-capi-header", "Generate pass C API header",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    emitPassCAPIHeader(records, passCAPIGroupName, os);
                    return false;
                  });
static GenRegistration
    genCAPIImpl("gen-pass-capi-impl", "Generate pass C API implementation",
                [](const RecordKeeper &records, raw_ostream &os) {
                  emitPassCAPIImpl(records, passCAPIGroupName, os);
                  return false;
                });
static GenRegistration genPassDoc("gen-pass-doc", "Generate pass documentation",
                                  [](const RecordKeeper &records,
                                     raw_ostream &os) {
                                    emitPassDocs(records, os);
                                    return false;
                                  });

static cl::OptionCategory passGenCat("Options for -gen-pass-decls");
static cl::opt<std::string>
    passGroupName("name", cl::desc("The name of this group of passes"),
                  cl::cat(passGenCat));

static GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitPasses(records, passGroupName, os);
                   return false;
                 });

//===----------------------------------------------------------------------===//
// Rewriter generator
//===----------------------------------------------------------------------===//

static GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });
