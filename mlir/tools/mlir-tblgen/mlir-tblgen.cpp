//===- mlir-tblgen.cpp - Top-Level TableGen implementation for MLIR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the main function for MLIR's TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Bytecode.h"
#include "mlir/TableGen/Directive.h"
#include "mlir/TableGen/DocGenUtilities.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Pass.h"
#include "mlir/TableGen/Python.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

#include "mlir/TableGen/CAPI.h"
#include "mlir/TableGen/LLVMIR.h"
#include "mlir/TableGen/OpenMP.h"
#include "mlir/TableGen/Rewriter.h"
#include "mlir/TableGen/SPIRV.h"

using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// AttrDef registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory attrdefGenCat("Options for -gen-attrdef-*");
static llvm::cl::opt<std::string>
    attrDialect("attrdefs-dialect",
                llvm::cl::desc("Generate attributes for this dialect"),
                llvm::cl::cat(attrdefGenCat), llvm::cl::CommaSeparated);

/// Whether a failure in parsing the assembly format should be a fatal error.
static llvm::cl::opt<bool> formatErrorIsFatal(
    "asmformat-error-is-fatal",
    llvm::cl::desc("Emit a fatal error if format parsing fails"),
    llvm::cl::init(true));

static mlir::GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  tblgen::AttrDefGenerator generator(records, os,
                                                     formatErrorIsFatal);
                  return generator.emitDefs(attrDialect);
                });
static mlir::GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   tblgen::AttrDefGenerator generator(records, os,
                                                      formatErrorIsFatal);
                   return generator.emitDecls(attrDialect);
                 });

//===----------------------------------------------------------------------===//
// TypeDef registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    typeDialect("typedefs-dialect",
                llvm::cl::desc("Generate types for this dialect"),
                llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genTypeDefs("gen-typedef-defs", "Generate TypeDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  tblgen::TypeDefGenerator generator(records, os,
                                                     formatErrorIsFatal);
                  return generator.emitDefs(typeDialect);
                });
static mlir::GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   tblgen::TypeDefGenerator generator(records, os,
                                                      formatErrorIsFatal);
                   return generator.emitDecls(typeDialect);
                 });

static mlir::GenRegistration
    genTypeConstrDefs("gen-type-constraint-defs",
                      "Generate type constraint definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        tblgen::emitTypeConstraintDefs(records, os);
                        return false;
                      });
static mlir::GenRegistration
    genTypeConstrDecls("gen-type-constraint-decls",
                       "Generate type constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         tblgen::emitTypeConstraintDecls(records, os);
                         return false;
                       });

//===----------------------------------------------------------------------===//
// Bytecode registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
static cl::opt<std::string>
    selectedBcDialect("bytecode-dialect", cl::desc("The dialect to gen for"),
                      cl::cat(dialectGenCat), cl::CommaSeparated);

static mlir::GenRegistration
    genBCRW("gen-bytecode", "Generate dialect bytecode readers/writers",
            [](const RecordKeeper &records, raw_ostream &os) {
              return tblgen::emitBCRW(records, os, selectedBcDialect);
            });

//===----------------------------------------------------------------------===//
// GEN: Dialect registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return tblgen::emitDialectDecls(records, os,
                                                      selectedDialect);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return tblgen::emitDialectDefs(records, os,
                                                    selectedDialect);
                   });

//===----------------------------------------------------------------------===//
// Directive registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory
    directiveGenCat("Options for gen-directive-decl");
static llvm::cl::opt<std::string>
    dialect("directives-dialect",
            llvm::cl::desc("Generate directives for this dialect"),
            llvm::cl::cat(directiveGenCat), llvm::cl::CommaSeparated);

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration genDirectiveDecls(
    "gen-directive-decl",
    "Generate declarations for directives (OpenMP/OpenACC etc.)",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::emitDirectiveDecls(records, dialect, os);
    });

//===----------------------------------------------------------------------===//
// Python Enum registration hooks
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

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genPythonEnumBindings("gen-python-enum-bindings",
                          "Generate Python bindings for enum attributes",
                          &tblgen::emitPythonEnums);

//===----------------------------------------------------------------------===//
// Enum registration hooks
//===----------------------------------------------------------------------===//

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDecls("gen-enum-decls", "Generate enum utility declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return tblgen::emitEnumDecls(records, os);
                 });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDefs("gen-enum-defs", "Generate enum utility definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return tblgen::emitEnumDefs(records, os);
                });

//===----------------------------------------------------------------------===//
// LLVMIR registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory intrinsicGenCat("Intrinsics Generator Options");

static llvm::cl::opt<std::string>
    nameFilter("llvmir-intrinsics-filter",
               llvm::cl::desc("Only keep the intrinsics with the specified "
                              "substring in their record name"),
               llvm::cl::cat(intrinsicGenCat));

static llvm::cl::opt<std::string>
    opBaseClass("dialect-opclass-base",
                llvm::cl::desc("The base class for the ops in the dialect we "
                               "are planning to emit"),
                llvm::cl::init("LLVM_IntrOp"), llvm::cl::cat(intrinsicGenCat));

static llvm::cl::opt<std::string> accessGroupRegexp(
    "llvmir-intrinsics-access-group-regexp",
    llvm::cl::desc("Mark intrinsics that match the specified "
                   "regexp as taking an access group metadata"),
    llvm::cl::cat(intrinsicGenCat));

static llvm::cl::opt<std::string> aliasAnalysisRegexp(
    "llvmir-intrinsics-alias-analysis-regexp",
    llvm::cl::desc("Mark intrinsics that match the specified "
                   "regexp as taking alias.scopes, noalias, and tbaa metadata"),
    llvm::cl::cat(intrinsicGenCat));

template <bool ConvertTo>
static bool emitLLVMIREnumConversionDefs(const RecordKeeper &records,
                                         raw_ostream &os) {
  for (const Record *def : records.getAllDerivedDefinitions("LLVM_EnumAttr"))
    if (ConvertTo)
      tblgen::emitOneEnumToConversion(def, os);
    else
      tblgen::emitOneEnumFromConversion(def, os);

  for (const Record *def : records.getAllDerivedDefinitions("LLVM_CEnumAttr"))
    if (ConvertTo)
      tblgen::emitOneCEnumToConversion(def, os);
    else
      tblgen::emitOneCEnumFromConversion(def, os);

  return false;
}

static mlir::GenRegistration
    genLLVMIRConversions("gen-llvmir-conversions",
                         "Generate LLVM IR conversions",
                         tblgen::emitLLVMIRConversionBuilders);

static mlir::GenRegistration genOpFromLLVMIRConversions(
    "gen-op-from-llvmir-conversions",
    "Generate conversions of operations from LLVM IR",
    tblgen::emitLLVMIROpMLIRBuilders);

static mlir::GenRegistration genIntrFromLLVMIRConversions(
    "gen-intr-from-llvmir-conversions",
    "Generate conversions of intrinsics from LLVM IR",
    tblgen::emitLLVMIRIntrMLIRBuilders);

static mlir::GenRegistration
    genEnumToLLVMConversion("gen-enum-to-llvmir-conversions",
                            "Generate conversions of EnumAttrs to LLVM IR",
                            emitLLVMIREnumConversionDefs</*ConvertTo=*/true>);

static mlir::GenRegistration genEnumFromLLVMConversion(
    "gen-enum-from-llvmir-conversions",
    "Generate conversions of EnumAttrs from LLVM IR",
    emitLLVMIREnumConversionDefs</*ConvertTo=*/false>);

static mlir::GenRegistration genConvertibleLLVMIRIntrinsics(
    "gen-convertible-llvmir-intrinsics",
    "Generate list of convertible LLVM IR intrinsics",
    tblgen::emitConvertibleLLVMIRIntrinsics);

static mlir::GenRegistration
    genLLVMIRIntrinsics("gen-llvmir-intrinsics", "Generate LLVM IR intrinsics",
                        [](const RecordKeeper &records, raw_ostream &os) {
                          return tblgen::emitLLVMIRIntrinsics(
                              records, os, nameFilter, accessGroupRegexp,
                              aliasAnalysisRegexp, opBaseClass);
                        });

//===----------------------------------------------------------------------===//
// OpenMP registration hooks
//===----------------------------------------------------------------------===//

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration
    verifyOpenmpOps("verify-openmp-ops",
                    "Verify OpenMP operations (produce no output file)",
                    tblgen::verifyOpenmpDecls);

static mlir::GenRegistration
    regOpenmpClauseOps("gen-openmp-clause-ops",
                       "Generate OpenMP clause operand structures",
                       tblgen::genOpenmpClauseOps);

//===----------------------------------------------------------------------===//
// OpDefinition registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return tblgen::emitOpDecls(records, os, formatErrorIsFatal,
                                            opIncFilter, opExcFilter,
                                            opShardCount);
               });

static mlir::GenRegistration
    genOpDefs("gen-op-defs", "Generate op definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                return tblgen::emitOpDefs(records, os, formatErrorIsFatal,
                                          opIncFilter, opExcFilter,
                                          opShardCount);
              });
//===----------------------------------------------------------------------===//
// Op Doc Registration
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

static mlir::GenRegistration
    genAttrDocRegister("gen-attrdef-doc",
                       "Generate dialect attribute documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         tblgen::emitAttrOrTypeDefDoc(records, os, "AttrDef");
                         return false;
                       });

static mlir::GenRegistration
    genOpDocRegister("gen-op-doc", "Generate dialect documentation",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       tblgen::emitOpDoc(records, os, stripPrefix,
                                         allowHugoSpecificFeatures, opIncFilter,
                                         opExcFilter);
                       return false;
                     });

static mlir::GenRegistration
    genTypeDocRegister("gen-typedef-doc", "Generate dialect type documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         tblgen::emitAttrOrTypeDefDoc(records, os, "TypeDef");
                         return false;
                       });

static mlir::GenRegistration
    genEnumDocRegister("gen-enum-doc", "Generate dialect enum documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         tblgen::emitEnumDoc(records, os);
                         return false;
                       });

static mlir::GenRegistration genDialectDocRegister(
    "gen-dialect-doc", "Generate dialect documentation",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::emitDialectDoc(records, os, selectedDialect, stripPrefix,
                                    allowHugoSpecificFeatures, opIncFilter,
                                    opExcFilter);
    });

//===----------------------------------------------------------------------===//
// Interface registration hooks
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
  mlir::GenRegistration genDecls, genDefs, genDocs;
};
} // namespace

static InterfaceGenRegistration<tblgen::AttrInterfaceGenerator>
    attrInterfaceGen("attr", "attribute");
static InterfaceGenRegistration<tblgen::OpInterfaceGenerator>
    opInterfaceGen("op", "op");
static InterfaceGenRegistration<tblgen::TypeInterfaceGenerator>
    typeInterfaceGen("type", "type");

//===----------------------------------------------------------------------===//
// Python bindings registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory
    clOpPythonBindingCat("Options for -gen-python-op-bindings");

static llvm::cl::opt<std::string>
    clDialectName("bind-dialect",
                  llvm::cl::desc("The dialect to run the generator for"),
                  llvm::cl::init(""), llvm::cl::cat(clOpPythonBindingCat));

static llvm::cl::opt<std::string> clDialectExtensionName(
    "dialect-extension", llvm::cl::desc("The prefix of the dialect extension"),
    llvm::cl::init(""), llvm::cl::cat(clOpPythonBindingCat));

static GenRegistration
    genPythonBindings("gen-python-op-bindings",
                      "Generate Python bindings for MLIR Ops",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return tblgen::emitAllPythonOps(
                            records, os, clDialectName, clDialectExtensionName);
                      });

//===----------------------------------------------------------------------===//
// Pass CAPI registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory
    passGenCat("Options for -gen-pass-capi-header and -gen-pass-capi-impl");
static llvm::cl::opt<std::string>
    groupName("prefix",
              llvm::cl::desc("The prefix to use for this group of passes. The "
                             "form will be mlirCreate<prefix><passname>, the "
                             "prefix can avoid conflicts across libraries."),
              llvm::cl::cat(passGenCat));

static mlir::GenRegistration
    genPassCAPIHeader("gen-pass-capi-header", "Generate pass C API header",

                      [](const RecordKeeper &records, raw_ostream &os) {
                        return tblgen::emitPasssCAPIHeader(records, os,
                                                           groupName);
                      });

static mlir::GenRegistration
    genPassCAPIImpl("gen-pass-capi-impl", "Generate pass C API implementation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return tblgen::emitPassCAPIImpl(records, os, groupName);
                    });

//===----------------------------------------------------------------------===//
// Pass Doc registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genPassDocRegister("gen-pass-doc", "Generate pass documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         tblgen::emitPassDocs(records, os);
                         return false;
                       });

//===----------------------------------------------------------------------===//
// Pass registration hooks
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory passDeclsGenCat("Options for -gen-pass-decls");
static llvm::cl::opt<std::string>
    groupNamePassDecls("name",
                       llvm::cl::desc("The name of this group of passes"),
                       llvm::cl::cat(passDeclsGenCat));

static mlir::GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   tblgen::emitPassDecls(records, os, groupNamePassDecls);
                   return false;
                 });

//===----------------------------------------------------------------------===//
// Rewriter registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   tblgen::emitRewriters(records, os);
                   return false;
                 });

//===----------------------------------------------------------------------===//
// SPIRV registration hooks
//===----------------------------------------------------------------------===//

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration genSPIRVInterfaceDecls(
    "gen-avail-interface-decls", "Generate availability interface declarations",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::emitSPRIVInterfaceDecls(records, os);
    });

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVInterfaceDefs("gen-avail-interface-defs",
                          "Generate op interface definitions",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return tblgen::emitSPRIVInterfaceDefs(records, os);
                          });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVEnumDecls("gen-spirv-enum-avail-decls",
                      "Generate SPIR-V enum availability declarations",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return tblgen::emitSPRIVEnumDecls(records, os);
                      });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVEnumDefs("gen-spirv-enum-avail-defs",
                     "Generate SPIR-V enum availability definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return tblgen::emitSPIRVEnumDefs(records, os);
                     });

static mlir::GenRegistration genSPIRVSerialization(
    "gen-spirv-serialization",
    "Generate SPIR-V (de)serialization utilities and functions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::emitSPIRVSerializationFns(records, os);
    });

static mlir::GenRegistration
    genSPIRVAttrUtils("gen-spirv-attr-utils",
                      "Generate SPIR-V attribute utility definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return tblgen::emitSPIRVAttrUtils(records, os);
                      });

static mlir::GenRegistration genSPIRVAvailabilityImpl(
    "gen-spirv-avail-impls", "Generate SPIR-V operation utility definitions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::emitSPIRVAvailabilityImpl(records, os);
    });

static mlir::GenRegistration genSPIRVCapabilityImplication(
    "gen-spirv-capability-implication",
    "Generate utility function to return implied "
    "capabilities for a given capability",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::emitSPIRVCapabilityImplication(records, os);
    });

//===----------------------------------------------------------------------===//
// main registration hooks
//===----------------------------------------------------------------------===//

// Generator that prints records.
static GenRegistration
    printRecords("print-records", "Print all records to stdout",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   os << records;
                   return false;
                 });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
