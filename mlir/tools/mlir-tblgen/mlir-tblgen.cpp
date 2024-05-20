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
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpInterfacesGen.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Commandline Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> formatErrorIsFatal(
    "asmformat-error-is-fatal",
    llvm::cl::desc("Emit a fatal error if format parsing fails"),
    llvm::cl::init(true));

cl::OptionCategory opDefGenCat("Options for op definition generators");

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

//===----------------------------------------------------------------------===//
// GEN: Registration hooks
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory attrdefGenCat("Options for -gen-attrdef-*");
static llvm::cl::opt<std::string>
    attrDialect("attrdefs-dialect",
                llvm::cl::desc("Generate attributes for this dialect"),
                llvm::cl::cat(attrdefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  AttrDefGenerator generator(records, os, formatErrorIsFatal);
                  return generator.emitDefs(attrDialect);
                });
static mlir::GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   AttrDefGenerator generator(records, os, formatErrorIsFatal);
                   return generator.emitDecls(attrDialect);
                 });

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    typeDialect("typedefs-dialect",
                llvm::cl::desc("Generate types for this dialect"),
                llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genTypeDefs("gen-typedef-defs", "Generate TypeDef definitions",
                [](const llvm::RecordKeeper &records, raw_ostream &os) {
                  TypeDefGenerator generator(records, os, formatErrorIsFatal);
                  return generator.emitDefs(typeDialect);
                });
static mlir::GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   TypeDefGenerator generator(records, os, formatErrorIsFatal);
                   return generator.emitDecls(typeDialect);
                 });
static mlir::GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpDecls(records, os, opIncFilter, opExcFilter,
                                    opShardCount, formatErrorIsFatal);
               });

//===----------------------------------------------------------------------===//
// OpDef
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genOpDefs("gen-op-defs", "Generate op definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                return emitOpDefs(records, os, opIncFilter, opExcFilter,
                                  opShardCount, formatErrorIsFatal);
              });

//===----------------------------------------------------------------------===//
// Bytecode
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory
    byteCodeDialectGenCat("Options for -gen-bytecode");
static llvm::cl::opt<std::string> selectedBcDialect(
    "bytecode-dialect", llvm::cl::desc("The dialect to gen for"),
    llvm::cl::cat(byteCodeDialectGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genBCRW("gen-bytecode", "Generate dialect bytecode readers/writers",
            [](const RecordKeeper &records, raw_ostream &os) {
              return emitBCRW(records, os, selectedBcDialect);
            });

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

static llvm::cl::OptionCategory
    directiveGenCat("Options for gen-directive-decl");

static llvm::cl::opt<std::string>
    dialect("directives-dialect",
            llvm::cl::desc("Generate directives for this dialect"),
            llvm::cl::cat(directiveGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const llvm::RecordKeeper &records, raw_ostream &os) {
                      return emitDialectDecls(records, os, selectedDialect);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const llvm::RecordKeeper &records, raw_ostream &os) {
                     return emitDialectDefs(records, os, selectedDialect);
                   });

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration genDirectiveDecls(
    "gen-directive-decl",
    "Generate declarations for directives (OpenMP/OpenACC etc.)",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitDirectiveDecls(records, dialect, os);
    });

//===----------------------------------------------------------------------===//
// Python
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

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genPythonEnumBindings("gen-python-enum-bindings",
                          "Generate Python bindings for enum attributes",
                          &emitPythonEnums);

static GenRegistration
    genPythonBindings("gen-python-op-bindings",
                      "Generate Python bindings for MLIR Ops",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        if (clDialectName.empty())
                          llvm::PrintFatalError("dialect name not provided");
                        return emitAllPythonOps(records, os, clDialectName,
                                                clDialectExtensionName);
                      });

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDecls("gen-enum-decls", "Generate enum utility declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitEnumDecls(records, os);
                 });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDefs("gen-enum-defs", "Generate enum utility definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitEnumDefs(records, os);
                });

//===----------------------------------------------------------------------===//
// LLVMIRConversion
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

static mlir::GenRegistration
    genLLVMIRConversions("gen-llvmir-conversions",
                         "Generate LLVM IR conversions", emitLLVMBuilders);

static mlir::GenRegistration genOpFromLLVMIRConversions(
    "gen-op-from-llvmir-conversions",
    "Generate conversions of operations from LLVM IR", emitLLVMOpMLIRBuilders);

static mlir::GenRegistration genIntrFromLLVMIRConversions(
    "gen-intr-from-llvmir-conversions",
    "Generate conversions of intrinsics from LLVM IR",
    emitLLVMIntrMLIRBuilders);

static mlir::GenRegistration
    genEnumToLLVMConversion("gen-enum-to-llvmir-conversions",
                            "Generate conversions of EnumAttrs to LLVM IR",
                            emitLLVMEnumConversionDefs</*ConvertTo=*/true>);

static mlir::GenRegistration
    genEnumFromLLVMConversion("gen-enum-from-llvmir-conversions",
                              "Generate conversions of EnumAttrs from LLVM IR",
                              emitLLVMEnumConversionDefs</*ConvertTo=*/false>);

static mlir::GenRegistration genConvertibleLLVMIRIntrinsics(
    "gen-convertible-llvmir-intrinsics",
    "Generate list of convertible LLVM IR intrinsics",
    emitLLVMConvertibleIntrinsics);

static mlir::GenRegistration genLLVMIRIntrinsics(
    "gen-llvmir-intrinsics", "Generate LLVM IR intrinsics",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitLLVMIntrinsics(records, os, nameFilter, accessGroupRegexp,
                                aliasAnalysisRegexp, opBaseClass);
    });

//===----------------------------------------------------------------------===//
// Docs
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory
    docCat("Options for -gen-(attrdef|typedef|op|dialect)-doc");
llvm::cl::opt<std::string>
    stripPrefix("strip-prefix",
                llvm::cl::desc("Strip prefix of the fully qualified names"),
                llvm::cl::init("::mlir::"), llvm::cl::cat(docCat));
llvm::cl::opt<bool> allowHugoSpecificFeatures(
    "allow-hugo-specific-features",
    llvm::cl::desc("Allows using features specific to Hugo"),
    llvm::cl::init(false), llvm::cl::cat(docCat));

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
                    emitOpDoc(records, os, stripPrefix,
                              allowHugoSpecificFeatures, opIncFilter,
                              opExcFilter);
                    return false;
                  });

static mlir::GenRegistration
    genTypeRegister("gen-typedef-doc", "Generate dialect type documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      emitAttrOrTypeDefDoc(records, os, "TypeDef");
                      return false;
                    });

static mlir::GenRegistration
    genDialectDocRegister("gen-dialect-doc", "Generate dialect documentation",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return emitDialectDoc(records, os, selectedDialect,
                                                  opIncFilter, opExcFilter,
                                                  stripPrefix,
                                                  allowHugoSpecificFeatures);
                          });

static mlir::GenRegistration
    genPassDocRegister("gen-pass-doc", "Generate pass documentation",
                       [](const llvm::RecordKeeper &records, raw_ostream &os) {
                         emitDocs(records, os);
                         return false;
                       });
//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

static mlir::tblgen::InterfaceGenRegistration<
    mlir::tblgen::AttrInterfaceGenerator>
    attrGen("attr", "attribute");
static InterfaceGenRegistration<OpInterfaceGenerator> opGen("op", "op");
static InterfaceGenRegistration<TypeInterfaceGenerator> typeGen("type", "type");

//===----------------------------------------------------------------------===//
// CAPI
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory
    passGenCat("Options for -gen-pass-capi-header and -gen-pass-capi-impl");
static llvm::cl::opt<std::string> groupPrefix(
    "prefix",
    llvm::cl::desc("The prefix to use for this group of passes. The "
                   "form will be mlirCreate<prefix><passname>, the "
                   "prefix can avoid conflicts across libraries."),
    llvm::cl::cat(passGenCat));
static mlir::GenRegistration
    genCAPIHeader("gen-pass-capi-header", "Generate pass C API header",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    return emitCAPIHeader(records, os, groupPrefix.getValue());
                  });

static mlir::GenRegistration
    genCAPIImpl("gen-pass-capi-impl", "Generate pass C API implementation",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitCAPIImpl(records, os, groupPrefix.getValue());
                });

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string>
    groupName("name", llvm::cl::desc("The name of this group of passes"),
              llvm::cl::cat(passGenCat));
static mlir::GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const llvm::RecordKeeper &records, raw_ostream &os) {
                   emitPasses(records, os, opIncFilter, groupName);
                   return false;
                 });

//===----------------------------------------------------------------------===//
// Rewriter
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDecls("gen-avail-interface-decls",
                      "Generate availability interface declarations",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitSPIRVInterfaceDecls(records, os);
                      });

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genInterfaceDefs("gen-avail-interface-defs",
                     "Generate op interface definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return emitSPIRVInterfaceDefs(records, os);
                     });

//===----------------------------------------------------------------------===//
// SPIRV
//===----------------------------------------------------------------------===//

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVEnumDecls("gen-spirv-enum-avail-decls",
                      "Generate SPIR-V enum availability declarations",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitSPIRVEnumDecls(records, os);
                      });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVEnumDefs("gen-spirv-enum-avail-defs",
                     "Generate SPIR-V enum availability definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return emitSPIRVEnumDefs(records, os);
                     });

static mlir::GenRegistration genSPIRVCapabilityImplication(
    "gen-spirv-capability-implication",
    "Generate utility function to return implied "
    "capabilities for a given capability",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSPIRVCapabilityImplication(records, os);
    });

static mlir::GenRegistration genSPIRVSerialization(
    "gen-spirv-serialization",
    "Generate SPIR-V (de)serialization utilities and functions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSPIRVSerializationFns(records, os);
    });

static mlir::GenRegistration
    genSPIRVOpUtils("gen-spirv-attr-utils",
                    "Generate SPIR-V attribute utility definitions",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return emitSPIRVAttrUtils(records, os);
                    });

static mlir::GenRegistration genSPIRVOpAvailabilityImpl(
    "gen-spirv-avail-impls", "Generate SPIR-V operation utility definitions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSPIRVAvailabilityImpl(records, os);
    });

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

// Generator that prints records.
GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
