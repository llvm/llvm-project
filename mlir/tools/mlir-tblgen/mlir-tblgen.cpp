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
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Python.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

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

static mlir::GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  tblgen::AttrDefGenerator generator(records, os);
                  return generator.emitDefs(attrDialect);
                });
static mlir::GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   tblgen::AttrDefGenerator generator(records, os);
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
                  tblgen::TypeDefGenerator generator(records, os);
                  return generator.emitDefs(typeDialect);
                });
static mlir::GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   tblgen::TypeDefGenerator generator(records, os);
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

static mlir::GenRegistration
    genBCRW("gen-bytecode", "Generate dialect bytecode readers/writers",
            [](const RecordKeeper &records, raw_ostream &os) {
              return tblgen::emitBCRW(records, os);
            });

//===----------------------------------------------------------------------===//
// GEN: Dialect registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return tblgen::emitDialectDecls(records, os);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return tblgen::emitDialectDefs(records, os);
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

static mlir::GenRegistration
    genLLVMIRConversions("gen-llvmir-conversions",
                         "Generate LLVM IR conversions",
                         emitLLVMIRConversionBuilders);

static mlir::GenRegistration genOpFromLLVMIRConversions(
    "gen-op-from-llvmir-conversions",
    "Generate conversions of operations from LLVM IR",
    emitLLVMIROpMLIRBuilders);

static mlir::GenRegistration genIntrFromLLVMIRConversions(
    "gen-intr-from-llvmir-conversions",
    "Generate conversions of intrinsics from LLVM IR",
    emitLLVMIRIntrMLIRBuilders);

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
    emitConvertibleLLVMIRIntrinsics);

static mlir::GenRegistration genLLVMIRIntrinsics("gen-llvmir-intrinsics",
                                                 "Generate LLVM IR intrinsics",
                                                 emitLLVMIRIntrinsics);

//===----------------------------------------------------------------------===//
// OpenMP registration hooks
//===----------------------------------------------------------------------===//

// Registers the generator to mlir-tblgen.
static mlir::GenRegistration
    verifyOpenmpOps("verify-openmp-ops",
                    "Verify OpenMP operations (produce no output file)",
                    verifyOpenmpDecls);

static mlir::GenRegistration
    regOpenmpClauseOps("gen-openmp-clause-ops",
                       "Generate OpenMP clause operand structures",
                       genOpenmpClauseOps);

//===----------------------------------------------------------------------===//
// OpDefinition registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpDecls(records, os);
               });

static mlir::GenRegistration genOpDefs("gen-op-defs", "Generate op definitions",
                                       [](const RecordKeeper &records,
                                          raw_ostream &os) {
                                         return emitOpDefs(records, os);
                                       });
//===----------------------------------------------------------------------===//
// Op Doc Registration
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genAttrDocRegister("gen-attrdef-doc",
                       "Generate dialect attribute documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitAttrOrTypeDefDoc(records, os, "AttrDef");
                         return false;
                       });

static mlir::GenRegistration
    genOpDocRegister("gen-op-doc", "Generate dialect documentation",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       emitOpDoc(records, os);
                       return false;
                     });

static mlir::GenRegistration
    genTypeDocRegister("gen-typedef-doc", "Generate dialect type documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitAttrOrTypeDefDoc(records, os, "TypeDef");
                         return false;
                       });

static mlir::GenRegistration
    genEnumDocRegister("gen-enum-doc", "Generate dialect enum documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitEnumDoc(records, os);
                         return false;
                       });

static mlir::GenRegistration
    genDialectDocRegister("gen-dialect-doc", "Generate dialect documentation",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return emitDialectDoc(records, os);
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

static GenRegistration
    genPythonBindings("gen-python-op-bindings",
                      "Generate Python bindings for MLIR Ops",
                      &tblgen::emitAllPythonOps);

//===----------------------------------------------------------------------===//
// Pass CAPI registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration genCAPIHeader("gen-pass-capi-header",
                                           "Generate pass C API header",
                                           &emitCAPIHeader);

static mlir::GenRegistration genCAPIImpl("gen-pass-capi-impl",
                                         "Generate pass C API implementation",
                                         &emitCAPIImpl);

//===----------------------------------------------------------------------===//
// Pass Doc registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genPassDocRegister("gen-pass-doc", "Generate pass documentation",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitPassDocs(records, os);
                         return false;
                       });

//===----------------------------------------------------------------------===//
// Pass registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitPassDecls(records, os);
                   return false;
                 });

//===----------------------------------------------------------------------===//
// Rewriter registration hooks
//===----------------------------------------------------------------------===//

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });

//===----------------------------------------------------------------------===//
// SPIRV registration hooks
//===----------------------------------------------------------------------===//

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVInterfaceDecls("gen-avail-interface-decls",
                           "Generate availability interface declarations",
                           [](const RecordKeeper &records, raw_ostream &os) {
                             return emitSPRIVInterfaceDecls(records, os);
                           });

// Registers the operation interface generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVInterfaceDefs("gen-avail-interface-defs",
                          "Generate op interface definitions",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return emitSPRIVInterfaceDefs(records, os);
                          });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVEnumDecls("gen-spirv-enum-avail-decls",
                      "Generate SPIR-V enum availability declarations",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitSPRIVEnumDecls(records, os);
                      });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSPIRVEnumDefs("gen-spirv-enum-avail-defs",
                     "Generate SPIR-V enum availability definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       return emitSPIRVEnumDefs(records, os);
                     });

static mlir::GenRegistration genSPIRVSerialization(
    "gen-spirv-serialization",
    "Generate SPIR-V (de)serialization utilities and functions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSPIRVSerializationFns(records, os);
    });

static mlir::GenRegistration
    genSPIRVAttrUtils("gen-spirv-attr-utils",
                      "Generate SPIR-V attribute utility definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return emitSPIRVAttrUtils(records, os);
                      });

static mlir::GenRegistration
    genSPIRVAvailabilityImpl("gen-spirv-avail-impls",
                             "Generate SPIR-V operation utility definitions",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               return emitSPIRVAvailabilityImpl(records, os);
                             });

static mlir::GenRegistration genSPIRVCapabilityImplication(
    "gen-spirv-capability-implication",
    "Generate utility function to return implied "
    "capabilities for a given capability",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSPIRVCapabilityImplication(records, os);
    });

//===----------------------------------------------------------------------===//
// main registration hooks
//===----------------------------------------------------------------------===//

// Generator that prints records.
GenRegistration printRecords("print-records", "Print all records to stdout",
                             [](const RecordKeeper &records, raw_ostream &os) {
                               os << records;
                               return false;
                             });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
