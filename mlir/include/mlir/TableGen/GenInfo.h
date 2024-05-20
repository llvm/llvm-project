//===- GenInfo.h - Generator info -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENINFO_H_
#define MLIR_TABLEGEN_GENINFO_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include <functional>
#include <utility>

namespace llvm {
class RecordKeeper;
} // namespace llvm

namespace mlir {

/// Generator function to invoke.
using GenFunction = std::function<bool(const llvm::RecordKeeper &recordKeeper,
                                       raw_ostream &os)>;

/// Structure to group information about a generator (argument to invoke via
/// mlir-tblgen, description, and generator function).
class GenInfo {
public:
  /// GenInfo constructor should not be invoked directly, instead use
  /// GenRegistration or registerGen.
  GenInfo(StringRef arg, StringRef description, GenFunction generator)
      : arg(arg), description(description), generator(std::move(generator)) {}

  /// Invokes the generator and returns whether the generator failed.
  bool invoke(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) const {
    assert(generator && "Cannot call generator with null generator");
    return generator(recordKeeper, os);
  }

  /// Returns the command line option that may be passed to 'mlir-tblgen' to
  /// invoke this generator.
  StringRef getGenArgument() const { return arg; }

  /// Returns a description for the generator.
  StringRef getGenDescription() const { return description; }

private:
  // The argument with which to invoke the generator via mlir-tblgen.
  StringRef arg;

  // Description of the generator.
  StringRef description;

  // Generator function.
  GenFunction generator;
};

/// GenRegistration provides a global initializer that registers a generator
/// function.
///
/// Usage:
///
///   // At namespace scope.
///   static GenRegistration Print("print", "Print records", [](...){...});
struct GenRegistration {
  GenRegistration(StringRef arg, StringRef description,
                  const GenFunction &function);
};

namespace tblgen {
bool emitBCRW(const llvm::RecordKeeper &records, raw_ostream &os,
              const std::string &selectedBcDialect);

bool emitOpDecls(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                 const std::string &opIncFilter, const std::string &opExcFilter,
                 unsigned opShardCount, bool formatErrorIsFatal);
bool emitOpDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                const std::string &opIncFilter, const std::string &opExcFilter,
                unsigned opShardCount, bool formatErrorIsFatal);

bool emitDialectDecls(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                      const std::string &selectedDialect);
bool emitDialectDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                     const std::string &selectedDialect);
bool emitDirectiveDecls(const llvm::RecordKeeper &recordKeeper,
                        llvm::StringRef dialect, raw_ostream &os);

bool emitPythonEnums(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);
bool emitAllPythonOps(const llvm::RecordKeeper &records, raw_ostream &os,
                      const std::string &clDialectName,
                      const std::string &clDialectExtensionName);

bool emitEnumDecls(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);
bool emitEnumDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);

bool emitLLVMBuilders(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);
bool emitLLVMOpMLIRBuilders(const llvm::RecordKeeper &recordKeeper,
                            raw_ostream &os);
bool emitLLVMIntrMLIRBuilders(const llvm::RecordKeeper &recordKeeper,
                              raw_ostream &os);
template <bool ConvertTo>
bool emitLLVMEnumConversionDefs(const llvm::RecordKeeper &recordKeeper,
                                raw_ostream &os);
bool emitLLVMConvertibleIntrinsics(const llvm::RecordKeeper &recordKeeper,
                                   raw_ostream &os);
bool emitLLVMIntrinsics(const llvm::RecordKeeper &records,
                        llvm::raw_ostream &os, const std::string &nameFilter,
                        const std::string &accessGroupRegexp,
                        const std::string &aliasAnalysisRegexp,
                        const std::string &opBaseClass);

void emitAttrOrTypeDefDoc(const llvm::RecordKeeper &recordKeeper,
                          raw_ostream &os, StringRef recordTypeName);
void emitOpDoc(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
               const std::string &emitOpDoc, bool allowHugoSpecificFeatures,
               const std::string &opIncFilter, const std::string &opExcFilter);
bool emitDialectDoc(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                    const std::string &selectedDialect,
                    const std::string &opIncFilter,
                    const std::string &opExcFilter,
                    const std::string &stripPrefix,
                    bool allowHugoSpecificFeatures);
void emitDocs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);

bool emitCAPIHeader(const llvm::RecordKeeper &records, raw_ostream &os,
                    std::string groupPrefix);
bool emitCAPIImpl(const llvm::RecordKeeper &records, raw_ostream &os,
                  std::string groupPrefix);
void emitPasses(const llvm::RecordKeeper &recordKeeper, raw_ostream &os,
                const std::string &opIncFilter, const std::string &groupName);
void emitRewriters(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);

bool emitSPIRVInterfaceDefs(const llvm::RecordKeeper &recordKeeper,
                            raw_ostream &os);
bool emitSPIRVInterfaceDecls(const llvm::RecordKeeper &recordKeeper,
                             raw_ostream &os);
bool emitSPIRVEnumDecls(const llvm::RecordKeeper &recordKeeper,
                        raw_ostream &os);
bool emitSPIRVEnumDefs(const llvm::RecordKeeper &recordKeeper, raw_ostream &os);
bool emitSPIRVCapabilityImplication(const llvm::RecordKeeper &recordKeeper,
                                    raw_ostream &os);
bool emitSPIRVSerializationFns(const llvm::RecordKeeper &recordKeeper,
                               raw_ostream &os);
bool emitSPIRVAttrUtils(const llvm::RecordKeeper &recordKeeper,
                        raw_ostream &os);
bool emitSPIRVAvailabilityImpl(const llvm::RecordKeeper &recordKeeper,
                               raw_ostream &os);

} // namespace tblgen

} // namespace mlir

#endif // MLIR_TABLEGEN_GENINFO_H_
