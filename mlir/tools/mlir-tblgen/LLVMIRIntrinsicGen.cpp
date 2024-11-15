//===- LLVMIntrinsicGen.cpp - TableGen utility for converting intrinsics --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a TableGen generator that converts TableGen definitions for LLVM
// intrinsics to TableGen definitions for MLIR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::Record;
using llvm::RecordKeeper;
using llvm::Regex;
using namespace mlir;

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

// Used to represent the indices of overloadable operands/results.
using IndicesTy = llvm::SmallBitVector;

/// Return a CodeGen value type entry from a type record.
static llvm::MVT::SimpleValueType getValueType(const Record *rec) {
  return (llvm::MVT::SimpleValueType)rec->getValueAsDef("VT")->getValueAsInt(
      "Value");
}

/// Return the indices of the definitions in a list of definitions that
/// represent overloadable types
static IndicesTy getOverloadableTypeIdxs(const Record &record,
                                         const char *listName) {
  auto results = record.getValueAsListOfDefs(listName);
  IndicesTy overloadedOps(results.size());
  for (const auto &r : llvm::enumerate(results)) {
    llvm::MVT::SimpleValueType vt = getValueType(r.value());
    switch (vt) {
    case llvm::MVT::iAny:
    case llvm::MVT::fAny:
    case llvm::MVT::Any:
    case llvm::MVT::iPTRAny:
    case llvm::MVT::vAny:
      overloadedOps.set(r.index());
      break;
    default:
      continue;
    }
  }
  return overloadedOps;
}

namespace {
/// A wrapper for LLVM's Tablegen class `Intrinsic` that provides accessors to
/// the fields of the record.
class LLVMIntrinsic {
public:
  LLVMIntrinsic(const Record &record) : record(record) {}

  /// Get the name of the operation to be used in MLIR.  Uses the appropriate
  /// field if not empty, constructs a name by replacing underscores with dots
  /// in the record name otherwise.
  std::string getOperationName() const {
    StringRef name = record.getValueAsString(fieldName);
    if (!name.empty())
      return name.str();

    name = record.getName();
    assert(name.starts_with("int_") &&
           "LLVM intrinsic names are expected to start with 'int_'");
    name = name.drop_front(4);
    SmallVector<StringRef, 8> chunks;
    StringRef targetPrefix = record.getValueAsString("TargetPrefix");
    name.split(chunks, '_');
    auto *chunksBegin = chunks.begin();
    // Remove the target prefix from target specific intrinsics.
    if (!targetPrefix.empty()) {
      assert(targetPrefix == *chunksBegin &&
             "Intrinsic has TargetPrefix, but "
             "record name doesn't begin with it");
      assert(chunks.size() >= 2 &&
             "Intrinsic has TargetPrefix, but "
             "chunks has only one element meaning the intrinsic name is empty");
      ++chunksBegin;
    }
    return llvm::join(chunksBegin, chunks.end(), ".");
  }

  /// Get the name of the record without the "intrinsic" prefix.
  StringRef getProperRecordName() const {
    StringRef name = record.getName();
    assert(name.starts_with("int_") &&
           "LLVM intrinsic names are expected to start with 'int_'");
    return name.drop_front(4);
  }

  /// Get the number of operands.
  unsigned getNumOperands() const {
    auto operands = record.getValueAsListOfDefs(fieldOperands);
    assert(llvm::all_of(
               operands,
               [](const Record *r) { return r->isSubClassOf("LLVMType"); }) &&
           "expected operands to be of LLVM type");
    return operands.size();
  }

  /// Get the number of results.  Note that LLVM does not support multi-value
  /// operations so, in fact, multiple results will be returned as a value of
  /// structure type.
  unsigned getNumResults() const {
    auto results = record.getValueAsListOfDefs(fieldResults);
    for (const Record *r : results) {
      (void)r;
      assert(r->isSubClassOf("LLVMType") &&
             "expected operands to be of LLVM type");
    }
    return results.size();
  }

  /// Return true if the intrinsic may have side effects, i.e. does not have the
  /// `IntrNoMem` property.
  bool hasSideEffects() const {
    return llvm::none_of(
        record.getValueAsListOfDefs(fieldTraits),
        [](const Record *r) { return r->getName() == "IntrNoMem"; });
  }

  /// Return true if the intrinsic is commutative, i.e. has the respective
  /// property.
  bool isCommutative() const {
    return llvm::any_of(
        record.getValueAsListOfDefs(fieldTraits),
        [](const Record *r) { return r->getName() == "Commutative"; });
  }

  IndicesTy getOverloadableOperandsIdxs() const {
    return getOverloadableTypeIdxs(record, fieldOperands);
  }

  IndicesTy getOverloadableResultsIdxs() const {
    return getOverloadableTypeIdxs(record, fieldResults);
  }

private:
  /// Names of the fields in the Intrinsic LLVM Tablegen class.
  const char *fieldName = "LLVMName";
  const char *fieldOperands = "ParamTypes";
  const char *fieldResults = "RetTypes";
  const char *fieldTraits = "IntrProperties";

  const Record &record;
};
} // namespace

/// Prints the elements in "range" separated by commas and surrounded by "[]".
template <typename Range>
void printBracketedRange(const Range &range, llvm::raw_ostream &os) {
  os << '[';
  llvm::interleaveComma(range, os);
  os << ']';
}

/// Emits ODS (TableGen-based) code for `record` representing an LLVM intrinsic.
/// Returns true on error, false on success.
static bool emitIntrinsic(const Record &record, llvm::raw_ostream &os) {
  LLVMIntrinsic intr(record);

  Regex accessGroupMatcher(accessGroupRegexp);
  bool requiresAccessGroup =
      !accessGroupRegexp.empty() && accessGroupMatcher.match(record.getName());

  Regex aliasAnalysisMatcher(aliasAnalysisRegexp);
  bool requiresAliasAnalysis = !aliasAnalysisRegexp.empty() &&
                               aliasAnalysisMatcher.match(record.getName());

  // Prepare strings for traits, if any.
  SmallVector<StringRef, 2> traits;
  if (intr.isCommutative())
    traits.push_back("Commutative");
  if (!intr.hasSideEffects())
    traits.push_back("NoMemoryEffect");

  // Prepare strings for operands.
  SmallVector<StringRef, 8> operands(intr.getNumOperands(), "LLVM_Type");
  if (requiresAccessGroup)
    operands.push_back(
        "OptionalAttr<LLVM_AccessGroupArrayAttr>:$access_groups");
  if (requiresAliasAnalysis) {
    operands.push_back("OptionalAttr<LLVM_AliasScopeArrayAttr>:$alias_scopes");
    operands.push_back(
        "OptionalAttr<LLVM_AliasScopeArrayAttr>:$noalias_scopes");
    operands.push_back("OptionalAttr<LLVM_TBAATagArrayAttr>:$tbaa");
  }

  // Emit the definition.
  os << "def LLVM_" << intr.getProperRecordName() << " : " << opBaseClass
     << "<\"" << intr.getOperationName() << "\", ";
  printBracketedRange(intr.getOverloadableResultsIdxs().set_bits(), os);
  os << ", ";
  printBracketedRange(intr.getOverloadableOperandsIdxs().set_bits(), os);
  os << ", ";
  printBracketedRange(traits, os);
  os << ", " << intr.getNumResults() << ", "
     << (requiresAccessGroup ? "1" : "0") << ", "
     << (requiresAliasAnalysis ? "1" : "0") << ">, Arguments<(ins"
     << (operands.empty() ? "" : " ");
  llvm::interleaveComma(operands, os);
  os << ")>;\n\n";

  return false;
}

/// Traverses the list of TableGen definitions derived from the "Intrinsic"
/// class and generates MLIR ODS definitions for those intrinsics that have
/// the name matching the filter.
static bool emitIntrinsics(const RecordKeeper &records, llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader("Operations for LLVM intrinsics", os, records);
  os << "include \"mlir/Dialect/LLVMIR/LLVMOpBase.td\"\n";
  os << "include \"mlir/Interfaces/SideEffectInterfaces.td\"\n\n";

  auto defs = records.getAllDerivedDefinitions("Intrinsic");
  for (const Record *r : defs) {
    if (!nameFilter.empty() && !r->getName().contains(nameFilter))
      continue;
    if (emitIntrinsic(*r, os))
      return true;
  }

  return false;
}

static mlir::GenRegistration genLLVMIRIntrinsics("gen-llvmir-intrinsics",
                                                 "Generate LLVM IR intrinsics",
                                                 emitIntrinsics);
