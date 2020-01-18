//===- LLVMIntrinsicGen.cpp - TableGen utility for converting intrinsics --===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a TableGen generator that converts TableGen definitions for LLVM
// intrinsics to TableGen definitions for MLIR operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/GenInfo.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

static llvm::cl::OptionCategory IntrinsicGenCat("Intrinsics Generator Options");

static llvm::cl::opt<std::string>
    nameFilter("llvmir-intrinsics-filter",
               llvm::cl::desc("Only keep the intrinsics with the specified "
                              "substring in their record name"),
               llvm::cl::cat(IntrinsicGenCat));

namespace {
/// A wrapper for LLVM's Tablegen class `Intrinsic` that provides accessors to
/// the fields of the record.
class LLVMIntrinsic {
public:
  LLVMIntrinsic(const llvm::Record &record) : record(record) {}

  /// Get the name of the operation to be used in MLIR.  Uses the appropriate
  /// field if not empty, constructs a name by replacing underscores with dots
  /// in the record name otherwise.
  std::string getOperationName() const {
    llvm::StringRef name = record.getValueAsString(fieldName);
    if (!name.empty())
      return name.str();

    name = record.getName();
    assert(name.startswith("int_") &&
           "LLVM intinsic names are expected to start with 'int_'");
    name = name.drop_front(4);
    llvm::SmallVector<llvm::StringRef, 8> chunks;
    name.split(chunks, '_');
    return llvm::join(chunks, ".");
  }

  /// Get the name of the record without the "intrinsic" prefix.
  llvm::StringRef getProperRecordName() const {
    llvm::StringRef name = record.getName();
    assert(name.startswith("int_") &&
           "LLVM intinsic names are expected to start with 'int_'");
    return name.drop_front(4);
  }

  /// Get the number of operands.
  unsigned getNumOperands() const {
    auto operands = record.getValueAsListOfDefs(fieldOperands);
    for (const llvm::Record *r : operands) {
      (void)r;
      assert(r->isSubClassOf("LLVMType") &&
             "expected operands to be of LLVM type");
    }
    return operands.size();
  }

  /// Get the number of results.  Note that LLVM does not support multi-value
  /// operations so, in fact, multiple results will be returned as a value of
  /// structure type.
  unsigned getNumResults() const {
    auto results = record.getValueAsListOfDefs(fieldResults);
    for (const llvm::Record *r : results) {
      (void)r;
      assert(r->isSubClassOf("LLVMType") &&
             "expected operands to be of LLVM type");
    }
    return results.size();
  }

  /// Return true if the intrinsic may have side effects, i.e. does not have the
  /// `IntrNoMem` property.
  bool hasSideEffects() const {
    auto props = record.getValueAsListOfDefs(fieldTraits);
    for (const llvm::Record *r : props) {
      if (r->getName() == "IntrNoMem")
        return true;
    }
    return false;
  }

  /// Return true if the intrinsic is commutative, i.e. has the respective
  /// property.
  bool isCommutative() const {
    auto props = record.getValueAsListOfDefs(fieldTraits);
    for (const llvm::Record *r : props) {
      if (r->getName() == "Commutative")
        return true;
    }
    return false;
  }

private:
  /// Names of the fileds in the Intrinsic LLVM Tablegen class.
  const char *fieldName = "LLVMName";
  const char *fieldOperands = "ParamTypes";
  const char *fieldResults = "RetTypes";
  const char *fieldTraits = "IntrProperties";

  const llvm::Record &record;
};
} // namespace

/// Emits C++ code constructing an LLVM IR intrinsic given the generated MLIR
/// operation.  In LLVM IR, intrinsics are constructed as function calls.
static void emitBuilder(const LLVMIntrinsic &intr, llvm::raw_ostream &os) {
  os << "    llvm::Module *module = builder.GetInsertBlock()->getModule();\n";
  os << "    llvm::Function *fn = llvm::Intrinsic::getDeclaration(\n";
  os << "        module, llvm::Intrinsic::" << intr.getProperRecordName()
     << ");\n";
  os << "    auto operands = llvm::to_vector<8, Value *>(\n";
  os << "        opInst.operand_begin(), opInst.operand_end());\n";
  os << "    " << (intr.getNumResults() > 0 ? "$res = " : "")
     << "builder.CreateCall(fn, operands);\n";
  os << "  ";
}

/// Emits ODS (TableGen-based) code for `record` representing an LLVM intrinsic.
/// Returns true on error, false on success.
static bool emitIntrinsic(const llvm::Record &record, llvm::raw_ostream &os) {
  LLVMIntrinsic intr(record);

  // Prepare strings for traits, if any.
  llvm::SmallVector<llvm::StringRef, 2> traits;
  if (intr.isCommutative())
    traits.push_back("Commutative");
  if (!intr.hasSideEffects())
    traits.push_back("NoSideEffect");

  // Prepare strings for operands.
  llvm::SmallVector<llvm::StringRef, 8> operands(intr.getNumOperands(),
                                                 "LLVM_Type");

  // Emit the definition.
  os << "def LLVM_" << intr.getProperRecordName() << " : LLVM_Op<\"intr."
     << intr.getOperationName() << "\", [";
  mlir::interleaveComma(traits, os);
  os << "]>, Arguments<(ins" << (operands.empty() ? "" : " ");
  mlir::interleaveComma(operands, os);
  os << ")>, Results<(outs"
     << (intr.getNumResults() == 0 ? "" : " LLVM_Type:$res") << ")> {\n"
     << "  let llvmBuilder = [{\n";
  emitBuilder(intr, os);
  os << "}];\n";
  os << "}\n\n";

  return false;
}

/// Traverses the list of TableGen definitions derived from the "Intrinsic"
/// class and generates MLIR ODS definitions for those intrinsics that have
/// the name matching the filter.
static bool emitIntrinsics(const llvm::RecordKeeper &records,
                           llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader("Operations for LLVM intrinsics", os);
  os << "include \"mlir/Dialect/LLVMIR/LLVMOpBase.td\"\n\n";

  auto defs = records.getAllDerivedDefinitions("Intrinsic");
  for (const llvm::Record *r : defs) {
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
