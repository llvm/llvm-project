//===- MIRGenericOpGen.cpp - Generate MIR generic-opcode ODS --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A TableGen backend that reads LLVM's generic (GlobalISel) opcode definitions
// from `GenericInstruction` records (llvm/include/llvm/Target/GenericOpcodes.td)
// and emits MLIR ODS operation definitions for the `mir` dialect.
//
// Each `G_*` record becomes an op `mir.g_*` derived from the `MIR_GenericOp`
// base class. Generic-type operands/results (type0/ptype0/...) map onto the
// LLT-typed value constraint `MIR_AnyLLT`; `variable_ops` maps to a variadic;
// immediate placeholders (i32imm, untyped_imm_0, unknown, ...) map to
// attributes. `hasSideEffects`/`mayLoad`/`mayStore` decide the `Pure` trait and
// `isCommutable` the `Commutative` trait.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::DagInit;
using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::RecordKeeper;
using namespace mlir;

/// Return the value of a boolean record field, treating an unset (`?`) or
/// missing field as `false`.
static bool getBitOrFalse(const Record &record, llvm::StringRef field) {
  const llvm::RecordVal *rv = record.getValue(field);
  if (!rv)
    return false;
  if (const auto *bit = llvm::dyn_cast_or_null<llvm::BitInit>(rv->getValue()))
    return bit->getValue();
  return false;
}

/// Return an operand/attribute name safe to use as a C++ identifier in
/// generated ODS accessors, appending `_` to C++ keywords.
static std::string sanitizeName(llvm::StringRef name) {
  static const char *kKeywords[] = {
      "register", "int",   "float",  "double", "char",  "bool",
      "void",     "class", "struct", "const",  "static", "new",
      "delete",   "operator", "template", "default", "return"};
  for (const char *kw : kKeywords)
    if (name == kw)
      return (name + "_").str();
  return name.str();
}

/// Classify a dag operand by the name of its `TypedOperand`/marker def and
/// append the corresponding ODS entry to either the operand or attribute list.
/// Returns false and does nothing for unrecognized kinds (the caller treats
/// this as "skip the whole op").
static bool appendArg(llvm::StringRef defName, llvm::StringRef rawName,
                      llvm::SmallVectorImpl<std::string> &operands,
                      llvm::SmallVectorImpl<std::string> &attributes) {
  std::string argName = sanitizeName(rawName);
  // Generic type variables and pointer type variables become LLT-typed values.
  if (defName.starts_with("type") || defName.starts_with("ptype")) {
    operands.push_back(("MIR_AnyLLT:$" + argName));
    return true;
  }
  // A variadic tail of same-typed values.
  if (defName == "variable_ops") {
    operands.push_back("Variadic<MIR_AnyLLT>:$varargs");
    return true;
  }
  // Immediate / opaque placeholders become attributes.
  if (defName == "unknown" || defName.ends_with("imm") ||
      defName.starts_with("untyped_imm") || defName.starts_with("i") ) {
    attributes.push_back(("AnyAttr:$" + argName));
    return true;
  }
  return false;
}

/// Emit the ODS definition for a single `GenericInstruction` record. Returns
/// true if the op was emitted, false if it was skipped.
static bool emitGenericOp(const Record &record, llvm::raw_ostream &os) {
  llvm::StringRef recordName = record.getName();
  std::string opName = recordName.lower(); // G_ADD -> g_add

  const DagInit *outs = record.getValueAsDag("OutOperandList");
  const DagInit *ins = record.getValueAsDag("InOperandList");

  llvm::SmallVector<std::string> operands, attributes, results;

  auto handleDag = [&](const DagInit *dag, bool isResult) -> bool {
    for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
      const auto *def = llvm::dyn_cast<DefInit>(dag->getArg(i));
      if (!def)
        return false;
      llvm::StringRef defName = def->getDef()->getName();
      llvm::StringRef argName = dag->getArgNameStr(i);
      if (isResult) {
        // Generic type variables become LLT-typed results; a trailing
        // `variable_ops` becomes a single variadic result.
        if (defName.starts_with("type") || defName.starts_with("ptype")) {
          results.push_back("MIR_AnyLLT:$" + sanitizeName(argName));
          continue;
        }
        if (defName == "variable_ops") {
          results.push_back("Variadic<MIR_AnyLLT>:$outs");
          continue;
        }
        return false;
      }
      if (!appendArg(defName, argName, operands, attributes))
        return false;
    }
    return true;
  };

  if (!handleDag(outs, /*isResult=*/true) || !handleDag(ins, /*isResult=*/false))
    return false;

  // Traits.
  llvm::SmallVector<llvm::StringRef> traits;
  bool pure = !getBitOrFalse(record, "hasSideEffects") &&
              !getBitOrFalse(record, "mayLoad") &&
              !getBitOrFalse(record, "mayStore");
  if (pure)
    traits.push_back("Pure");
  if (getBitOrFalse(record, "isCommutable"))
    traits.push_back("Commutative");

  // Emit.
  os << "def MIR_" << recordName << " : MIR_GenericOp<\"" << opName << "\", [";
  llvm::interleaveComma(traits, os);
  os << "]> {\n";

  os << "  let arguments = (ins";
  llvm::SmallVector<std::string> args(operands);
  args.append(attributes.begin(), attributes.end());
  if (!args.empty()) {
    os << " ";
    llvm::interleaveComma(args, os);
  }
  os << ");\n";

  os << "  let results = (outs";
  if (!results.empty()) {
    os << " ";
    llvm::interleaveComma(results, os);
  }
  os << ");\n";

  os << "}\n\n";
  return true;
}

static bool emitGenericOps(const RecordKeeper &records, llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader("MIR dialect generic (G_*) operations", os,
                             records);
  // Emit includes so the generated file is self-contained. Include guards in
  // these headers make re-inclusion (when MIROps.td also includes them) safe.
  os << "include \"mlir/Dialect/MIR/IR/MIRTypes.td\"\n";
  os << "include \"mlir/Interfaces/SideEffectInterfaces.td\"\n\n";
  unsigned emitted = 0, skipped = 0;
  for (const Record *r : records.getAllDerivedDefinitions("GenericInstruction")) {
    if (emitGenericOp(*r, os))
      ++emitted;
    else {
      ++skipped;
      os << "// skipped: " << r->getName() << "\n";
    }
  }
  os << "// Emitted " << emitted << " ops, skipped " << skipped << ".\n";
  return false;
}

static mlir::GenRegistration
    genMIRGenericOps("gen-mir-generic-ops",
                     "Generate MIR dialect generic (G_*) operation ODS",
                     emitGenericOps);
