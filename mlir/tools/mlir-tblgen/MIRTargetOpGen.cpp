//===- MIRTargetOpGen.cpp - Generate target MIR op ODS --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A TableGen backend that reads a target's selected-instruction definitions
// (`Instruction` records in a given target Namespace) and emits MLIR ODS for a
// target MIR dialect (e.g. `aarch64_mir`), collapsing the many per-encoding /
// per-operand-kind record variants down to one op per assembler mnemonic.
//
// The collapse follows the design established for these dialects: within a
// mnemonic, operand-kind/width variation is absorbed by modeling operands and
// results as LLT-typed values (mir dialect's MIR_AnyLLT); the concrete target
// opcode is recovered at export time from the operand types plus an optional
// `variant` discriminator. This backend also emits a lowering table mapping
// each generated op back to the set of concrete opcodes it subsumes.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::Record;
using llvm::RecordKeeper;
using namespace mlir;

static llvm::cl::OptionCategory targetOpCat("MIR target-op generator options");

static llvm::cl::opt<std::string>
    targetNamespace("mir-target",
                    llvm::cl::desc("Target Namespace to filter instructions by "
                                   "(e.g. AArch64)"),
                    llvm::cl::init("AArch64"), llvm::cl::cat(targetOpCat));

static llvm::cl::opt<std::string>
    dialectClass("mir-target-opbase",
                 llvm::cl::desc("ODS base class for the generated ops"),
                 llvm::cl::init("AArch64MIR_TargetOp"),
                 llvm::cl::cat(targetOpCat));

/// Value of a boolean record field, treating unset/missing as false.
static bool getBitOrFalse(const Record &record, llvm::StringRef field) {
  const llvm::RecordVal *rv = record.getValue(field);
  if (!rv)
    return false;
  if (const auto *bit = llvm::dyn_cast_or_null<llvm::BitInit>(rv->getValue()))
    return bit->getValue();
  return false;
}

/// Extract the assembler mnemonic (first whitespace/brace/comma-delimited
/// token) from an AsmString. Returns empty if none.
static llvm::StringRef getMnemonic(llvm::StringRef asmString) {
  asmString = asmString.ltrim();
  size_t end = asmString.find_first_of(" \t{,");
  return end == llvm::StringRef::npos ? asmString : asmString.take_front(end);
}

/// Turn a mnemonic into a valid MLIR op name: lowercase, runs of non-alnum
/// characters collapsed to a single '_', leading/trailing '_' trimmed.
static std::string sanitizeMnemonic(llvm::StringRef mnemonic) {
  std::string out;
  bool lastUnderscore = false;
  for (char c : mnemonic) {
    if (llvm::isAlnum(c)) {
      out += static_cast<char>(llvm::toLower(c));
      lastUnderscore = false;
    } else if (!lastUnderscore) {
      out += '_';
      lastUnderscore = true;
    }
  }
  llvm::StringRef trimmed = llvm::StringRef(out).trim('_');
  return trimmed.empty() ? std::string("op") : trimmed.str();
}

/// Turn a name into a valid ODS/C++ identifier fragment. The result also
/// becomes the op's C++ class name (ODS strips the dialect prefix), so guard
/// against C++ keywords and alternative tokens (e.g. the `and` mnemonic).
static std::string sanitizeIdent(llvm::StringRef name) {
  std::string out;
  for (char c : name)
    out += llvm::isAlnum(c) ? c : '_';
  if (out.empty() || !llvm::isAlpha(out[0]))
    out = "op_" + out;
  static const char *kReserved[] = {
      "and",     "and_eq", "bitand", "bitor", "compl", "not",    "not_eq",
      "or",      "or_eq",  "xor",    "xor_eq", "int",  "float",  "double",
      "char",    "bool",   "void",   "class",  "struct", "const", "static",
      "new",     "delete", "operator", "template", "default", "return",
      "register"};
  for (const char *kw : kReserved)
    if (out == kw)
      return out + "_";
  return out;
}

namespace {
/// Accumulated information for one mnemonic cluster.
struct Cluster {
  bool anyResults = false;   // some variant defines results
  bool anyOperands = false;  // some variant takes inputs
  bool allPure = true;       // every variant is side-effect free
  bool anyCommutable = false;
  llvm::SmallVector<std::string> opcodes; // concrete record names subsumed
};
} // namespace

/// Count the args in a dag operand list.
static unsigned numDagArgs(const Record &record, llvm::StringRef field) {
  if (const auto *dag = llvm::dyn_cast_or_null<llvm::DagInit>(
          record.getValue(field) ? record.getValue(field)->getValue()
                                  : nullptr))
    return dag->getNumArgs();
  return 0;
}

static bool emitTargetOps(const RecordKeeper &records, llvm::raw_ostream &os) {
  llvm::emitSourceFileHeader(
      (llvm::Twine(targetNamespace) + " MIR target operations").str(), os,
      records);
  os << "include \"mlir/Dialect/MIR/IR/MIRTypes.td\"\n";
  os << "include \"mlir/Interfaces/SideEffectInterfaces.td\"\n\n";

  // Cluster instructions by mnemonic, preserving first-seen order.
  llvm::StringMap<unsigned> clusterIndex;
  llvm::SmallVector<std::pair<std::string, Cluster>> clusters;
  unsigned considered = 0;
  for (const Record *r : records.getAllDerivedDefinitions("Instruction")) {
    if (r->getValueAsString("Namespace") != targetNamespace)
      continue;
    if (getBitOrFalse(*r, "isPseudo") || getBitOrFalse(*r, "isCodeGenOnly"))
      continue;
    llvm::StringRef mnemonic = getMnemonic(r->getValueAsString("AsmString"));
    if (mnemonic.empty())
      continue;
    ++considered;
    auto it = clusterIndex.find(mnemonic);
    if (it == clusterIndex.end()) {
      it = clusterIndex.insert({mnemonic, clusters.size()}).first;
      clusters.push_back({mnemonic.str(), Cluster{}});
    }
    Cluster &c = clusters[it->second].second;
    c.anyResults |= numDagArgs(*r, "OutOperandList") > 0;
    c.anyOperands |= numDagArgs(*r, "InOperandList") > 0;
    bool pure = !getBitOrFalse(*r, "hasSideEffects") &&
                !getBitOrFalse(*r, "mayLoad") && !getBitOrFalse(*r, "mayStore");
    c.allPure &= pure;
    c.anyCommutable |= getBitOrFalse(*r, "isCommutable");
    c.opcodes.push_back(r->getName().str());
  }

  // Emit one op per cluster, uniquifying identifiers.
  llvm::StringSet<> usedNames;
  unsigned emitted = 0;
  for (const auto &kv : clusters) {
    llvm::StringRef mnemonic = kv.first;
    const Cluster &c = kv.second;

    std::string opName = sanitizeMnemonic(mnemonic);
    std::string ident = sanitizeIdent(opName);
    std::string base = ident;
    unsigned suffix = 0;
    while (!usedNames.insert(ident).second)
      ident = base + "_" + llvm::utostr(suffix++);

    llvm::SmallVector<llvm::StringRef> traits;
    if (c.allPure)
      traits.push_back("Pure");
    if (c.anyCommutable)
      traits.push_back("Commutative");

    os << "def " << targetNamespace << "MIR_" << ident << " : "
       << dialectClass << "<\"" << opName << "\", [";
    llvm::interleaveComma(traits, os);
    os << "]> {\n";
    os << "  let arguments = (ins Variadic<MIR_AnyLLT>:$srcs,\n"
          "                       OptionalAttr<StrAttr>:$variant);\n";
    os << "  let results = (outs Variadic<MIR_AnyLLT>:$rets);\n";
    // Lowering table: the concrete opcodes this op subsumes.
    os << "  // opcodes:";
    for (llvm::StringRef opc : c.opcodes)
      os << " " << opc;
    os << "\n}\n\n";
    ++emitted;
  }

  os << "// Considered " << considered << " instructions, emitted " << emitted
     << " ops across " << clusters.size() << " mnemonics.\n";
  return false;
}

static mlir::GenRegistration
    genMIRTargetOps("gen-mir-target-ops",
                    "Generate target MIR dialect operation ODS", emitTargetOps);
