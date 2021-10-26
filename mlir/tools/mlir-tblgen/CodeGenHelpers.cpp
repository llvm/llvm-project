//===- CodeGenHelpers.cpp - MLIR op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDefinitionsGen uses the description of operations to generate C++
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

StaticVerifierFunctionEmitter::StaticVerifierFunctionEmitter(
    const llvm::RecordKeeper &records, raw_ostream &os)
    : os(os), uniqueOutputLabel(getUniqueName(records)) {}

void StaticVerifierFunctionEmitter::emitFunctionsFor(
    StringRef signatureFormat, StringRef errorHandlerFormat,
    StringRef typeArgName, ArrayRef<llvm::Record *> opDefs, bool emitDecl) {
  llvm::Optional<NamespaceEmitter> namespaceEmitter;
  if (!emitDecl)
    namespaceEmitter.emplace(os, Operator(*opDefs[0]).getCppNamespace());

  emitTypeConstraintMethods(signatureFormat, errorHandlerFormat, typeArgName,
                            opDefs, emitDecl);
}

StringRef StaticVerifierFunctionEmitter::getTypeConstraintFn(
    const Constraint &constraint) const {
  auto it = localTypeConstraints.find(constraint.getAsOpaquePointer());
  assert(it != localTypeConstraints.end() && "expected valid constraint fn");
  return it->second;
}

std::string StaticVerifierFunctionEmitter::getUniqueName(
    const llvm::RecordKeeper &records) {
  // Use the input file name when generating a unique name.
  std::string inputFilename = records.getInputFilename();

  // Drop all but the base filename.
  StringRef nameRef = llvm::sys::path::filename(inputFilename);
  nameRef.consume_back(".td");

  // Sanitize any invalid characters.
  std::string uniqueName;
  for (char c : nameRef) {
    if (llvm::isAlnum(c) || c == '_')
      uniqueName.push_back(c);
    else
      uniqueName.append(llvm::utohexstr((unsigned char)c));
  }
  return uniqueName;
}

void StaticVerifierFunctionEmitter::emitTypeConstraintMethods(
    StringRef signatureFormat, StringRef errorHandlerFormat,
    StringRef typeArgName, ArrayRef<llvm::Record *> opDefs, bool emitDecl) {
  // Collect a set of all of the used type constraints within the operation
  // definitions.
  llvm::SetVector<const void *> typeConstraints;
  for (Record *def : opDefs) {
    Operator op(*def);
    for (NamedTypeConstraint &operand : op.getOperands())
      if (operand.hasPredicate())
        typeConstraints.insert(operand.constraint.getAsOpaquePointer());
    for (NamedTypeConstraint &result : op.getResults())
      if (result.hasPredicate())
        typeConstraints.insert(result.constraint.getAsOpaquePointer());
  }

  // Record the mapping from predicate to constraint. If two constraints has the
  // same predicate and constraint summary, they can share the same verification
  // function.
  llvm::DenseMap<Pred, const void *> predToConstraint;
  FmtContext fctx;
  for (auto it : llvm::enumerate(typeConstraints)) {
    std::string name;
    Constraint constraint = Constraint::getFromOpaquePointer(it.value());
    Pred pred = constraint.getPredicate();
    auto iter = predToConstraint.find(pred);
    if (iter != predToConstraint.end()) {
      do {
        Constraint built = Constraint::getFromOpaquePointer(iter->second);
        // We may have the different constraints but have the same predicate,
        // for example, ConstraintA and Variadic<ConstraintA>, note that
        // Variadic<> doesn't introduce new predicate. In this case, we can
        // share the same predicate function if they also have consistent
        // summary, otherwise we may report the wrong message while verification
        // fails.
        if (constraint.getSummary() == built.getSummary()) {
          name = getTypeConstraintFn(built).str();
          break;
        }
        ++iter;
      } while (iter != predToConstraint.end() && iter->first == pred);
    }

    if (!name.empty()) {
      localTypeConstraints.try_emplace(it.value(), name);
      continue;
    }

    // Generate an obscure and unique name for this type constraint.
    name = (Twine("__mlir_ods_local_type_constraint_") + uniqueOutputLabel +
            Twine(it.index()))
               .str();
    predToConstraint.insert(
        std::make_pair(constraint.getPredicate(), it.value()));
    localTypeConstraints.try_emplace(it.value(), name);

    // Only generate the methods if we are generating definitions.
    if (emitDecl)
      continue;

    os << formatv(signatureFormat.data(), name) << " {\n";
    os.indent() << "if (!("
                << tgfmt(constraint.getConditionTemplate(),
                         &fctx.withSelf(typeArgName))
                << ")) {\n";
    os.indent() << "return "
                << formatv(errorHandlerFormat.data(), constraint.getSummary())
                << ";\n";
    os.unindent() << "}\nreturn ::mlir::success();\n";
    os.unindent() << "}\n\n";
  }
}

std::string mlir::tblgen::escapeString(StringRef value) {
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os.write_escaped(value);
  return os.str();
}
