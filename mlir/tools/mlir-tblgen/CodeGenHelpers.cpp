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
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

StaticVerifierFunctionEmitter::StaticVerifierFunctionEmitter(
    const llvm::RecordKeeper &records)
    : uniqueOutputLabel(getUniqueName(records)) {}

StaticVerifierFunctionEmitter &
StaticVerifierFunctionEmitter::setSelf(StringRef str) {
  fctx.withSelf(str);
  return *this;
}

StaticVerifierFunctionEmitter &
StaticVerifierFunctionEmitter::setBuilder(StringRef str) {
  fctx.withBuilder(str);
  return *this;
}

void StaticVerifierFunctionEmitter::emitConstraintMethodsInNamespace(
    StringRef signatureFormat, StringRef errorHandlerFormat,
    StringRef cppNamespace, ArrayRef<const void *> constraints, raw_ostream &os,
    bool emitDecl) {
  llvm::Optional<NamespaceEmitter> namespaceEmitter;
  if (!emitDecl)
    namespaceEmitter.emplace(os, cppNamespace);

  emitConstraintMethods(signatureFormat, errorHandlerFormat, constraints, os,
                        emitDecl);
}

StringRef StaticVerifierFunctionEmitter::getConstraintFn(
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

void StaticVerifierFunctionEmitter::emitConstraintMethods(
    StringRef signatureFormat, StringRef errorHandlerFormat,
    ArrayRef<const void *> constraints, raw_ostream &rawOs, bool emitDecl) {
  raw_indented_ostream os(rawOs);

  // Record the mapping from predicate to constraint. If two constraints has the
  // same predicate and constraint summary, they can share the same verification
  // function.
  llvm::DenseMap<Pred, const void *> predToConstraint;
  for (auto it : llvm::enumerate(constraints)) {
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
          name = getConstraintFn(built).str();
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
    os.indent() << "if (!(" << tgfmt(constraint.getConditionTemplate(), &fctx)
                << ")) {\n";
    os.indent() << "return "
                << formatv(errorHandlerFormat.data(),
                           escapeString(constraint.getSummary()))
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
