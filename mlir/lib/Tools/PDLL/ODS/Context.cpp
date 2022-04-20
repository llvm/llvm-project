//===- Context.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/ODS/Context.h"
#include "mlir/Tools/PDLL/ODS/Constraint.h"
#include "mlir/Tools/PDLL/ODS/Dialect.h"
#include "mlir/Tools/PDLL/ODS/Operation.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pdll::ods;

//===----------------------------------------------------------------------===//
// Context
//===----------------------------------------------------------------------===//

Context::Context() = default;
Context::~Context() = default;

const AttributeConstraint &
Context::insertAttributeConstraint(StringRef name, StringRef summary,
                                   StringRef cppClass) {
  std::unique_ptr<AttributeConstraint> &constraint = attributeConstraints[name];
  if (!constraint) {
    constraint.reset(new AttributeConstraint(name, summary, cppClass));
  } else {
    assert(constraint->getCppClass() == cppClass &&
           constraint->getSummary() == summary &&
           "constraint with the same name was already registered with a "
           "different class");
  }
  return *constraint;
}

const TypeConstraint &Context::insertTypeConstraint(StringRef name,
                                                    StringRef summary,
                                                    StringRef cppClass) {
  std::unique_ptr<TypeConstraint> &constraint = typeConstraints[name];
  if (!constraint)
    constraint.reset(new TypeConstraint(name, summary, cppClass));
  return *constraint;
}

Dialect &Context::insertDialect(StringRef name) {
  std::unique_ptr<Dialect> &dialect = dialects[name];
  if (!dialect)
    dialect.reset(new Dialect(name));
  return *dialect;
}

const Dialect *Context::lookupDialect(StringRef name) const {
  auto it = dialects.find(name);
  return it == dialects.end() ? nullptr : &*it->second;
}

std::pair<Operation *, bool> Context::insertOperation(StringRef name,
                                                      StringRef summary,
                                                      StringRef desc,
                                                      SMLoc loc) {
  std::pair<StringRef, StringRef> dialectAndName = name.split('.');
  return insertDialect(dialectAndName.first)
      .insertOperation(name, summary, desc, loc);
}

const Operation *Context::lookupOperation(StringRef name) const {
  std::pair<StringRef, StringRef> dialectAndName = name.split('.');
  if (const Dialect *dialect = lookupDialect(dialectAndName.first))
    return dialect->lookupOperation(name);
  return nullptr;
}

template <typename T>
SmallVector<T *> sortMapByName(const llvm::StringMap<std::unique_ptr<T>> &map) {
  SmallVector<T *> storage;
  for (auto &entry : map)
    storage.push_back(entry.second.get());
  llvm::sort(storage, [](const auto &lhs, const auto &rhs) {
    return lhs->getName() < rhs->getName();
  });
  return storage;
}

void Context::print(raw_ostream &os) const {
  auto printVariableLengthCst = [&](StringRef cst, VariableLengthKind kind) {
    switch (kind) {
    case VariableLengthKind::Optional:
      os << "Optional<" << cst << ">";
      break;
    case VariableLengthKind::Single:
      os << cst;
      break;
    case VariableLengthKind::Variadic:
      os << "Variadic<" << cst << ">";
      break;
    }
  };

  llvm::ScopedPrinter printer(os);
  llvm::DictScope odsScope(printer, "ODSContext");
  for (const Dialect *dialect : sortMapByName(dialects)) {
    printer.startLine() << "Dialect `" << dialect->getName() << "` {\n";
    printer.indent();

    for (const Operation *op : sortMapByName(dialect->getOperations())) {
      printer.startLine() << "Operation `" << op->getName() << "` {\n";
      printer.indent();

      // Attributes.
      ArrayRef<Attribute> attributes = op->getAttributes();
      if (!attributes.empty()) {
        printer.startLine() << "Attributes { ";
        llvm::interleaveComma(attributes, os, [&](const Attribute &attr) {
          os << attr.getName() << " : ";

          auto kind = attr.isOptional() ? VariableLengthKind::Optional
                                        : VariableLengthKind::Single;
          printVariableLengthCst(attr.getConstraint().getDemangledName(), kind);
        });
        os << " }\n";
      }

      // Operands.
      ArrayRef<OperandOrResult> operands = op->getOperands();
      if (!operands.empty()) {
        printer.startLine() << "Operands { ";
        llvm::interleaveComma(
            operands, os, [&](const OperandOrResult &operand) {
              os << operand.getName() << " : ";
              printVariableLengthCst(operand.getConstraint().getDemangledName(),
                                     operand.getVariableLengthKind());
            });
        os << " }\n";
      }

      // Results.
      ArrayRef<OperandOrResult> results = op->getResults();
      if (!results.empty()) {
        printer.startLine() << "Results { ";
        llvm::interleaveComma(results, os, [&](const OperandOrResult &result) {
          os << result.getName() << " : ";
          printVariableLengthCst(result.getConstraint().getDemangledName(),
                                 result.getVariableLengthKind());
        });
        os << " }\n";
      }

      printer.objectEnd();
    }
    printer.objectEnd();
  }
  for (const AttributeConstraint *cst : sortMapByName(attributeConstraints)) {
    printer.startLine() << "AttributeConstraint `" << cst->getDemangledName()
                        << "` {\n";
    printer.indent();

    printer.startLine() << "Summary: " << cst->getSummary() << "\n";
    printer.startLine() << "CppClass: " << cst->getCppClass() << "\n";
    printer.objectEnd();
  }
  for (const TypeConstraint *cst : sortMapByName(typeConstraints)) {
    printer.startLine() << "TypeConstraint `" << cst->getDemangledName()
                        << "` {\n";
    printer.indent();

    printer.startLine() << "Summary: " << cst->getSummary() << "\n";
    printer.startLine() << "CppClass: " << cst->getCppClass() << "\n";
    printer.objectEnd();
  }
  printer.objectEnd();
}
