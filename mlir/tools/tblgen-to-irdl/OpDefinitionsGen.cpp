//===- OpDefinitionsGen.cpp - IRDL op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDefinitionsGen uses the description of operations to generate IRDL
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/GenNameParser.h"
#include "mlir/TableGen/Interfaces.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using tblgen::NamedTypeConstraint;

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-irdl-dialect");
llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::Required);

Value createConstraint(OpBuilder &builder, tblgen::Constraint constraint) {
  MLIRContext *ctx = builder.getContext();
  const Record &predRec = constraint.getDef();

  if (predRec.isSubClassOf("Variadic") || predRec.isSubClassOf("Optional"))
    return createConstraint(builder, predRec.getValueAsDef("baseType"));

  if (predRec.getName() == "AnyType") {
    auto op = builder.create<irdl::AnyOp>(UnknownLoc::get(ctx));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("TypeDef")) {
    std::string typeName = ("!" + predRec.getValueAsString("typeName")).str();
    auto op = builder.create<irdl::BaseOp>(UnknownLoc::get(ctx),
                                           StringAttr::get(ctx, typeName));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyTypeOf")) {
    std::vector<Value> constraints;
    for (Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AllOfType")) {
    std::vector<Value> constraints;
    for (Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  std::string condition = constraint.getPredicate().getCondition();
  // Build a CPredOp to match the C constraint built.
  irdl::CPredOp op = builder.create<irdl::CPredOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, condition));
  return op;
}

/// Returns the name of the operation without the dialect prefix.
static StringRef getOperatorName(tblgen::Operator &tblgenOp) {
  StringRef opName = tblgenOp.getDef().getValueAsString("opName");
  return opName;
}

/// Extract an operation to IRDL.
irdl::OperationOp createIRDLOperation(OpBuilder &builder,
                                      tblgen::Operator &tblgenOp) {
  MLIRContext *ctx = builder.getContext();
  StringRef opName = getOperatorName(tblgenOp);

  irdl::OperationOp op = builder.create<irdl::OperationOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, opName));

  // Add the block in the region.
  Block &opBlock = op.getBody().emplaceBlock();
  OpBuilder consBuilder = OpBuilder::atBlockBegin(&opBlock);

  auto getValues = [&](tblgen::Operator::const_value_range namedCons) {
    SmallVector<Value> operands;
    SmallVector<irdl::VariadicityAttr> variadicity;
    for (const NamedTypeConstraint &namedCons : namedCons) {
      auto operand = createConstraint(consBuilder, namedCons.constraint);
      operands.push_back(operand);

      irdl::VariadicityAttr var;
      if (namedCons.isOptional())
        var = consBuilder.getAttr<irdl::VariadicityAttr>(
            irdl::Variadicity::optional);
      else if (namedCons.isVariadic())
        var = consBuilder.getAttr<irdl::VariadicityAttr>(
            irdl::Variadicity::variadic);
      else
        var = consBuilder.getAttr<irdl::VariadicityAttr>(
            irdl::Variadicity::single);

      variadicity.push_back(var);
    }
    return std::make_tuple(operands, variadicity);
  };

  auto [operands, operandVariadicity] = getValues(tblgenOp.getOperands());
  auto [results, resultVariadicity] = getValues(tblgenOp.getResults());

  // Create the operands and results operations.
  consBuilder.create<irdl::OperandsOp>(UnknownLoc::get(ctx), operands,
                                       operandVariadicity);
  consBuilder.create<irdl::ResultsOp>(UnknownLoc::get(ctx), results,
                                      resultVariadicity);

  return op;
}

static irdl::DialectOp createIRDLDialect(OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  return builder.create<irdl::DialectOp>(UnknownLoc::get(ctx),
                                         StringAttr::get(ctx, selectedDialect));
}

static std::vector<llvm::Record *>
getOpDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("Op"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("Op");
}

static bool emitDialectIRDLDefs(const RecordKeeper &recordKeeper,
                                raw_ostream &os) {
  // Initialize.
  MLIRContext ctx;
  ctx.getOrLoadDialect<irdl::IRDLDialect>();
  OpBuilder builder(&ctx);

  // Create a module op and set it as the insertion point.
  OwningOpRef<ModuleOp> module =
      builder.create<ModuleOp>(UnknownLoc::get(&ctx));
  builder = builder.atBlockBegin(module->getBody());
  // Create the dialect and insert it.
  irdl::DialectOp dialect = createIRDLDialect(builder);
  // Set insertion point to start of DialectOp.
  builder = builder.atBlockBegin(&dialect.getBody().emplaceBlock());

  std::vector<Record *> defs = getOpDefinitions(recordKeeper);
  for (auto *def : defs) {
    tblgen::Operator tblgenOp(def);
    if (tblgenOp.getDialectName() != selectedDialect)
      continue;

    createIRDLOperation(builder, tblgenOp);
  }

  // Print the module.
  module->print(os);

  return false;
}

static mlir::GenRegistration
    genOpDefs("gen-dialect-irdl-defs", "Generate IRDL dialect definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                return emitDialectIRDLDefs(records, os);
              });
