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

Value createPredicate(OpBuilder &builder, tblgen::Pred pred) {
  MLIRContext *ctx = builder.getContext();

  if (pred.isCombined()) {
    auto combiner = pred.getDef().getValueAsDef("kind")->getName();
    if (combiner == "PredCombinerAnd" || combiner == "PredCombinerOr") {
      std::vector<Value> constraints;
      for (auto *child : pred.getDef().getValueAsListOfDefs("children")) {
        constraints.push_back(createPredicate(builder, tblgen::Pred(child)));
      }
      if (combiner == "PredCombinerAnd") {
        auto op =
            builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
        return op.getOutput();
      }
      auto op =
          builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
      return op.getOutput();
    }
  }

  std::string condition = pred.getCondition();
  // Build a CPredOp to match the C constraint built.
  irdl::CPredOp op = builder.create<irdl::CPredOp>(
      UnknownLoc::get(ctx), StringAttr::get(ctx, condition));
  return op;
}

Value typeToConstraint(OpBuilder &builder, Type type) {
  MLIRContext *ctx = builder.getContext();
  auto op =
      builder.create<irdl::IsOp>(UnknownLoc::get(ctx), TypeAttr::get(type));
  return op.getOutput();
}

std::optional<Type> recordToType(MLIRContext *ctx, const Record &predRec) {

  if (predRec.isSubClassOf("I")) {
    auto width = predRec.getValueAsInt("bitwidth");
    return IntegerType::get(ctx, width, IntegerType::Signless);
  }

  if (predRec.isSubClassOf("SI")) {
    auto width = predRec.getValueAsInt("bitwidth");
    return IntegerType::get(ctx, width, IntegerType::Signed);
  }

  if (predRec.isSubClassOf("UI")) {
    auto width = predRec.getValueAsInt("bitwidth");
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  }

  // Index type
  if (predRec.getName() == "Index") {
    return IndexType::get(ctx);
  }

  // Float types
  if (predRec.isSubClassOf("F")) {
    auto width = predRec.getValueAsInt("bitwidth");
    switch (width) {
    case 16:
      return FloatType::getF16(ctx);
    case 32:
      return FloatType::getF32(ctx);
    case 64:
      return FloatType::getF64(ctx);
    case 80:
      return FloatType::getF80(ctx);
    case 128:
      return FloatType::getF128(ctx);
    }
  }

  if (predRec.getName() == "NoneType") {
    return NoneType::get(ctx);
  }

  if (predRec.getName() == "BF16") {
    return FloatType::getBF16(ctx);
  }

  if (predRec.getName() == "TF32") {
    return FloatType::getTF32(ctx);
  }

  if (predRec.getName() == "F8E4M3FN") {
    return FloatType::getFloat8E4M3FN(ctx);
  }

  if (predRec.getName() == "F8E5M2") {
    return FloatType::getFloat8E5M2(ctx);
  }

  if (predRec.getName() == "F8E4M3") {
    return FloatType::getFloat8E4M3(ctx);
  }

  if (predRec.getName() == "F8E4M3FNUZ") {
    return FloatType::getFloat8E4M3FNUZ(ctx);
  }

  if (predRec.getName() == "F8E4M3B11FNUZ") {
    return FloatType::getFloat8E4M3B11FNUZ(ctx);
  }

  if (predRec.getName() == "F8E5M2FNUZ") {
    return FloatType::getFloat8E5M2FNUZ(ctx);
  }

  if (predRec.getName() == "F8E3M4") {
    return FloatType::getFloat8E3M4(ctx);
  }

  if (predRec.isSubClassOf("Complex")) {
    const Record *elementRec = predRec.getValueAsDef("elementType");
    auto elementType = recordToType(ctx, *elementRec);
    if (elementType.has_value()) {
      return ComplexType::get(elementType.value());
    }
  }

  return std::nullopt;
}

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
    for (const Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AllOfType")) {
    std::vector<Value> constraints;
    for (const Record *child : predRec.getValueAsListOfDefs("allowedTypes")) {
      constraints.push_back(
          createConstraint(builder, tblgen::Constraint(child)));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  // Integer types
  if (predRec.getName() == "AnyInteger") {
    auto op = builder.create<irdl::BaseOp>(
        UnknownLoc::get(ctx), StringAttr::get(ctx, "!builtin.integer"));
    return op.getOutput();
  }

  if (predRec.isSubClassOf("AnyI")) {
    auto width = predRec.getValueAsInt("bitwidth");
    std::vector<Value> types = {
        typeToConstraint(builder,
                         IntegerType::get(ctx, width, IntegerType::Signless)),
        typeToConstraint(builder,
                         IntegerType::get(ctx, width, IntegerType::Signed)),
        typeToConstraint(builder,
                         IntegerType::get(ctx, width, IntegerType::Unsigned))};
    auto op = builder.create<irdl::AnyOfOp>(UnknownLoc::get(ctx), types);
    return op.getOutput();
  }

  auto type = recordToType(ctx, predRec);

  if (type.has_value()) {
    return typeToConstraint(builder, type.value());
  }

  // Confined type
  if (predRec.isSubClassOf("ConfinedType")) {
    std::vector<Value> constraints;
    constraints.push_back(createConstraint(
        builder, tblgen::Constraint(predRec.getValueAsDef("baseType"))));
    for (Record *child : predRec.getValueAsListOfDefs("predicateList")) {
      constraints.push_back(createPredicate(builder, tblgen::Pred(child)));
    }
    auto op = builder.create<irdl::AllOfOp>(UnknownLoc::get(ctx), constraints);
    return op.getOutput();
  }

  return createPredicate(builder, constraint.getPredicate());
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
  if (!operands.empty())
    consBuilder.create<irdl::OperandsOp>(UnknownLoc::get(ctx), operands,
                                         operandVariadicity);
  if (!results.empty())
    consBuilder.create<irdl::ResultsOp>(UnknownLoc::get(ctx), results,
                                        resultVariadicity);

  return op;
}

static irdl::DialectOp createIRDLDialect(OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  return builder.create<irdl::DialectOp>(UnknownLoc::get(ctx),
                                         StringAttr::get(ctx, selectedDialect));
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

  for (const Record *def :
       recordKeeper.getAllDerivedDefinitionsIfDefined("Op")) {
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
