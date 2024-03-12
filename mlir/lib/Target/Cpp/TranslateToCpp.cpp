//===- TranslateToCpp.cpp - Translating to C++ calls ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/Cpp/TranslateToCpp.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppTranslationUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <utility>

#define DEBUG_TYPE "translate-to-cpp"

using namespace mlir;
using namespace mlir::emitc;
using llvm::formatv;

/// Return the precedence of a operator as an integer, higher values
/// imply higher precedence.
static FailureOr<int> getOperatorPrecedence(Operation *operation) {
  return llvm::TypeSwitch<Operation *, FailureOr<int>>(operation)
      .Case<emitc::AddOp>([&](auto op) { return 12; })
      .Case<emitc::ApplyOp>([&](auto op) { return 15; })
      .Case<emitc::BitwiseAndOp>([&](auto op) { return 7; })
      .Case<emitc::BitwiseLeftShiftOp>([&](auto op) { return 11; })
      .Case<emitc::BitwiseNotOp>([&](auto op) { return 15; })
      .Case<emitc::BitwiseOrOp>([&](auto op) { return 5; })
      .Case<emitc::BitwiseRightShiftOp>([&](auto op) { return 11; })
      .Case<emitc::BitwiseXorOp>([&](auto op) { return 6; })
      .Case<emitc::CallOp>([&](auto op) { return 16; })
      .Case<emitc::CallOpaqueOp>([&](auto op) { return 16; })
      .Case<emitc::CastOp>([&](auto op) { return 15; })
      .Case<emitc::CmpOp>([&](auto op) -> FailureOr<int> {
        switch (op.getPredicate()) {
        case emitc::CmpPredicate::eq:
        case emitc::CmpPredicate::ne:
          return 8;
        case emitc::CmpPredicate::lt:
        case emitc::CmpPredicate::le:
        case emitc::CmpPredicate::gt:
        case emitc::CmpPredicate::ge:
          return 9;
        case emitc::CmpPredicate::three_way:
          return 10;
        }
        return op->emitError("unsupported cmp predicate");
      })
      .Case<emitc::ConditionalOp>([&](auto op) { return 2; })
      .Case<emitc::DivOp>([&](auto op) { return 13; })
      .Case<emitc::LogicalAndOp>([&](auto op) { return 4; })
      .Case<emitc::LogicalNotOp>([&](auto op) { return 15; })
      .Case<emitc::LogicalOrOp>([&](auto op) { return 3; })
      .Case<emitc::MulOp>([&](auto op) { return 13; })
      .Case<emitc::RemOp>([&](auto op) { return 13; })
      .Case<emitc::SubOp>([&](auto op) { return 12; })
      .Case<emitc::UnaryMinusOp>([&](auto op) { return 15; })
      .Case<emitc::UnaryPlusOp>([&](auto op) { return 15; })
      .Default([](auto op) { return op->emitError("unsupported operation"); });
}

CppEmitter::CppEmitter(raw_ostream &os, Operation *module,
                       bool declareVariablesAtTop)
    : os(os), module(module), declareVariablesAtTop(declareVariablesAtTop),
      iface(module->getContext()) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef CppEmitter::getOrCreateName(Value val) {
  if (auto literal = dyn_cast_if_present<emitc::LiteralOp>(val.getDefiningOp()))
    return literal.getValue();
  if (!valueMapper.count(val))
    valueMapper.insert(val, formatv("v{0}", ++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef CppEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, formatv("label{0}", ++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool CppEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool CppEmitter::hasValueInScope(Value val) { return valueMapper.count(val); }

bool CppEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult CppEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      switch (llvm::APFloatBase::SemanticsToEnum(val.getSemantics())) {
      case llvm::APFloatBase::S_IEEEsingle:
        os << "(float)";
        break;
      case llvm::APFloatBase::S_IEEEdouble:
        os << "(double)";
        break;
      default:
        break;
      };
      os << strValue;
    } else if (val.isNaN()) {
      os << "NAN";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "-";
      os << "INFINITY";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print opaque attributes.
  if (auto oAttr = dyn_cast<emitc::OpaqueAttr>(attr)) {
    os << oAttr.getValue();
    return success();
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult CppEmitter::emitExpression(ExpressionOp expressionOp) {
  assert(emittedExpressionPrecedence.empty() &&
         "Expected precedence stack to be empty");
  Operation *rootOp = expressionOp.getRootOp();

  emittedExpression = expressionOp;
  FailureOr<int> precedence = getOperatorPrecedence(rootOp);
  if (failed(precedence))
    return failure();
  pushExpressionPrecedence(precedence.value());

  if (failed(emitOperation(*rootOp, /*trailingSemicolon=*/false)))
    return failure();

  popExpressionPrecedence();
  assert(emittedExpressionPrecedence.empty() &&
         "Expected precedence stack to be empty");
  emittedExpression = nullptr;

  return success();
}

LogicalResult CppEmitter::emitOperand(Value value) {
  if (isPartOfCurrentExpression(value)) {
    Operation *def = value.getDefiningOp();
    assert(def && "Expected operand to be defined by an operation");
    FailureOr<int> precedence = getOperatorPrecedence(def);
    if (failed(precedence))
      return failure();
    bool encloseInParenthesis = precedence.value() < getExpressionPrecedence();
    if (encloseInParenthesis) {
      os << "(";
      pushExpressionPrecedence(lowestPrecedence());
    } else
      pushExpressionPrecedence(precedence.value());

    if (failed(emitOperation(*def, /*trailingSemicolon=*/false)))
      return failure();

    if (encloseInParenthesis)
      os << ")";

    popExpressionPrecedence();
    return success();
  }

  auto expressionOp = dyn_cast_if_present<ExpressionOp>(value.getDefiningOp());
  if (expressionOp && shouldBeInlined(expressionOp))
    return emitExpression(expressionOp);

  auto literalOp = dyn_cast_if_present<LiteralOp>(value.getDefiningOp());
  if (!literalOp && !hasValueInScope(value))
    return failure();
  os << getOrCreateName(value);
  return success();
}

LogicalResult CppEmitter::emitOperands(Operation &op) {
  return interleaveCommaWithError(op.getOperands(), os, [&](Value operand) {
    // If an expression is being emitted, push lowest precedence as these
    // operands are either wrapped by parenthesis.
    if (getEmittedExpression())
      pushExpressionPrecedence(lowestPrecedence());
    if (failed(emitOperand(operand)))
      return failure();
    if (getEmittedExpression())
      popExpressionPrecedence();
    return success();
  });
}

LogicalResult
CppEmitter::emitOperandsAndAttributes(Operation &op,
                                      ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult CppEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(OpResult result,
                                                  bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  if (failed(emitVariableDeclaration(result.getOwner()->getLoc(),
                                     result.getType(),
                                     getOrCreateName(result))))
    return failure();
  if (trailingSemicolon)
    os << ";\n";
  return success();
}

LogicalResult CppEmitter::emitAssignPrefix(Operation &op) {
  // If op is being emitted as part of an expression, bail out.
  if (getEmittedExpression())
    return success();

  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult CppEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

LogicalResult CppEmitter::emitOperation(Operation &op, bool trailingSemicolon) {
  const CppTranslationDialectInterface *opIface = iface.getInterfaceFor(&op);
  if (!opIface)
    return op.emitError("cannot be converted to Cpp: missing "
                        "`CppTranslationDialectInterface` registration for "
                        "dialect for op: ")
           << op.getName();

  if (failed(opIface->emitOperation(&op, *this, trailingSemicolon)))
    return failure();

  return success();
}

LogicalResult CppEmitter::emitVariableDeclaration(Location loc, Type type,
                                                  StringRef name) {
  if (auto arrType = dyn_cast<emitc::ArrayType>(type)) {
    if (failed(emitType(loc, arrType.getElementType())))
      return failure();
    os << " " << name;
    for (auto dim : arrType.getShape()) {
      os << "[" << dim << "]";
    }
    return success();
  }
  if (failed(emitType(loc, type)))
    return failure();
  os << " " << name;
  return success();
}

LogicalResult CppEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (isa<ArrayType>(tType.getElementType()))
      return emitError(loc, "cannot emit tensor of array type ") << type;
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  if (auto oType = dyn_cast<emitc::OpaqueType>(type)) {
    os << oType.getValue();
    return success();
  }
  if (auto aType = dyn_cast<emitc::ArrayType>(type)) {
    if (failed(emitType(loc, aType.getElementType())))
      return failure();
    for (auto dim : aType.getShape())
      os << "[" << dim << "]";
    return success();
  }
  if (auto pType = dyn_cast<emitc::PointerType>(type)) {
    if (isa<ArrayType>(pType.getPointee()))
      return emitError(loc, "cannot emit pointer to array type ") << type;
    if (failed(emitType(loc, pType.getPointee())))
      return failure();
    os << "*";
    return success();
  }
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult CppEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult CppEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  if (llvm::any_of(types, [](Type type) { return isa<ArrayType>(type); })) {
    return emitError(loc, "cannot emit tuple of array type");
  }
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult mlir::translateToCpp(Operation *op, raw_ostream &os,
                                   bool declareVariablesAtTop) {
  CppEmitter emitter(os, op, declareVariablesAtTop);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}
