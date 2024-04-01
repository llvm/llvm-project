
#include "Dialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::obs;

#include "Dialect.cpp.inc"

void OBSDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "Ops.cpp.inc"
    >();
}

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {

    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;

    if (parser.parseOperandList(operands, 2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc, 
            result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    
    result.addTypes(type);
    return mlir::success();
}

static void printBinary(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer <<  " : " ;
    
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(), [=](Type type) { return type == resultType; })) {
        printer << resultType;
    }

    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
    auto dataType = RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = DenseElementsAttr::get(dataType, value);
    ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    mlir::DenseElementsAttr value;

    if (parser.parseOptionalAttrDict(result.attributes) || 
        parser.parseAttribute(value, "value", result.attributes))
        return failure();
    
    result.addTypes(value.getType());
    return success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
    printer << getValue();
}

mlir::LogicalResult ConstantOp::verify() {
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    if (!resultType)
        return success();

    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
    for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
        if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
            return emitOpError("return type shape mismatches its attribute at dimension")
            << dim << ": " << attrType.getShape()[dim]
            << " != " << resultType.getShape()[dim];
        }
    }
    return mlir::success();
}

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, 
                  mlir::Value lhs, mlir::Value rhs) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands({lhs, rhs});
}

mlir::ParseResult AddOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    return parseBinaryOp(parser, result);
}

void AddOp::print(mlir::OpAsmPrinter &p) {
    printBinary(p, * this);
}

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          StringRef callee, ArrayRef<mlir::Value> arguments) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(arguments);
    state.addAttribute("callee", mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		   llvm::StringRef name, mlir::FunctionType type,
		   llvm::ArrayRef<mlir::NamedAttribute> attrs ) {
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
				mlir::OperationState &result) {
  auto buildFuncType =
    [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
       llvm::ArrayRef<mlir::Type> results,
       mlir::function_interface_impl::VariadicFlag,
       std::string &) { return builder.getFunctionType(argTypes, results); };
  return mlir::function_interface_impl::
    parseFunctionOp(parser, result, false,
		    getFunctionTypeAttrName(result.name),
		    buildFuncType,
		    getArgAttrsAttrName(result.name),
		    getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::
    printFunctionOp(p, *this, false,
		    getFunctionTypeAttrName(),
		    getArgAttrsAttrName(),
		    getResAttrsAttrName());
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
		  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

mlir::ParseResult MulOp::parse(mlir::OpAsmParser &parser,
			       mlir::OperationState &result) {
  return parseBinaryOp(parser, result);
}

void MulOp::print(mlir::OpAsmPrinter &p) {
  printBinary(p, * this);
}

mlir::LogicalResult ReturnOp::verify() {
    auto function = cast<FuncOp>((*this).getParentOp());
    if (getNumOperands() > 1) {
        return emitOpError() << "expects at most 1 return operand" ;
    }
    const auto &results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size()) {
        return emitOpError() << "does not return the same number of values("
                             << getNumOperands() << ") as the enclosing function ("
                             << results.size() << ")"; 
    }

    if (!hasOperand()) {
        return mlir::success();
    }
    
    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    if (inputType == resultType || llvm::isa<mlir::UnrankedTensorType>(inputType) ||
        llvm::isa<mlir::UnrankedTensorType>(resultType) ) {
        return mlir::success();
    }
    return emitError() << "type of return operand (" << inputType
                       << ") doesn't match function result type (" << resultType
                       << ")";
}

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value value) {
    state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    state.addOperands(value);
}

mlir::LogicalResult TransposeOp::verify() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
    auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
    if (!inputType || !resultType)
        return mlir::success();

    auto inputShape = inputType.getShape();
    if(!std::equal(inputShape.begin(), inputShape.end(), resultType.getShape().rbegin())) {
        return emitError() << "expected result shape to be a transpose of the input";
    }
    return mlir::success();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"




