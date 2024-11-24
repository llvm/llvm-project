//===- LLVMTypeSyntax.cpp - Parsing/printing for MLIR LLVM Dialect types --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Printing.
//===----------------------------------------------------------------------===//

/// If the given type is compatible with the LLVM dialect, prints it using
/// internal functions to avoid getting a verbose `!llvm` prefix. Otherwise
/// prints it as usual.
static void dispatchPrint(AsmPrinter &printer, Type type) {
  if (isCompatibleType(type) &&
      !llvm::isa<IntegerType, FloatType, VectorType>(type))
    return mlir::LLVM::detail::printType(type, printer);
  printer.printType(type);
}

/// Returns the keyword to use for the given type.
static StringRef getTypeKeyword(Type type) {
  return TypeSwitch<Type, StringRef>(type)
      .Case<LLVMVoidType>([&](Type) { return "void"; })
      .Case<LLVMPPCFP128Type>([&](Type) { return "ppc_fp128"; })
      .Case<LLVMTokenType>([&](Type) { return "token"; })
      .Case<LLVMLabelType>([&](Type) { return "label"; })
      .Case<LLVMMetadataType>([&](Type) { return "metadata"; })
      .Case<LLVMFunctionType>([&](Type) { return "func"; })
      .Case<LLVMPointerType>([&](Type) { return "ptr"; })
      .Case<LLVMFixedVectorType, LLVMScalableVectorType>(
          [&](Type) { return "vec"; })
      .Case<LLVMArrayType>([&](Type) { return "array"; })
      .Case<LLVMStructType>([&](Type) { return "struct"; })
      .Case<LLVMTargetExtType>([&](Type) { return "target"; })
      .Case<LLVMX86AMXType>([&](Type) { return "x86_amx"; })
      .Default([](Type) -> StringRef {
        llvm_unreachable("unexpected 'llvm' type kind");
      });
}

/// Prints the given LLVM dialect type recursively. This leverages closedness of
/// the LLVM dialect type system to avoid printing the dialect prefix
/// repeatedly. For recursive structures, only prints the name of the structure
/// when printing a self-reference. Note that this does not apply to sibling
/// references. For example,
///   struct<"a", (ptr<struct<"a">>)>
///   struct<"c", (ptr<struct<"b", (ptr<struct<"c">>)>>,
///                ptr<struct<"b", (ptr<struct<"c">>)>>)>
/// note that "b" is printed twice.
void mlir::LLVM::detail::printType(Type type, AsmPrinter &printer) {
  if (!type) {
    printer << "<<NULL-TYPE>>";
    return;
  }

  printer << getTypeKeyword(type);

  llvm::TypeSwitch<Type>(type)
      .Case<LLVMPointerType, LLVMArrayType, LLVMFixedVectorType,
            LLVMScalableVectorType, LLVMFunctionType, LLVMTargetExtType,
            LLVMStructType>([&](auto type) { type.print(printer); });
}

//===----------------------------------------------------------------------===//
// Parsing.
//===----------------------------------------------------------------------===//

static ParseResult dispatchParse(AsmParser &parser, Type &type);

/// Parses an LLVM dialect vector type.
///   llvm-type ::= `vec<` `? x`? integer `x` llvm-type `>`
/// Supports both fixed and scalable vectors.
static Type parseVectorType(AsmParser &parser) {
  SmallVector<int64_t, 2> dims;
  SMLoc dimPos, typePos;
  Type elementType;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLess() || parser.getCurrentLocation(&dimPos) ||
      parser.parseDimensionList(dims, /*allowDynamic=*/true) ||
      parser.getCurrentLocation(&typePos) ||
      dispatchParse(parser, elementType) || parser.parseGreater())
    return Type();

  // We parsed a generic dimension list, but vectors only support two forms:
  //  - single non-dynamic entry in the list (fixed vector);
  //  - two elements, the first dynamic (indicated by ShapedType::kDynamic)
  //  and the second
  //    non-dynamic (scalable vector).
  if (dims.empty() || dims.size() > 2 ||
      ((dims.size() == 2) ^ (ShapedType::isDynamic(dims[0]))) ||
      (dims.size() == 2 && ShapedType::isDynamic(dims[1]))) {
    parser.emitError(dimPos)
        << "expected '? x <integer> x <type>' or '<integer> x <type>'";
    return Type();
  }

  bool isScalable = dims.size() == 2;
  if (isScalable)
    return parser.getChecked<LLVMScalableVectorType>(loc, elementType, dims[1]);
  if (elementType.isSignlessIntOrFloat()) {
    parser.emitError(typePos)
        << "cannot use !llvm.vec for built-in primitives, use 'vector' instead";
    return Type();
  }
  return parser.getChecked<LLVMFixedVectorType>(loc, elementType, dims[0]);
}

/// Parses a type appearing inside another LLVM dialect-compatible type. This
/// will try to parse any type in full form (including types with the `!llvm`
/// prefix), and on failure fall back to parsing the short-hand version of the
/// LLVM dialect types without the `!llvm` prefix.
static Type dispatchParse(AsmParser &parser, bool allowAny = true) {
  SMLoc keyLoc = parser.getCurrentLocation();

  // Try parsing any MLIR type.
  Type type;
  OptionalParseResult result = parser.parseOptionalType(type);
  if (result.has_value()) {
    if (failed(result.value()))
      return nullptr;
    if (!allowAny) {
      parser.emitError(keyLoc) << "unexpected type, expected keyword";
      return nullptr;
    }
    return type;
  }

  // If no type found, fallback to the shorthand form.
  StringRef key;
  if (failed(parser.parseKeyword(&key)))
    return Type();

  MLIRContext *ctx = parser.getContext();
  return StringSwitch<function_ref<Type()>>(key)
      .Case("void", [&] { return LLVMVoidType::get(ctx); })
      .Case("ppc_fp128", [&] { return LLVMPPCFP128Type::get(ctx); })
      .Case("token", [&] { return LLVMTokenType::get(ctx); })
      .Case("label", [&] { return LLVMLabelType::get(ctx); })
      .Case("metadata", [&] { return LLVMMetadataType::get(ctx); })
      .Case("func", [&] { return LLVMFunctionType::parse(parser); })
      .Case("ptr", [&] { return LLVMPointerType::parse(parser); })
      .Case("vec", [&] { return parseVectorType(parser); })
      .Case("array", [&] { return LLVMArrayType::parse(parser); })
      .Case("struct", [&] { return LLVMStructType::parse(parser); })
      .Case("target", [&] { return LLVMTargetExtType::parse(parser); })
      .Case("x86_amx", [&] { return LLVMX86AMXType::get(ctx); })
      .Default([&] {
        parser.emitError(keyLoc) << "unknown LLVM type: " << key;
        return Type();
      })();
}

/// Helper to use in parse lists.
static ParseResult dispatchParse(AsmParser &parser, Type &type) {
  type = dispatchParse(parser);
  return success(type != nullptr);
}

/// Parses one of the LLVM dialect types.
Type mlir::LLVM::detail::parseType(DialectAsmParser &parser) {
  SMLoc loc = parser.getCurrentLocation();
  Type type = dispatchParse(parser, /*allowAny=*/false);
  if (!type)
    return type;
  if (!isCompatibleOuterType(type)) {
    parser.emitError(loc) << "unexpected type, expected keyword";
    return nullptr;
  }
  return type;
}

ParseResult LLVM::parsePrettyLLVMType(AsmParser &p, Type &type) {
  return dispatchParse(p, type);
}

ParseResult LLVM::parsePrettyLLVMType(AsmParser &p,
                                      SmallVectorImpl<Type> &types) {
  return p.parseCommaSeparatedList(
      [&] { return dispatchParse(p, types.emplace_back()); });
}

void LLVM::printPrettyLLVMType(AsmPrinter &p, TypeRange types) {
  interleaveComma(types, p.getStream(),
                  [&](Type type) { return dispatchPrint(p, type); });
}
