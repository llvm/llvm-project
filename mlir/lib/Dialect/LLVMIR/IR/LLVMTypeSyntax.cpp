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
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// Printing.
//===----------------------------------------------------------------------===//

static void printTypeImpl(llvm::raw_ostream &os, LLVMType type,
                          llvm::SetVector<StringRef> &stack);

/// Returns the keyword to use for the given type.
static StringRef getTypeKeyword(LLVMType type) {
  return TypeSwitch<Type, StringRef>(type)
      .Case<LLVMVoidType>([&](Type) { return "void"; })
      .Case<LLVMHalfType>([&](Type) { return "half"; })
      .Case<LLVMBFloatType>([&](Type) { return "bfloat"; })
      .Case<LLVMFloatType>([&](Type) { return "float"; })
      .Case<LLVMDoubleType>([&](Type) { return "double"; })
      .Case<LLVMFP128Type>([&](Type) { return "fp128"; })
      .Case<LLVMX86FP80Type>([&](Type) { return "x86_fp80"; })
      .Case<LLVMPPCFP128Type>([&](Type) { return "ppc_fp128"; })
      .Case<LLVMX86MMXType>([&](Type) { return "x86_mmx"; })
      .Case<LLVMTokenType>([&](Type) { return "token"; })
      .Case<LLVMLabelType>([&](Type) { return "label"; })
      .Case<LLVMMetadataType>([&](Type) { return "metadata"; })
      .Case<LLVMFunctionType>([&](Type) { return "func"; })
      .Case<LLVMIntegerType>([&](Type) { return "i"; })
      .Case<LLVMPointerType>([&](Type) { return "ptr"; })
      .Case<LLVMVectorType>([&](Type) { return "vec"; })
      .Case<LLVMArrayType>([&](Type) { return "array"; })
      .Case<LLVMStructType>([&](Type) { return "struct"; })
      .Default([](Type) -> StringRef {
        llvm_unreachable("unexpected 'llvm' type kind");
      });
}

/// Prints the body of a structure type. Uses `stack` to avoid printing
/// recursive structs indefinitely.
static void printStructTypeBody(llvm::raw_ostream &os, LLVMStructType type,
                                llvm::SetVector<StringRef> &stack) {
  if (type.isIdentified() && type.isOpaque()) {
    os << "opaque";
    return;
  }

  if (type.isPacked())
    os << "packed ";

  // Put the current type on stack to avoid infinite recursion.
  os << '(';
  if (type.isIdentified())
    stack.insert(type.getName());
  llvm::interleaveComma(type.getBody(), os, [&](LLVMType subtype) {
    printTypeImpl(os, subtype, stack);
  });
  if (type.isIdentified())
    stack.pop_back();
  os << ')';
}

/// Prints a structure type. Uses `stack` to keep track of the identifiers of
/// the structs being printed. Checks if the identifier of a struct is contained
/// in `stack`, i.e. whether a self-reference to a recursive stack is being
/// printed, and only prints the name to avoid infinite recursion.
static void printStructType(llvm::raw_ostream &os, LLVMStructType type,
                            llvm::SetVector<StringRef> &stack) {
  os << "<";
  if (type.isIdentified()) {
    os << '"' << type.getName() << '"';
    // If we are printing a reference to one of the enclosing structs, just
    // print the name and stop to avoid infinitely long output.
    if (stack.count(type.getName())) {
      os << '>';
      return;
    }
    os << ", ";
  }

  printStructTypeBody(os, type, stack);
  os << '>';
}

/// Prints a type containing a fixed number of elements.
template <typename TypeTy>
static void printArrayOrVectorType(llvm::raw_ostream &os, TypeTy type,
                                   llvm::SetVector<StringRef> &stack) {
  os << '<' << type.getNumElements() << " x ";
  printTypeImpl(os, type.getElementType(), stack);
  os << '>';
}

/// Prints a function type.
static void printFunctionType(llvm::raw_ostream &os, LLVMFunctionType funcType,
                              llvm::SetVector<StringRef> &stack) {
  os << '<';
  printTypeImpl(os, funcType.getReturnType(), stack);
  os << " (";
  llvm::interleaveComma(
      funcType.getParams(), os,
      [&os, &stack](LLVMType subtype) { printTypeImpl(os, subtype, stack); });
  if (funcType.isVarArg()) {
    if (funcType.getNumParams() != 0)
      os << ", ";
    os << "...";
  }
  os << ")>";
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
static void printTypeImpl(llvm::raw_ostream &os, LLVMType type,
                          llvm::SetVector<StringRef> &stack) {
  if (!type) {
    os << "<<NULL-TYPE>>";
    return;
  }

  os << getTypeKeyword(type);

  if (auto intType = type.dyn_cast<LLVMIntegerType>()) {
    os << intType.getBitWidth();
    return;
  }

  if (auto ptrType = type.dyn_cast<LLVMPointerType>()) {
    os << '<';
    printTypeImpl(os, ptrType.getElementType(), stack);
    if (ptrType.getAddressSpace() != 0)
      os << ", " << ptrType.getAddressSpace();
    os << '>';
    return;
  }

  if (auto arrayType = type.dyn_cast<LLVMArrayType>())
    return printArrayOrVectorType(os, arrayType, stack);
  if (auto vectorType = type.dyn_cast<LLVMFixedVectorType>())
    return printArrayOrVectorType(os, vectorType, stack);

  if (auto vectorType = type.dyn_cast<LLVMScalableVectorType>()) {
    os << "<? x " << vectorType.getMinNumElements() << " x ";
    printTypeImpl(os, vectorType.getElementType(), stack);
    os << '>';
    return;
  }

  if (auto structType = type.dyn_cast<LLVMStructType>())
    return printStructType(os, structType, stack);

  if (auto funcType = type.dyn_cast<LLVMFunctionType>())
    return printFunctionType(os, funcType, stack);
}

void mlir::LLVM::detail::printType(LLVMType type, DialectAsmPrinter &printer) {
  llvm::SetVector<StringRef> stack;
  return printTypeImpl(printer.getStream(), type, stack);
}

//===----------------------------------------------------------------------===//
// Parsing.
//===----------------------------------------------------------------------===//

static LLVMType parseTypeImpl(DialectAsmParser &parser,
                              llvm::SetVector<StringRef> &stack);

/// Helper to be chained with other parsing functions.
static ParseResult parseTypeImpl(DialectAsmParser &parser,
                                 llvm::SetVector<StringRef> &stack,
                                 LLVMType &result) {
  result = parseTypeImpl(parser, stack);
  return success(result != nullptr);
}

/// Parses an LLVM dialect function type.
///   llvm-type :: = `func<` llvm-type `(` llvm-type-list `...`? `)>`
static LLVMFunctionType parseFunctionType(DialectAsmParser &parser,
                                          llvm::SetVector<StringRef> &stack) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  LLVMType returnType;
  if (parser.parseLess() || parseTypeImpl(parser, stack, returnType) ||
      parser.parseLParen())
    return LLVMFunctionType();

  // Function type without arguments.
  if (succeeded(parser.parseOptionalRParen())) {
    if (succeeded(parser.parseGreater()))
      return LLVMFunctionType::getChecked(loc, returnType, {},
                                          /*isVarArg=*/false);
    return LLVMFunctionType();
  }

  // Parse arguments.
  SmallVector<LLVMType, 8> argTypes;
  do {
    if (succeeded(parser.parseOptionalEllipsis())) {
      if (parser.parseOptionalRParen() || parser.parseOptionalGreater())
        return LLVMFunctionType();
      return LLVMFunctionType::getChecked(loc, returnType, argTypes,
                                          /*isVarArg=*/true);
    }

    argTypes.push_back(parseTypeImpl(parser, stack));
    if (!argTypes.back())
      return LLVMFunctionType();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseOptionalRParen() || parser.parseOptionalGreater())
    return LLVMFunctionType();
  return LLVMFunctionType::getChecked(loc, returnType, argTypes,
                                      /*isVarArg=*/false);
}

/// Parses an LLVM dialect pointer type.
///   llvm-type ::= `ptr<` llvm-type (`,` integer)? `>`
static LLVMPointerType parsePointerType(DialectAsmParser &parser,
                                        llvm::SetVector<StringRef> &stack) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  LLVMType elementType;
  if (parser.parseLess() || parseTypeImpl(parser, stack, elementType))
    return LLVMPointerType();

  unsigned addressSpace = 0;
  if (succeeded(parser.parseOptionalComma()) &&
      failed(parser.parseInteger(addressSpace)))
    return LLVMPointerType();
  if (failed(parser.parseGreater()))
    return LLVMPointerType();
  return LLVMPointerType::getChecked(loc, elementType, addressSpace);
}

/// Parses an LLVM dialect vector type.
///   llvm-type ::= `vec<` `? x`? integer `x` llvm-type `>`
/// Supports both fixed and scalable vectors.
static LLVMVectorType parseVectorType(DialectAsmParser &parser,
                                      llvm::SetVector<StringRef> &stack) {
  SmallVector<int64_t, 2> dims;
  llvm::SMLoc dimPos;
  LLVMType elementType;
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.getCurrentLocation(&dimPos) ||
      parser.parseDimensionList(dims, /*allowDynamic=*/true) ||
      parseTypeImpl(parser, stack, elementType) || parser.parseGreater())
    return LLVMVectorType();

  // We parsed a generic dimension list, but vectors only support two forms:
  //  - single non-dynamic entry in the list (fixed vector);
  //  - two elements, the first dynamic (indicated by -1) and the second
  //    non-dynamic (scalable vector).
  if (dims.empty() || dims.size() > 2 ||
      ((dims.size() == 2) ^ (dims[0] == -1)) ||
      (dims.size() == 2 && dims[1] == -1)) {
    parser.emitError(dimPos)
        << "expected '? x <integer> x <type>' or '<integer> x <type>'";
    return LLVMVectorType();
  }

  bool isScalable = dims.size() == 2;
  if (isScalable)
    return LLVMScalableVectorType::getChecked(loc, elementType, dims[1]);
  return LLVMFixedVectorType::getChecked(loc, elementType, dims[0]);
}

/// Parses an LLVM dialect array type.
///   llvm-type ::= `array<` integer `x` llvm-type `>`
static LLVMArrayType parseArrayType(DialectAsmParser &parser,
                                    llvm::SetVector<StringRef> &stack) {
  SmallVector<int64_t, 1> dims;
  llvm::SMLoc sizePos;
  LLVMType elementType;
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.getCurrentLocation(&sizePos) ||
      parser.parseDimensionList(dims, /*allowDynamic=*/false) ||
      parseTypeImpl(parser, stack, elementType) || parser.parseGreater())
    return LLVMArrayType();

  if (dims.size() != 1) {
    parser.emitError(sizePos) << "expected ? x <type>";
    return LLVMArrayType();
  }

  return LLVMArrayType::getChecked(loc, elementType, dims[0]);
}

/// Attempts to set the body of an identified structure type. Reports a parsing
/// error at `subtypesLoc` in case of failure, uses `stack` to make sure the
/// types printed in the error message look like they did when parsed.
static LLVMStructType trySetStructBody(LLVMStructType type,
                                       ArrayRef<LLVMType> subtypes,
                                       bool isPacked, DialectAsmParser &parser,
                                       llvm::SMLoc subtypesLoc,
                                       llvm::SetVector<StringRef> &stack) {
  for (LLVMType t : subtypes) {
    if (!LLVMStructType::isValidElementType(t)) {
      parser.emitError(subtypesLoc)
          << "invalid LLVM structure element type: " << t;
      return LLVMStructType();
    }
  }

  if (succeeded(type.setBody(subtypes, isPacked)))
    return type;

  std::string currentBody;
  llvm::raw_string_ostream currentBodyStream(currentBody);
  printStructTypeBody(currentBodyStream, type, stack);
  auto diag = parser.emitError(subtypesLoc)
              << "identified type already used with a different body";
  diag.attachNote() << "existing body: " << currentBodyStream.str();
  return LLVMStructType();
}

/// Parses an LLVM dialect structure type.
///   llvm-type ::= `struct<` (string-literal `,`)? `packed`?
///                 `(` llvm-type-list `)` `>`
///               | `struct<` string-literal `>`
///               | `struct<` string-literal `, opaque>`
static LLVMStructType parseStructType(DialectAsmParser &parser,
                                      llvm::SetVector<StringRef> &stack) {
  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  if (failed(parser.parseLess()))
    return LLVMStructType();

  // If we are parsing a self-reference to a recursive struct, i.e. the parsing
  // stack already contains a struct with the same identifier, bail out after
  // the name.
  StringRef name;
  bool isIdentified = succeeded(parser.parseOptionalString(&name));
  if (isIdentified) {
    if (stack.count(name)) {
      if (failed(parser.parseGreater()))
        return LLVMStructType();
      return LLVMStructType::getIdentifiedChecked(loc, name);
    }
    if (failed(parser.parseComma()))
      return LLVMStructType();
  }

  // Handle intentionally opaque structs.
  llvm::SMLoc kwLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("opaque"))) {
    if (!isIdentified)
      return parser.emitError(kwLoc, "only identified structs can be opaque"),
             LLVMStructType();
    if (failed(parser.parseGreater()))
      return LLVMStructType();
    auto type = LLVMStructType::getOpaqueChecked(loc, name);
    if (!type.isOpaque()) {
      parser.emitError(kwLoc, "redeclaring defined struct as opaque");
      return LLVMStructType();
    }
    return type;
  }

  // Check for packedness.
  bool isPacked = succeeded(parser.parseOptionalKeyword("packed"));
  if (failed(parser.parseLParen()))
    return LLVMStructType();

  // Fast pass for structs with zero subtypes.
  if (succeeded(parser.parseOptionalRParen())) {
    if (failed(parser.parseGreater()))
      return LLVMStructType();
    if (!isIdentified)
      return LLVMStructType::getLiteralChecked(loc, {}, isPacked);
    auto type = LLVMStructType::getIdentifiedChecked(loc, name);
    return trySetStructBody(type, {}, isPacked, parser, kwLoc, stack);
  }

  // Parse subtypes. For identified structs, put the identifier of the struct on
  // the stack to support self-references in the recursive calls.
  SmallVector<LLVMType, 4> subtypes;
  llvm::SMLoc subtypesLoc = parser.getCurrentLocation();
  do {
    if (isIdentified)
      stack.insert(name);
    LLVMType type = parseTypeImpl(parser, stack);
    if (!type)
      return LLVMStructType();
    subtypes.push_back(type);
    if (isIdentified)
      stack.pop_back();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseGreater())
    return LLVMStructType();

  // Construct the struct with body.
  if (!isIdentified)
    return LLVMStructType::getLiteralChecked(loc, subtypes, isPacked);
  auto type = LLVMStructType::getIdentifiedChecked(loc, name);
  return trySetStructBody(type, subtypes, isPacked, parser, subtypesLoc, stack);
}

/// Parses one of the LLVM dialect types.
static LLVMType parseTypeImpl(DialectAsmParser &parser,
                              llvm::SetVector<StringRef> &stack) {
  // Special case for integers (i[1-9][0-9]*) that are literals rather than
  // keywords for the parser, so they are not caught by the main dispatch below.
  // Try parsing it a built-in integer type instead.
  Type maybeIntegerType;
  MLIRContext *ctx = parser.getBuilder().getContext();
  llvm::SMLoc keyLoc = parser.getCurrentLocation();
  Location loc = parser.getEncodedSourceLoc(keyLoc);
  OptionalParseResult result = parser.parseOptionalType(maybeIntegerType);
  if (result.hasValue()) {
    if (failed(*result))
      return LLVMType();

    if (!maybeIntegerType.isSignlessInteger()) {
      parser.emitError(keyLoc) << "unexpected type, expected i* or keyword";
      return LLVMType();
    }
    return LLVMIntegerType::getChecked(
        loc, maybeIntegerType.getIntOrFloatBitWidth());
  }

  // Dispatch to concrete functions.
  StringRef key;
  if (failed(parser.parseKeyword(&key)))
    return LLVMType();

  return llvm::StringSwitch<function_ref<LLVMType()>>(key)
      .Case("void", [&] { return LLVMVoidType::get(ctx); })
      .Case("half", [&] { return LLVMHalfType::get(ctx); })
      .Case("bfloat", [&] { return LLVMBFloatType::get(ctx); })
      .Case("float", [&] { return LLVMFloatType::get(ctx); })
      .Case("double", [&] { return LLVMDoubleType::get(ctx); })
      .Case("fp128", [&] { return LLVMFP128Type::get(ctx); })
      .Case("x86_fp80", [&] { return LLVMX86FP80Type::get(ctx); })
      .Case("ppc_fp128", [&] { return LLVMPPCFP128Type::get(ctx); })
      .Case("x86_mmx", [&] { return LLVMX86MMXType::get(ctx); })
      .Case("token", [&] { return LLVMTokenType::get(ctx); })
      .Case("label", [&] { return LLVMLabelType::get(ctx); })
      .Case("metadata", [&] { return LLVMMetadataType::get(ctx); })
      .Case("func", [&] { return parseFunctionType(parser, stack); })
      .Case("ptr", [&] { return parsePointerType(parser, stack); })
      .Case("vec", [&] { return parseVectorType(parser, stack); })
      .Case("array", [&] { return parseArrayType(parser, stack); })
      .Case("struct", [&] { return parseStructType(parser, stack); })
      .Default([&] {
        parser.emitError(keyLoc) << "unknown LLVM type: " << key;
        return LLVMType();
      })();
}

LLVMType mlir::LLVM::detail::parseType(DialectAsmParser &parser) {
  llvm::SetVector<StringRef> stack;
  return parseTypeImpl(parser, stack);
}
