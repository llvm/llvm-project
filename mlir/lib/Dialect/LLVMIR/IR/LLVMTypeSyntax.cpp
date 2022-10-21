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
  if (isCompatibleType(type) && !type.isa<IntegerType, FloatType, VectorType>())
    return mlir::LLVM::detail::printType(type, printer);
  printer.printType(type);
}

/// Returns the keyword to use for the given type.
static StringRef getTypeKeyword(Type type) {
  return TypeSwitch<Type, StringRef>(type)
      .Case<LLVMVoidType>([&](Type) { return "void"; })
      .Case<LLVMPPCFP128Type>([&](Type) { return "ppc_fp128"; })
      .Case<LLVMX86MMXType>([&](Type) { return "x86_mmx"; })
      .Case<LLVMTokenType>([&](Type) { return "token"; })
      .Case<LLVMLabelType>([&](Type) { return "label"; })
      .Case<LLVMMetadataType>([&](Type) { return "metadata"; })
      .Case<LLVMFunctionType>([&](Type) { return "func"; })
      .Case<LLVMPointerType>([&](Type) { return "ptr"; })
      .Case<LLVMFixedVectorType, LLVMScalableVectorType>(
          [&](Type) { return "vec"; })
      .Case<LLVMArrayType>([&](Type) { return "array"; })
      .Case<LLVMStructType>([&](Type) { return "struct"; })
      .Default([](Type) -> StringRef {
        llvm_unreachable("unexpected 'llvm' type kind");
      });
}

/// Prints a structure type. Keeps track of known struct names to handle self-
/// or mutually-referring structs without falling into infinite recursion.
static void printStructType(AsmPrinter &printer, LLVMStructType type) {
  // This keeps track of the names of identified structure types that are
  // currently being printed. Since such types can refer themselves, this
  // tracking is necessary to stop the recursion: the current function may be
  // called recursively from AsmPrinter::printType after the appropriate
  // dispatch. We maintain the invariant of this storage being modified
  // exclusively in this function, and at most one name being added per call.
  // TODO: consider having such functionality inside AsmPrinter.
  thread_local SetVector<StringRef> knownStructNames;
  unsigned stackSize = knownStructNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownStructNames.size() == stackSize &&
           "malformed identified stack when printing recursive structs");
  });

  printer << "<";
  if (type.isIdentified()) {
    printer << '"' << type.getName() << '"';
    // If we are printing a reference to one of the enclosing structs, just
    // print the name and stop to avoid infinitely long output.
    if (knownStructNames.count(type.getName())) {
      printer << '>';
      return;
    }
    printer << ", ";
  }

  if (type.isIdentified() && type.isOpaque()) {
    printer << "opaque>";
    return;
  }

  if (type.isPacked())
    printer << "packed ";

  // Put the current type on stack to avoid infinite recursion.
  printer << '(';
  if (type.isIdentified())
    knownStructNames.insert(type.getName());
  llvm::interleaveComma(type.getBody(), printer.getStream(),
                        [&](Type subtype) { dispatchPrint(printer, subtype); });
  if (type.isIdentified())
    knownStructNames.pop_back();
  printer << ')';
  printer << '>';
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
            LLVMScalableVectorType, LLVMFunctionType>(
          [&](auto type) { type.print(printer); })
      .Case([&](LLVMStructType structType) {
        printStructType(printer, structType);
      });
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
  //  - two elements, the first dynamic (indicated by ShapedType::kDynamicSize)
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

/// Attempts to set the body of an identified structure type. Reports a parsing
/// error at `subtypesLoc` in case of failure.
static LLVMStructType trySetStructBody(LLVMStructType type,
                                       ArrayRef<Type> subtypes, bool isPacked,
                                       AsmParser &parser, SMLoc subtypesLoc) {
  for (Type t : subtypes) {
    if (!LLVMStructType::isValidElementType(t)) {
      parser.emitError(subtypesLoc)
          << "invalid LLVM structure element type: " << t;
      return LLVMStructType();
    }
  }

  if (succeeded(type.setBody(subtypes, isPacked)))
    return type;

  parser.emitError(subtypesLoc)
      << "identified type already used with a different body";
  return LLVMStructType();
}

/// Parses an LLVM dialect structure type.
///   llvm-type ::= `struct<` (string-literal `,`)? `packed`?
///                 `(` llvm-type-list `)` `>`
///               | `struct<` string-literal `>`
///               | `struct<` string-literal `, opaque>`
static LLVMStructType parseStructType(AsmParser &parser) {
  // This keeps track of the names of identified structure types that are
  // currently being parsed. Since such types can refer themselves, this
  // tracking is necessary to stop the recursion: the current function may be
  // called recursively from AsmParser::parseType after the appropriate
  // dispatch. We maintain the invariant of this storage being modified
  // exclusively in this function, and at most one name being added per call.
  // TODO: consider having such functionality inside AsmParser.
  thread_local SetVector<StringRef> knownStructNames;
  unsigned stackSize = knownStructNames.size();
  (void)stackSize;
  auto guard = llvm::make_scope_exit([&]() {
    assert(knownStructNames.size() == stackSize &&
           "malformed identified stack when parsing recursive structs");
  });

  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  if (failed(parser.parseLess()))
    return LLVMStructType();

  // If we are parsing a self-reference to a recursive struct, i.e. the parsing
  // stack already contains a struct with the same identifier, bail out after
  // the name.
  std::string name;
  bool isIdentified = succeeded(parser.parseOptionalString(&name));
  if (isIdentified) {
    if (knownStructNames.count(name)) {
      if (failed(parser.parseGreater()))
        return LLVMStructType();
      return LLVMStructType::getIdentifiedChecked(
          [loc] { return emitError(loc); }, loc.getContext(), name);
    }
    if (failed(parser.parseComma()))
      return LLVMStructType();
  }

  // Handle intentionally opaque structs.
  SMLoc kwLoc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("opaque"))) {
    if (!isIdentified)
      return parser.emitError(kwLoc, "only identified structs can be opaque"),
             LLVMStructType();
    if (failed(parser.parseGreater()))
      return LLVMStructType();
    auto type = LLVMStructType::getOpaqueChecked(
        [loc] { return emitError(loc); }, loc.getContext(), name);
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
      return LLVMStructType::getLiteralChecked([loc] { return emitError(loc); },
                                               loc.getContext(), {}, isPacked);
    auto type = LLVMStructType::getIdentifiedChecked(
        [loc] { return emitError(loc); }, loc.getContext(), name);
    return trySetStructBody(type, {}, isPacked, parser, kwLoc);
  }

  // Parse subtypes. For identified structs, put the identifier of the struct on
  // the stack to support self-references in the recursive calls.
  SmallVector<Type, 4> subtypes;
  SMLoc subtypesLoc = parser.getCurrentLocation();
  do {
    if (isIdentified)
      knownStructNames.insert(name);
    Type type;
    if (dispatchParse(parser, type))
      return LLVMStructType();
    subtypes.push_back(type);
    if (isIdentified)
      knownStructNames.pop_back();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseGreater())
    return LLVMStructType();

  // Construct the struct with body.
  if (!isIdentified)
    return LLVMStructType::getLiteralChecked(
        [loc] { return emitError(loc); }, loc.getContext(), subtypes, isPacked);
  auto type = LLVMStructType::getIdentifiedChecked(
      [loc] { return emitError(loc); }, loc.getContext(), name);
  return trySetStructBody(type, subtypes, isPacked, parser, subtypesLoc);
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
      .Case("x86_mmx", [&] { return LLVMX86MMXType::get(ctx); })
      .Case("token", [&] { return LLVMTokenType::get(ctx); })
      .Case("label", [&] { return LLVMLabelType::get(ctx); })
      .Case("metadata", [&] { return LLVMMetadataType::get(ctx); })
      .Case("func", [&] { return LLVMFunctionType::parse(parser); })
      .Case("ptr", [&] { return LLVMPointerType::parse(parser); })
      .Case("vec", [&] { return parseVectorType(parser); })
      .Case("array", [&] { return LLVMArrayType::parse(parser); })
      .Case("struct", [&] { return parseStructType(parser); })
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

ParseResult LLVM::parsePrettyLLVMType(AsmParser &p, FailureOr<Type> &type) {
  type.emplace();
  return dispatchParse(p, *type);
}

void LLVM::printPrettyLLVMType(AsmPrinter &p, Type type) {
  return dispatchPrint(p, type);
}
