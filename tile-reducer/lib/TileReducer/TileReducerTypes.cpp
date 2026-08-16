//===- TileReducerTypes.cpp - TileReducer types -----------------*- C++ -*-===//

#include "TileReducer/TileReducerTypes.h"

#include "TileReducer/TileReducerDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tr;

#define GET_TYPEDEF_CLASSES
#include "TileReducer/TileReducerOpsTypes.cpp.inc"

void TileReducerDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TileReducer/TileReducerOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Shared shape printer / parser
//
//   dim ::= integer | `?` | identifier
//   type ::= `<` (dim `x`)* element-type `>`
//
// A named dim (`M`) is a dynamic extent that pretty-prints as that name.
// `parseXInDimensionList` handles the `xf32` juxtaposition.
//===----------------------------------------------------------------------===//

static void printShape(AsmPrinter &p, ArrayRef<int64_t> shape, Type elem,
                       ArrayRef<std::string> names = {}) {
  p << "<";
  for (int64_t i = 0, e = static_cast<int64_t>(shape.size()); i < e; ++i) {
    if (i < static_cast<int64_t>(names.size()) && !names[i].empty())
      p << names[i];
    else if (ShapedType::isDynamic(shape[i]))
      p << "?";
    else
      p << shape[i];
    p << "x";
  }
  p.printType(elem);
  p << ">";
}

/// `MxKxf32` lexes as one identifier. Split on `x`; the last piece is the
/// element type and the rest are named (dynamic) dimensions.
static ParseResult splitJuxtaposed(StringRef ident, AsmParser &parser,
                                   SmallVectorImpl<int64_t> &shape,
                                   SmallVectorImpl<std::string> *names,
                                   Type &elem, bool allowDynamic) {
  SmallVector<StringRef> parts;
  ident.split(parts, 'x', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (parts.size() < 2)
    return parser.emitError(parser.getNameLoc(),
                            "expected <dims x element-type>, got '")
           << ident << "'";

  StringRef typeStr = parts.pop_back_val();
  elem = parseType(typeStr, parser.getContext());
  if (!elem)
    return parser.emitError(parser.getNameLoc(),
                            "failed to parse element type '")
           << typeStr << "'";

  for (StringRef part : parts) {
    int64_t value = 0;
    if (!part.getAsInteger(10, value)) {
      if (value < 0)
        return parser.emitError(parser.getNameLoc(),
                                "dimension must be non-negative");
      shape.push_back(value);
      if (names)
        names->emplace_back();
      continue;
    }
    if (part == "?") {
      if (!allowDynamic)
        return parser.emitError(parser.getNameLoc(),
                                "tile dimensions must be static");
      shape.push_back(ShapedType::kDynamic);
      if (names)
        names->emplace_back();
      continue;
    }
    if (!allowDynamic)
      return parser.emitError(parser.getNameLoc(),
                              "tile dimensions must be static, got '")
             << part << "'";
    if (!llvm::all_of(part, [](char c) { return llvm::isAlpha(c); }))
      return parser.emitError(parser.getNameLoc(),
                              "invalid dimension name '")
             << part << "'";
    shape.push_back(ShapedType::kDynamic);
    if (names)
      names->push_back(part.str());
  }
  return success();
}

static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape,
                              Type &elem, SmallVectorImpl<std::string> *names,
                              bool allowDynamic) {
  if (parser.parseLess())
    return failure();

  while (true) {
    Type maybeType;
    OptionalParseResult typeRes = parser.parseOptionalType(maybeType);
    if (typeRes.has_value() && succeeded(*typeRes)) {
      elem = maybeType;
      break;
    }

    int64_t value = 0;
    OptionalParseResult intRes = parser.parseOptionalInteger(value);
    if (intRes.has_value()) {
      if (failed(*intRes))
        return failure();
      if (value < 0)
        return parser.emitError(parser.getNameLoc(),
                                "dimension must be non-negative");
      shape.push_back(value);
      if (names)
        names->emplace_back();
      if (failed(parser.parseXInDimensionList()))
        return failure();
      continue;
    }

    if (succeeded(parser.parseOptionalQuestion())) {
      if (!allowDynamic)
        return parser.emitError(parser.getNameLoc(),
                                "tile dimensions must be static");
      shape.push_back(ShapedType::kDynamic);
      if (names)
        names->emplace_back();
      if (failed(parser.parseXInDimensionList()))
        return failure();
      continue;
    }

    StringRef ident;
    if (succeeded(parser.parseOptionalKeyword(&ident)))
      return splitJuxtaposed(ident, parser, shape, names, elem, allowDynamic);

    return parser.emitError(parser.getNameLoc(), "expected dimension or type");
  }
  return parser.parseGreater();
}

static LogicalResult verifyElement(function_ref<InFlightDiagnostic()> emitError,
                                   Type elem) {
  if (isa<FloatType, IntegerType>(elem))
    return success();
  return emitError() << "element type must be an integer or float, got "
                     << elem;
}

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

LogicalResult TileType::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty())
    return emitError() << "tile must have rank >= 1";
  for (int64_t d : shape) {
    if (ShapedType::isDynamic(d))
      return emitError() << "tile dimensions must be static";
    if (d <= 0)
      return emitError() << "tile dimensions must be positive, got " << d;
  }
  return verifyElement(emitError, elementType);
}

Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elem;
  if (failed(parseShape(parser, shape, elem, /*names=*/nullptr,
                        /*allowDynamic=*/false)))
    return {};
  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), shape, elem);
}

void TileType::print(AsmPrinter &printer) const {
  printShape(printer, getShape(), getElementType());
}

//===----------------------------------------------------------------------===//
// BufferType
//===----------------------------------------------------------------------===//

LogicalResult BufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<int64_t> shape, Type elementType,
                                 ArrayRef<std::string> dimNames) {
  if (shape.empty())
    return emitError() << "buffer must have rank >= 1";
  if (!dimNames.empty() && dimNames.size() != shape.size())
    return emitError() << "expected " << shape.size()
                       << " dimension names, got " << dimNames.size();
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (!ShapedType::isDynamic(d) && d < 0)
      return emitError() << "static dimension must be non-negative";
    if (!dimNames.empty() && !dimNames[i].empty() && !ShapedType::isDynamic(d))
      return emitError() << "named dimension '" << dimNames[i]
                         << "' must be dynamic";
  }
  return verifyElement(emitError, elementType);
}

Type BufferType::parse(AsmParser &parser) {
  SmallVector<int64_t> shape;
  SmallVector<std::string> names;
  Type elem;
  if (failed(parseShape(parser, shape, elem, &names, /*allowDynamic=*/true)))
    return {};
  return getChecked([&] { return parser.emitError(parser.getNameLoc()); },
                    parser.getContext(), shape, elem, names);
}

void BufferType::print(AsmPrinter &printer) const {
  printShape(printer, getShape(), getElementType(), getDimNames());
}
