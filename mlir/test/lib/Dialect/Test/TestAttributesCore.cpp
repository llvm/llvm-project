//===- TestAttributesCore.cpp - Test attributes in `Core` -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the attributes of the test dialect in partition `Core`.
//
//===----------------------------------------------------------------------===//

#include "TestAttributesCore.h"
#include "TestDialectBase.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// CompoundAAttr
//===----------------------------------------------------------------------===//

Attribute CompoundAAttr::parse(AsmParser &parser, Type type) {
  int widthOfSomething;
  Type oneType;
  SmallVector<int, 4> arrayOfInts;
  if (parser.parseLess() || parser.parseInteger(widthOfSomething) ||
      parser.parseComma() || parser.parseType(oneType) || parser.parseComma() ||
      parser.parseLSquare())
    return Attribute();

  int intVal;
  while (!*parser.parseOptionalInteger(intVal)) {
    arrayOfInts.push_back(intVal);
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRSquare() || parser.parseGreater())
    return Attribute();
  return get(parser.getContext(), widthOfSomething, oneType, arrayOfInts);
}

void CompoundAAttr::print(AsmPrinter &printer) const {
  printer << "<" << getWidthOfSomething() << ", " << getOneType() << ", [";
  llvm::interleaveComma(getArrayOfInts(), printer);
  printer << "]>";
}

Attribute TestI64ElementsAttr::parse(AsmParser &parser, Type type) {
  SmallVector<uint64_t> elements;
  if (parser.parseLess() || parser.parseLSquare())
    return Attribute();
  uint64_t intVal;
  while (succeeded(*parser.parseOptionalInteger(intVal))) {
    elements.push_back(intVal);
    if (parser.parseOptionalComma())
      break;
  }

  if (parser.parseRSquare() || parser.parseGreater())
    return Attribute();
  return parser.getChecked<TestI64ElementsAttr>(
      parser.getContext(), llvm::cast<ShapedType>(type), elements);
}

void TestI64ElementsAttr::print(AsmPrinter &printer) const {
  printer << "<[";
  llvm::interleaveComma(getElements(), printer);
  printer << "]>";
}

LogicalResult
TestI64ElementsAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ShapedType type, ArrayRef<uint64_t> elements) {
  if (type.getNumElements() != static_cast<int64_t>(elements.size())) {
    return emitError()
           << "number of elements does not match the provided shape type, got: "
           << elements.size() << ", but expected: " << type.getNumElements();
  }
  if (type.getRank() != 1 || !type.getElementType().isSignlessInteger(64))
    return emitError() << "expected single rank 64-bit shape type, but got: "
                       << type;
  return success();
}

//===----------------------------------------------------------------------===//
// TestSubElementsAccessAttr
//===----------------------------------------------------------------------===//

Attribute TestSubElementsAccessAttr::parse(::mlir::AsmParser &parser,
                                           ::mlir::Type type) {
  Attribute first, second, third;
  if (parser.parseLess() || parser.parseAttribute(first) ||
      parser.parseComma() || parser.parseAttribute(second) ||
      parser.parseComma() || parser.parseAttribute(third) ||
      parser.parseGreater()) {
    return {};
  }
  return get(parser.getContext(), first, second, third);
}

void TestSubElementsAccessAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<" << getFirst() << ", " << getSecond() << ", " << getThird()
          << ">";
}

//===----------------------------------------------------------------------===//
// TestExtern1DI64ElementsAttr
//===----------------------------------------------------------------------===//

ArrayRef<uint64_t> TestExtern1DI64ElementsAttr::getElements() const {
  if (auto *blob = getHandle().getBlob())
    return blob->getDataAs<uint64_t>();
  return {};
}

//===----------------------------------------------------------------------===//
// TestCopyCountAttr Implementation
//===----------------------------------------------------------------------===//

LogicalResult TestCopyCountAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> /*emitError*/,
    CopyCount /*copy_count*/) {
  return success();
}

//===----------------------------------------------------------------------===//
// TestSymbolRefAttr
//===----------------------------------------------------------------------===//

LogicalResult
TestSymbolRefAttr::verifySymbolUses(Operation *op,
                                    SymbolTableCollection &symbolTable) const {
  // Verify that the referenced symbol exists
  if (!symbolTable.lookupNearestSymbolFrom<SymbolOpInterface>(op, getSymbol()))
    return op->emitOpError()
           << "TestSymbolRefAttr::verifySymbolUses: '" << getSymbol()
           << "' does not reference a valid symbol";
  return success();
}

//===----------------------------------------------------------------------===//
// Generated Attribute Definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CopyCountAttr Implementation
//===----------------------------------------------------------------------===//

CopyCount::CopyCount(const CopyCount &rhs) : value(rhs.value) {
  CopyCount::counter++;
}

CopyCount &CopyCount::operator=(const CopyCount &rhs) {
  CopyCount::counter++;
  value = rhs.value;
  return *this;
}

int CopyCount::counter;

static bool operator==(const test::CopyCount &lhs, const test::CopyCount &rhs) {
  return lhs.value == rhs.value;
}

llvm::raw_ostream &test::operator<<(llvm::raw_ostream &os,
                                    const test::CopyCount &value) {
  return os << value.value;
}

template <>
struct mlir::FieldParser<test::CopyCount> {
  static FailureOr<test::CopyCount> parse(AsmParser &parser) {
    std::string value;
    if (parser.parseKeyword(value))
      return failure();
    return test::CopyCount(value);
  }
};
namespace test {
static llvm::hash_code hash_value(const test::CopyCount &copyCount) {
  return llvm::hash_value(copyCount.value);
}
} // namespace test

//===----------------------------------------------------------------------===//
// TestOpAsmAttrInterfaceAttr
//===----------------------------------------------------------------------===//

::mlir::OpAsmDialectInterface::AliasResult
TestOpAsmAttrInterfaceAttr::getAlias(::llvm::raw_ostream &os) const {
  os << "op_asm_attr_interface_";
  os << getValue().getValue();
  return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
}

//===----------------------------------------------------------------------===//
// TestConstMemorySpaceAttr
//===----------------------------------------------------------------------===//

bool TestConstMemorySpaceAttr::isValidLoad(
    Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool TestConstMemorySpaceAttr::isValidStore(
    Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (emitError)
    emitError() << "memory space is read-only";
  return false;
}

bool TestConstMemorySpaceAttr::isValidAtomicOp(
    mlir::ptr::AtomicBinOp binOp, Type type, mlir::ptr::AtomicOrdering ordering,
    std::optional<int64_t> alignment, const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (emitError)
    emitError() << "memory space is read-only";
  return false;
}

bool TestConstMemorySpaceAttr::isValidAtomicXchg(
    Type type, mlir::ptr::AtomicOrdering successOrdering,
    mlir::ptr::AtomicOrdering failureOrdering, std::optional<int64_t> alignment,
    const ::mlir::DataLayout *dataLayout,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (emitError)
    emitError() << "memory space is read-only";
  return false;
}

bool TestConstMemorySpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src, function_ref<InFlightDiagnostic()> emitError) const {
  if (emitError)
    emitError() << "memory space doesn't allow addrspace casts";
  return false;
}

bool TestConstMemorySpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    function_ref<InFlightDiagnostic()> emitError) const {
  if (emitError)
    emitError() << "memory space doesn't allow int-ptr casts";
  return false;
}

//===----------------------------------------------------------------------===//
// Tablegen Generated Definitions
//===----------------------------------------------------------------------===//

#include "TestAttrInterfaces.cpp.inc"
#include "TestOpEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "TestAttributesCore.cpp.inc"

//===----------------------------------------------------------------------===//
// Dynamic Attributes
//===----------------------------------------------------------------------===//

/// Define a singleton dynamic attribute.
static std::unique_ptr<DynamicAttrDefinition>
getDynamicSingletonAttr(TestDialect *testDialect) {
  return DynamicAttrDefinition::get(
      "dynamic_singleton", testDialect,
      [](function_ref<InFlightDiagnostic()> emitError,
         ArrayRef<Attribute> args) {
        if (!args.empty()) {
          emitError() << "expected 0 attribute arguments, but had "
                      << args.size();
          return failure();
        }
        return success();
      });
}

/// Define a dynamic attribute representing a pair or attributes.
static std::unique_ptr<DynamicAttrDefinition>
getDynamicPairAttr(TestDialect *testDialect) {
  return DynamicAttrDefinition::get(
      "dynamic_pair", testDialect,
      [](function_ref<InFlightDiagnostic()> emitError,
         ArrayRef<Attribute> args) {
        if (args.size() != 2) {
          emitError() << "expected 2 attribute arguments, but had "
                      << args.size();
          return failure();
        }
        return success();
      });
}

static std::unique_ptr<DynamicAttrDefinition>
getDynamicCustomAssemblyFormatAttr(TestDialect *testDialect) {
  auto verifier = [](function_ref<InFlightDiagnostic()> emitError,
                     ArrayRef<Attribute> args) {
    if (args.size() != 2) {
      emitError() << "expected 2 attribute arguments, but had " << args.size();
      return failure();
    }
    return success();
  };

  auto parser = [](AsmParser &parser,
                   llvm::SmallVectorImpl<Attribute> &parsedParams) {
    Attribute leftAttr, rightAttr;
    if (parser.parseLess() || parser.parseAttribute(leftAttr) ||
        parser.parseColon() || parser.parseAttribute(rightAttr) ||
        parser.parseGreater())
      return failure();
    parsedParams.push_back(leftAttr);
    parsedParams.push_back(rightAttr);
    return success();
  };

  auto printer = [](AsmPrinter &printer, ArrayRef<Attribute> params) {
    printer << "<" << params[0] << ":" << params[1] << ">";
  };

  return DynamicAttrDefinition::get("dynamic_custom_assembly_format",
                                    testDialect, std::move(verifier),
                                    std::move(parser), std::move(printer));
}

//===----------------------------------------------------------------------===//
// TestCustomStorageCtorAttr
//===----------------------------------------------------------------------===//

test::detail::TestCustomStorageCtorAttrAttrStorage *
test::detail::TestCustomStorageCtorAttrAttrStorage::construct(
    mlir::StorageUniquer::StorageAllocator &, std::tuple<int> &&) {
  // Note: this tests linker error ("undefined symbol"), the actual
  // implementation is not important.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TestTensorEncodingAttr
//===----------------------------------------------------------------------===//

::llvm::LogicalResult TestTensorEncodingAttr::verifyEncoding(
    mlir::ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TestMemRefLayoutAttr
//===----------------------------------------------------------------------===//

mlir::AffineMap TestMemRefLayoutAttr::getAffineMap() const {
  return mlir::AffineMap::getMultiDimIdentityMap(1, getContext());
}

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

void TestDialect::registerAttributes() {
  registerTestDialectAttributes(this);
  registerDynamicAttr(getDynamicSingletonAttr(this));
  registerDynamicAttr(getDynamicPairAttr(this));
  registerDynamicAttr(getDynamicCustomAssemblyFormatAttr(this));
}
