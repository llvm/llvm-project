//===-- VectorLayoutInterfaceTest.cpp - Unit Tests for Vector Layouts -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::detail;

class NamedStridedLayoutAttrStorage : public AttributeStorage {
public:
  using KeyTy =
      std::tuple<ArrayRef<std::string>, ArrayRef<int64_t>, ArrayRef<int64_t>>;

  NamedStridedLayoutAttrStorage(ArrayRef<std::string> names,
                                ArrayRef<int64_t> strides,
                                ArrayRef<int64_t> vectorShape)
      : names(names), strides(strides), vectorShape(vectorShape) {}

  bool operator==(const KeyTy &key) const {
    return (std::get<0>(key) == names) && (std::get<1>(key) == strides) &&
           (std::get<2>(key) == vectorShape);
  }

  static NamedStridedLayoutAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    ArrayRef<std::string> names = allocator.copyInto(std::get<0>(key));
    ArrayRef<int64_t> strides = allocator.copyInto(std::get<1>(key));
    ArrayRef<int64_t> vectorShape = allocator.copyInto(std::get<2>(key));
    return new (allocator.allocate<NamedStridedLayoutAttrStorage>())
        NamedStridedLayoutAttrStorage(names, strides, vectorShape);
  }

  ArrayRef<std::string> names;
  ArrayRef<int64_t> strides;
  ArrayRef<int64_t> vectorShape;
};

struct NamedStridedLayoutAttr
    : public Attribute::AttrBase<NamedStridedLayoutAttr, Attribute,
                                 NamedStridedLayoutAttrStorage,
                                 VectorLayoutAttrInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NamedStridedLayoutAttr)
  using Base::Base;
  static NamedStridedLayoutAttr get(MLIRContext *ctx,
                                    ArrayRef<std::string> names,
                                    ArrayRef<int64_t> strides,
                                    ArrayRef<int64_t> vectorShape) {
    return Base::get(ctx, names, strides, vectorShape);
  }

  LogicalResult verifyLayout(ArrayRef<int64_t> shape, Type elementType,
                             function_ref<InFlightDiagnostic()> emitError) {
    if (shape == getVectorShape())
      return success();
    return failure();
  }

  ArrayRef<std::string> getNames() { return getImpl()->names; }
  ArrayRef<int64_t> getStrides() { return getImpl()->strides; }
  ArrayRef<int64_t> getVectorShape() { return getImpl()->vectorShape; }
};

struct VLTestDialect : Dialect {
  explicit VLTestDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<VLTestDialect>()) {
    ctx->loadDialect<VLTestDialect>();
    addAttributes<NamedStridedLayoutAttr>();
  }
  static StringRef getDialectNamespace() { return "vltest"; }

  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override {
    auto layoutAttr = llvm::cast<NamedStridedLayoutAttr>(attr);
    SmallVector<int64_t> mutableVectorShape(layoutAttr.getVectorShape());
    size_t i{0}, j{0};
    auto addCommaIf = [&](bool condition) {
      if (condition)
        printer << ", ";
    };
    auto addLParenIf = [&](bool condition) {
      if (condition)
        printer << "[";
    };
    auto addRParenIf = [&](bool condition) {
      if (condition)
        printer << "]";
    };
    for (const auto &[name, stride] :
         llvm::zip(layoutAttr.getNames(), layoutAttr.getStrides())) {
      addLParenIf(j == 0);
      printer << name << " : " << stride;
      mutableVectorShape[i] /= stride;
      addCommaIf(mutableVectorShape[i] > 1);
      bool finishedParsingList = mutableVectorShape[i] == 1;
      addRParenIf(finishedParsingList);
      addCommaIf(finishedParsingList && (i < mutableVectorShape.size() - 1));
      j = finishedParsingList ? 0 : j + 1;
      i = finishedParsingList ? i + 1 : i;
    }
  }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override {
    SmallVector<int64_t> strides, vectorShape;
    SmallVector<std::string> names;
    if (!succeeded(parser.parseKeyword("named_strided_layout")))
      return {};
    if (!succeeded(parser.parseLess()))
      return {};
    do {
      if (!succeeded(parser.parseLSquare()))
        return {};
      std::string name;
      int64_t stride;
      int64_t shape = 1;
      do {
        if (succeeded(parser.parseString(&name)) &&
            succeeded(parser.parseColon()) &&
            succeeded(parser.parseInteger(stride))) {
          names.push_back(name);
          strides.push_back(stride);
          shape *= stride;
        }
      } while (succeeded(parser.parseOptionalComma()));
      if (!succeeded(parser.parseRSquare()))
        return {};
      vectorShape.push_back(shape);
    } while (succeeded(parser.parseOptionalComma()));
    if (!succeeded(parser.parseGreater()))
      return {};
    return NamedStridedLayoutAttr::get(parser.getContext(), names, strides,
                                       vectorShape);
  }
};

TEST(VectorLayoutAttrInterface, NamedStridedLayout) {
  const char *ir = R"MLIR(
    #layout = #vltest.named_strided_layout<["BatchX" : 2, "LaneX" : 4, "VectorX" : 2],
                                           ["BatchY" : 1, "LaneY" : 8, "VectorY" : 2]>
    %lhs = "arith.constant"() {value = dense<0.0> : vector<16x16xf16, #layout>}
        : () -> (vector<16x16xf16, #layout>)
  )MLIR";

  DialectRegistry registry;
  registry.insert<VLTestDialect, arith::ArithDialect>();
  MLIRContext ctx(registry);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);

  arith::ConstantOp op =
      llvm::cast<arith::ConstantOp>(module->getBody()->getOperations().front());
  Type type = op.getResult().getType();
  if (auto vectorType = llvm::cast<VectorType>(type)) {
    VectorLayoutAttrInterface layout = vectorType.getLayout();
    auto namedStridedLayout = llvm::cast<NamedStridedLayoutAttr>(layout);
    ArrayRef<std::string> names = namedStridedLayout.getNames();
    ArrayRef<int64_t> strides = namedStridedLayout.getStrides();
    ArrayRef<int64_t> vectorShape = namedStridedLayout.getVectorShape();
    EXPECT_EQ(vectorShape.size(), 2u);
    EXPECT_EQ(vectorShape[0], 16u);
    EXPECT_EQ(vectorShape[1], 16u);
    EXPECT_EQ(strides.size(), 6u);
    EXPECT_EQ(strides[0], 2u);
    EXPECT_EQ(strides[1], 4u);
    EXPECT_EQ(strides[2], 2u);
    EXPECT_EQ(strides[3], 1u);
    EXPECT_EQ(strides[4], 8u);
    EXPECT_EQ(strides[5], 2u);
    EXPECT_EQ(names.size(), 6u);
    EXPECT_EQ(names[0], "BatchX");
    EXPECT_EQ(names[1], "LaneX");
    EXPECT_EQ(names[2], "VectorX");
    EXPECT_EQ(names[3], "BatchY");
    EXPECT_EQ(names[4], "LaneY");
    EXPECT_EQ(names[5], "VectorY");
  }
}

TEST(VectorLayoutAttrInterface, RoundTripTest) {
  const char *ir = R"MLIR(
    #layout = #vltest.named_strided_layout<["BatchX" : 2, "LaneX" : 4, "VectorX" : 2],
                                           ["BatchY" : 1, "LaneY" : 8, "VectorY" : 2]>
    %lhs = "arith.constant"() {value = dense<0.0> : vector<16x16xf16, #layout>}
        : () -> (vector<16x16xf16, #layout>)
  )MLIR";

  DialectRegistry registry;
  registry.insert<VLTestDialect, arith::ArithDialect>();
  MLIRContext ctx(registry);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  std::string moduleStr;
  llvm::raw_string_ostream stream(moduleStr);
  stream << *module;
  stream.flush();
  const std::string expectedResult =
      "module {\n"
      "  %cst = arith.constant dense<0.000000e+00> :"
      " vector<16x16xf16, #vltest<[BatchX : 2, LaneX : 4, VectorX : 2],"
      " [BatchY : 1, LaneY : 8, VectorY : 2]>>\n}";
  EXPECT_EQ(moduleStr, expectedResult);
}
