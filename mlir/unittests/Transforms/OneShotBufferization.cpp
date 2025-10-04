//===- OneShotBufferization.cpp - One-shot bufferization unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {

struct TestTensorAttr : public StringAttr {
  using mlir::StringAttr::StringAttr;

  static bool classof(mlir::Attribute attr) {
    return mlir::isa<mlir::StringAttr>(attr);
  }

  static TestTensorAttr fromStringAttr(StringAttr attr) {
    return mlir::dyn_cast<TestTensorAttr>(attr);
  }
};

class TestTensorEncodingVerifier final
    : public mlir::VerifiableTensorEncoding::ExternalModel<
          TestTensorEncodingVerifier, TestTensorAttr> {
public:
  using ConcreteEntity = mlir::StringAttr;

  mlir::LogicalResult verifyEncoding(
      mlir::Attribute attr, mlir::ArrayRef<int64_t> shape, mlir::Type,
      mlir::function_ref<mlir::InFlightDiagnostic()> emitError) const {
    std::ignore = shape;

    if (mlir::isa<TestTensorAttr>(attr)) {
      return mlir::success();
    }
    return emitError() << "Unknown Tensor enconding: " << attr;
  }
};

struct TestMemRefAttr : public mlir::StringAttr {
  using mlir::StringAttr::StringAttr;

  static bool classof(mlir::Attribute attr) {
    return mlir::isa<mlir::StringAttr>(attr);
  }

  mlir::AffineMap getAffineMap() const {
    return mlir::AffineMap::getMultiDimIdentityMap(1, getContext());
  }
};

class TestMemRefAttrLayout final
    : public mlir::MemRefLayoutAttrInterface::ExternalModel<
          TestMemRefAttrLayout, TestMemRefAttr> {
public:
  using ConcreteEntity = mlir::StringAttr;

  bool isIdentity(mlir::Attribute) const { return true; }
  mlir::AffineMap getAffineMap(mlir::Attribute attr) const {
    return cast<TestMemRefAttr>(attr).getAffineMap();
  }
  mlir::LogicalResult
  verifyLayout(mlir::Attribute attr, mlir::ArrayRef<int64_t> shape,
               mlir::function_ref<mlir::InFlightDiagnostic()> emitError) const {
    std::ignore = shape;

    if (mlir::isa<TestMemRefAttr>(attr)) {
      return mlir::success();
    }
    return emitError() << "Unknown MemRef layout: " << attr;
  }
};

TEST(OneShotBufferizationTest, BufferizeTensorEncodingIntoMemRefLayout) {
  MLIRContext context;
  context.getOrLoadDialect<BuiltinDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<bufferization::BufferizationDialect>();

  DialectRegistry registry;
  registry.addExtension(+[](mlir::MLIRContext *ctx, BuiltinDialect *) {
    TestTensorAttr::attachInterface<TestTensorEncodingVerifier>(*ctx);
    TestMemRefAttr::attachInterface<TestMemRefAttrLayout>(*ctx);
  });
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  context.appendDialectRegistry(registry);

  const char *const code = R"mlir(
    func.func @foo(%t: tensor<42xf32, "hello">)
        -> tensor<42xf32, "hello"> {
      return %t : tensor<42xf32, "hello">
    }

    func.func @bar(%t1: tensor<42xf32, "hello">)
        -> (tensor<42xf32, "hello">, tensor<12xf32, "not hello">) {
      %out1 = func.call @foo(%t1) : (tensor<42xf32, "hello">)
        -> tensor<42xf32, "hello">

      %out2 = bufferization.alloc_tensor() : tensor<12xf32, "not hello">

      return %out1, %out2 : tensor<42xf32, "hello">, tensor<12xf32, "not hello">
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(code, &context);
  ASSERT_NE(module.get(), nullptr) << "parsing should be successful";

  bufferization::OneShotBufferizationOptions options{};
  options.bufferizeFunctionBoundaries = true;
  options.constructMemRefLayoutFn =
      [](TensorType tensor) -> MemRefLayoutAttrInterface {
    assert(isa<RankedTensorType>(tensor) && "tests only builtin tensors");
    auto tensorType = cast<RankedTensorType>(tensor);
    if (auto encoding = dyn_cast<TestTensorAttr>(tensorType.getEncoding())) {
      return cast<MemRefLayoutAttrInterface>(
          TestMemRefAttr::get(tensor.getContext(), encoding.strref()));
    }
    return {};
  };
  options.functionArgTypeConverterFn =
      [&](bufferization::TensorLikeType tensor, Attribute memSpace,
          func::FuncOp, const bufferization::BufferizationOptions &) {
        assert(isa<RankedTensorType>(tensor) && "tests only builtin tensors");
        auto tensorType = cast<RankedTensorType>(tensor);
        auto layout = options.constructMemRefLayoutFn(tensorType);
        return cast<bufferization::BufferLikeType>(
            MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                            layout, memSpace));
      };

  bufferization::BufferizationState state;
  ASSERT_TRUE(succeeded(bufferization::runOneShotModuleBufferize(
      module->getOperation(), options, state)));

  const auto checkType = [](Type type, StringRef expectedLayoutValue) {
    if (auto memref = dyn_cast<MemRefType>(type)) {
      if (auto layout = memref.getLayout();
          isa_and_nonnull<TestMemRefAttr>(layout)) {
        return cast<TestMemRefAttr>(layout) == expectedLayoutValue;
      }
    }
    return false;
  };

  auto fooOp = *module->getOps<func::FuncOp>().begin();
  ASSERT_TRUE(checkType(fooOp.getArgumentTypes()[0], "hello"));
  ASSERT_TRUE(checkType(fooOp.getResultTypes()[0], "hello"));

  auto barOp = *std::next(module->getOps<func::FuncOp>().begin());
  ASSERT_TRUE(checkType(barOp.getArgumentTypes()[0], "hello"));
  ASSERT_TRUE(checkType(barOp.getResultTypes()[0], "hello"));
  ASSERT_TRUE(checkType(barOp.getResultTypes()[1], "not hello"));
}

} // end anonymous namespace
