//===- InterfaceAttachmentTest.cpp - Test attaching interfaces ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the tests for attaching interfaces to attributes and types
// without having to specify them on the attribute or type class directly.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestAttributes.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestTypes.h"
#include "mlir/IR/OwningOpRef.h"

using namespace mlir;
using namespace test;

namespace {

/// External interface model for the integer type. Only provides non-default
/// methods.
struct Model
    : public TestExternalTypeInterface::ExternalModel<Model, IntegerType> {
  unsigned getBitwidthPlusArg(Type type, unsigned arg) const {
    return type.getIntOrFloatBitWidth() + arg;
  }

  static unsigned staticGetSomeValuePlusArg(unsigned arg) { return 42 + arg; }
};

/// External interface model for the float type. Provides non-deafult and
/// overrides default methods.
struct OverridingModel
    : public TestExternalTypeInterface::ExternalModel<OverridingModel,
                                                      FloatType> {
  unsigned getBitwidthPlusArg(Type type, unsigned arg) const {
    return type.getIntOrFloatBitWidth() + arg;
  }

  static unsigned staticGetSomeValuePlusArg(unsigned arg) { return 42 + arg; }

  unsigned getBitwidthPlusDoubleArgument(Type type, unsigned arg) const {
    return 128;
  }

  static unsigned staticGetArgument(unsigned arg) { return 420; }
};

TEST(InterfaceAttachment, Type) {
  MLIRContext context;

  // Check that the type has no interface.
  IntegerType i8 = IntegerType::get(&context, 8);
  ASSERT_FALSE(isa<TestExternalTypeInterface>(i8));

  // Attach an interface and check that the type now has the interface.
  IntegerType::attachInterface<Model>(context);
  TestExternalTypeInterface iface = dyn_cast<TestExternalTypeInterface>(i8);
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getBitwidthPlusArg(10), 18u);
  EXPECT_EQ(iface.staticGetSomeValuePlusArg(0), 42u);
  EXPECT_EQ(iface.getBitwidthPlusDoubleArgument(2), 12u);
  EXPECT_EQ(iface.staticGetArgument(17), 17u);

  // Same, but with the default implementation overridden.
  FloatType flt = Float32Type::get(&context);
  ASSERT_FALSE(isa<TestExternalTypeInterface>(flt));
  Float32Type::attachInterface<OverridingModel>(context);
  iface = dyn_cast<TestExternalTypeInterface>(flt);
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getBitwidthPlusArg(10), 42u);
  EXPECT_EQ(iface.staticGetSomeValuePlusArg(10), 52u);
  EXPECT_EQ(iface.getBitwidthPlusDoubleArgument(3), 128u);
  EXPECT_EQ(iface.staticGetArgument(17), 420u);

  // Other contexts shouldn't have the attribute attached.
  MLIRContext other;
  IntegerType i8other = IntegerType::get(&other, 8);
  EXPECT_FALSE(isa<TestExternalTypeInterface>(i8other));
}

/// External interface model for the test type from the test dialect.
struct TestTypeModel
    : public TestExternalTypeInterface::ExternalModel<TestTypeModel,
                                                      test::TestType> {
  unsigned getBitwidthPlusArg(Type type, unsigned arg) const { return arg; }

  static unsigned staticGetSomeValuePlusArg(unsigned arg) { return 10 + arg; }
};

TEST(InterfaceAttachment, TypeDelayedContextConstruct) {
  // Put the interface in the registry.
  DialectRegistry registry;
  registry.insert<test::TestDialect>();
  registry.addExtension(+[](MLIRContext *ctx, test::TestDialect *dialect) {
    test::TestType::attachInterface<TestTypeModel>(*ctx);
  });

  // Check that when a context is constructed with the given registry, the type
  // interface gets registered.
  MLIRContext context(registry);
  context.loadDialect<test::TestDialect>();
  test::TestType testType = test::TestType::get(&context);
  auto iface = dyn_cast<TestExternalTypeInterface>(testType);
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getBitwidthPlusArg(42), 42u);
  EXPECT_EQ(iface.staticGetSomeValuePlusArg(10), 20u);
}

TEST(InterfaceAttachment, TypeDelayedContextAppend) {
  // Put the interface in the registry.
  DialectRegistry registry;
  registry.insert<test::TestDialect>();
  registry.addExtension(+[](MLIRContext *ctx, test::TestDialect *dialect) {
    test::TestType::attachInterface<TestTypeModel>(*ctx);
  });

  // Check that when the registry gets appended to the context, the interface
  // becomes available for objects in loaded dialects.
  MLIRContext context;
  context.loadDialect<test::TestDialect>();
  test::TestType testType = test::TestType::get(&context);
  EXPECT_FALSE(isa<TestExternalTypeInterface>(testType));
  context.appendDialectRegistry(registry);
  EXPECT_TRUE(isa<TestExternalTypeInterface>(testType));
}

TEST(InterfaceAttachment, RepeatedRegistration) {
  DialectRegistry registry;
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    IntegerType::attachInterface<Model>(*ctx);
  });
  MLIRContext context(registry);

  // Should't fail on repeated registration through the dialect registry.
  context.appendDialectRegistry(registry);
}

TEST(InterfaceAttachment, TypeBuiltinDelayed) {
  // Builtin dialect needs to registration or loading, but delayed interface
  // registration must still work.
  DialectRegistry registry;
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    IntegerType::attachInterface<Model>(*ctx);
  });

  MLIRContext context(registry);
  IntegerType i16 = IntegerType::get(&context, 16);
  EXPECT_TRUE(isa<TestExternalTypeInterface>(i16));

  MLIRContext initiallyEmpty;
  IntegerType i32 = IntegerType::get(&initiallyEmpty, 32);
  EXPECT_FALSE(isa<TestExternalTypeInterface>(i32));
  initiallyEmpty.appendDialectRegistry(registry);
  EXPECT_TRUE(isa<TestExternalTypeInterface>(i32));
}

/// The interface provides a default implementation that expects
/// ConcreteType::getWidth to exist, which is the case for IntegerType. So this
/// just derives from the ExternalModel.
struct TestExternalFallbackTypeIntegerModel
    : public TestExternalFallbackTypeInterface::ExternalModel<
          TestExternalFallbackTypeIntegerModel, IntegerType> {};

/// The interface provides a default implementation that expects
/// ConcreteType::getWidth to exist, which is *not* the case for VectorType. Use
/// FallbackModel instead to override this and make sure the code still compiles
/// because we never instantiate the ExternalModel class template with a
/// template argument that would have led to compilation failures.
struct TestExternalFallbackTypeVectorModel
    : public TestExternalFallbackTypeInterface::FallbackModel<
          TestExternalFallbackTypeVectorModel> {
  unsigned getBitwidth(Type type) const {
    IntegerType elementType =
        dyn_cast_or_null<IntegerType>(cast<VectorType>(type).getElementType());
    return elementType ? elementType.getWidth() : 0;
  }
};

TEST(InterfaceAttachment, Fallback) {
  MLIRContext context;

  // Just check that we can attach the interface.
  IntegerType i8 = IntegerType::get(&context, 8);
  ASSERT_FALSE(isa<TestExternalFallbackTypeInterface>(i8));
  IntegerType::attachInterface<TestExternalFallbackTypeIntegerModel>(context);
  ASSERT_TRUE(isa<TestExternalFallbackTypeInterface>(i8));

  // Call the method so it is guaranteed not to be instantiated.
  VectorType vec = VectorType::get({42}, i8);
  ASSERT_FALSE(isa<TestExternalFallbackTypeInterface>(vec));
  VectorType::attachInterface<TestExternalFallbackTypeVectorModel>(context);
  ASSERT_TRUE(isa<TestExternalFallbackTypeInterface>(vec));
  EXPECT_EQ(cast<TestExternalFallbackTypeInterface>(vec).getBitwidth(), 8u);
}

/// External model for attribute interfaces.
struct TestExternalIntegerAttrModel
    : public TestExternalAttrInterface::ExternalModel<
          TestExternalIntegerAttrModel, IntegerAttr> {
  const Dialect *getDialectPtr(Attribute attr) const {
    return &cast<IntegerAttr>(attr).getDialect();
  }

  static int getSomeNumber() { return 42; }
};

TEST(InterfaceAttachment, Attribute) {
  MLIRContext context;

  // Attribute interfaces use the exact same mechanism as types, so just check
  // that the basics work for attributes.
  IntegerAttr attr = IntegerAttr::get(IntegerType::get(&context, 32), 42);
  ASSERT_FALSE(isa<TestExternalAttrInterface>(attr));
  IntegerAttr::attachInterface<TestExternalIntegerAttrModel>(context);
  auto iface = dyn_cast<TestExternalAttrInterface>(attr);
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getDialectPtr(), &attr.getDialect());
  EXPECT_EQ(iface.getSomeNumber(), 42);
}

/// External model for an interface attachable to a non-builtin attribute.
struct TestExternalSimpleAAttrModel
    : public TestExternalAttrInterface::ExternalModel<
          TestExternalSimpleAAttrModel, test::SimpleAAttr> {
  const Dialect *getDialectPtr(Attribute attr) const {
    return &attr.getDialect();
  }

  static int getSomeNumber() { return 21; }
};

TEST(InterfaceAttachmentTest, AttributeDelayed) {
  // Attribute interfaces use the exact same mechanism as types, so just check
  // that the delayed registration work for attributes.
  DialectRegistry registry;
  registry.insert<test::TestDialect>();
  registry.addExtension(+[](MLIRContext *ctx, test::TestDialect *dialect) {
    test::SimpleAAttr::attachInterface<TestExternalSimpleAAttrModel>(*ctx);
  });

  MLIRContext context(registry);
  context.loadDialect<test::TestDialect>();
  auto attr = test::SimpleAAttr::get(&context);
  EXPECT_TRUE(isa<TestExternalAttrInterface>(attr));

  MLIRContext initiallyEmpty;
  initiallyEmpty.loadDialect<test::TestDialect>();
  attr = test::SimpleAAttr::get(&initiallyEmpty);
  EXPECT_FALSE(isa<TestExternalAttrInterface>(attr));
  initiallyEmpty.appendDialectRegistry(registry);
  EXPECT_TRUE(isa<TestExternalAttrInterface>(attr));
}

/// External interface model for the module operation. Only provides non-default
/// methods.
struct TestExternalOpModel
    : public TestExternalOpInterface::ExternalModel<TestExternalOpModel,
                                                    ModuleOp> {
  unsigned getNameLengthPlusArg(Operation *op, unsigned arg) const {
    return op->getName().getStringRef().size() + arg;
  }

  static unsigned getNameLengthPlusArgTwice(unsigned arg) {
    return ModuleOp::getOperationName().size() + 2 * arg;
  }
};

/// External interface model for the func operation. Provides non-deafult and
/// overrides default methods.
struct TestExternalOpOverridingModel
    : public TestExternalOpInterface::FallbackModel<
          TestExternalOpOverridingModel> {
  unsigned getNameLengthPlusArg(Operation *op, unsigned arg) const {
    return op->getName().getStringRef().size() + arg;
  }

  static unsigned getNameLengthPlusArgTwice(unsigned arg) {
    return UnrealizedConversionCastOp::getOperationName().size() + 2 * arg;
  }

  unsigned getNameLengthTimesArg(Operation *op, unsigned arg) const {
    return 42;
  }

  static unsigned getNameLengthMinusArg(unsigned arg) { return 21; }
};

TEST(InterfaceAttachment, Operation) {
  MLIRContext context;
  OpBuilder builder(&context);

  // Initially, the operation doesn't have the interface.
  OwningOpRef<ModuleOp> moduleOp =
      builder.create<ModuleOp>(UnknownLoc::get(&context));
  ASSERT_FALSE(isa<TestExternalOpInterface>(moduleOp->getOperation()));

  // We can attach an external interface and now the operaiton has it.
  ModuleOp::attachInterface<TestExternalOpModel>(context);
  auto iface = dyn_cast<TestExternalOpInterface>(moduleOp->getOperation());
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getNameLengthPlusArg(10), 24u);
  EXPECT_EQ(iface.getNameLengthTimesArg(3), 42u);
  EXPECT_EQ(iface.getNameLengthPlusArgTwice(18), 50u);
  EXPECT_EQ(iface.getNameLengthMinusArg(5), 9u);

  // Default implementation can be overridden.
  OwningOpRef<UnrealizedConversionCastOp> castOp =
      builder.create<UnrealizedConversionCastOp>(UnknownLoc::get(&context),
                                                 TypeRange(), ValueRange());
  ASSERT_FALSE(isa<TestExternalOpInterface>(castOp->getOperation()));
  UnrealizedConversionCastOp::attachInterface<TestExternalOpOverridingModel>(
      context);
  iface = dyn_cast<TestExternalOpInterface>(castOp->getOperation());
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getNameLengthPlusArg(10), 44u);
  EXPECT_EQ(iface.getNameLengthTimesArg(0), 42u);
  EXPECT_EQ(iface.getNameLengthPlusArgTwice(8), 50u);
  EXPECT_EQ(iface.getNameLengthMinusArg(1000), 21u);

  // Another context doesn't have the interfaces registered.
  MLIRContext other;
  OwningOpRef<ModuleOp> otherModuleOp =
      ModuleOp::create(UnknownLoc::get(&other));
  ASSERT_FALSE(isa<TestExternalOpInterface>(otherModuleOp->getOperation()));
}

template <class ConcreteOp>
struct TestExternalTestOpModel
    : public TestExternalOpInterface::ExternalModel<
          TestExternalTestOpModel<ConcreteOp>, ConcreteOp> {
  unsigned getNameLengthPlusArg(Operation *op, unsigned arg) const {
    return op->getName().getStringRef().size() + arg;
  }

  static unsigned getNameLengthPlusArgTwice(unsigned arg) {
    return ConcreteOp::getOperationName().size() + 2 * arg;
  }
};

TEST(InterfaceAttachment, OperationDelayedContextConstruct) {
  DialectRegistry registry;
  registry.insert<test::TestDialect>();
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    ModuleOp::attachInterface<TestExternalOpModel>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, test::TestDialect *dialect) {
    test::OpJ::attachInterface<TestExternalTestOpModel<test::OpJ>>(*ctx);
    test::OpH::attachInterface<TestExternalTestOpModel<test::OpH>>(*ctx);
  });

  // Construct the context directly from a registry. The interfaces are
  // expected to be readily available on operations.
  MLIRContext context(registry);
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());
  auto opJ =
      builder.create<test::OpJ>(builder.getUnknownLoc(), builder.getI32Type());
  auto opH =
      builder.create<test::OpH>(builder.getUnknownLoc(), opJ.getResult());
  auto opI =
      builder.create<test::OpI>(builder.getUnknownLoc(), opJ.getResult());

  EXPECT_TRUE(isa<TestExternalOpInterface>(module->getOperation()));
  EXPECT_TRUE(isa<TestExternalOpInterface>(opJ.getOperation()));
  EXPECT_TRUE(isa<TestExternalOpInterface>(opH.getOperation()));
  EXPECT_FALSE(isa<TestExternalOpInterface>(opI.getOperation()));
}

TEST(InterfaceAttachment, OperationDelayedContextAppend) {
  DialectRegistry registry;
  registry.insert<test::TestDialect>();
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    ModuleOp::attachInterface<TestExternalOpModel>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, test::TestDialect *dialect) {
    test::OpJ::attachInterface<TestExternalTestOpModel<test::OpJ>>(*ctx);
    test::OpH::attachInterface<TestExternalTestOpModel<test::OpH>>(*ctx);
  });

  // Construct the context, create ops, and only then append the registry. The
  // interfaces are expected to be available after appending the registry.
  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(module->getBody(), module->getBody()->begin());
  auto opJ =
      builder.create<test::OpJ>(builder.getUnknownLoc(), builder.getI32Type());
  auto opH =
      builder.create<test::OpH>(builder.getUnknownLoc(), opJ.getResult());
  auto opI =
      builder.create<test::OpI>(builder.getUnknownLoc(), opJ.getResult());

  EXPECT_FALSE(isa<TestExternalOpInterface>(module->getOperation()));
  EXPECT_FALSE(isa<TestExternalOpInterface>(opJ.getOperation()));
  EXPECT_FALSE(isa<TestExternalOpInterface>(opH.getOperation()));
  EXPECT_FALSE(isa<TestExternalOpInterface>(opI.getOperation()));

  context.appendDialectRegistry(registry);

  EXPECT_TRUE(isa<TestExternalOpInterface>(module->getOperation()));
  EXPECT_TRUE(isa<TestExternalOpInterface>(opJ.getOperation()));
  EXPECT_TRUE(isa<TestExternalOpInterface>(opH.getOperation()));
  EXPECT_FALSE(isa<TestExternalOpInterface>(opI.getOperation()));
}

} // namespace
