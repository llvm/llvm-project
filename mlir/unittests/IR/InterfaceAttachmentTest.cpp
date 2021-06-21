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
#include "mlir/IR/BuiltinTypes.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestAttributes.h"
#include "../../test/lib/Dialect/Test/TestTypes.h"

using namespace mlir;
using namespace mlir::test;

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
  ASSERT_FALSE(i8.isa<TestExternalTypeInterface>());

  // Attach an interface and check that the type now has the interface.
  IntegerType::attachInterface<Model>(context);
  TestExternalTypeInterface iface = i8.dyn_cast<TestExternalTypeInterface>();
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getBitwidthPlusArg(10), 18u);
  EXPECT_EQ(iface.staticGetSomeValuePlusArg(0), 42u);
  EXPECT_EQ(iface.getBitwidthPlusDoubleArgument(2), 12u);
  EXPECT_EQ(iface.staticGetArgument(17), 17u);

  // Same, but with the default implementation overridden.
  FloatType flt = Float32Type::get(&context);
  ASSERT_FALSE(flt.isa<TestExternalTypeInterface>());
  Float32Type::attachInterface<OverridingModel>(context);
  iface = flt.dyn_cast<TestExternalTypeInterface>();
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getBitwidthPlusArg(10), 42u);
  EXPECT_EQ(iface.staticGetSomeValuePlusArg(10), 52u);
  EXPECT_EQ(iface.getBitwidthPlusDoubleArgument(3), 128u);
  EXPECT_EQ(iface.staticGetArgument(17), 420u);

  // Other contexts shouldn't have the attribute attached.
  MLIRContext other;
  IntegerType i8other = IntegerType::get(&other, 8);
  EXPECT_FALSE(i8other.isa<TestExternalTypeInterface>());
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
    IntegerType elementType = type.cast<VectorType>()
                                  .getElementType()
                                  .dyn_cast_or_null<IntegerType>();
    return elementType ? elementType.getWidth() : 0;
  }
};

TEST(InterfaceAttachment, Fallback) {
  MLIRContext context;

  // Just check that we can attach the interface.
  IntegerType i8 = IntegerType::get(&context, 8);
  ASSERT_FALSE(i8.isa<TestExternalFallbackTypeInterface>());
  IntegerType::attachInterface<TestExternalFallbackTypeIntegerModel>(context);
  ASSERT_TRUE(i8.isa<TestExternalFallbackTypeInterface>());

  // Call the method so it is guaranteed not to be instantiated.
  VectorType vec = VectorType::get({42}, i8);
  ASSERT_FALSE(vec.isa<TestExternalFallbackTypeInterface>());
  VectorType::attachInterface<TestExternalFallbackTypeVectorModel>(context);
  ASSERT_TRUE(vec.isa<TestExternalFallbackTypeInterface>());
  EXPECT_EQ(vec.cast<TestExternalFallbackTypeInterface>().getBitwidth(), 8u);
}

/// External model for attribute interfaces.
struct TextExternalIntegerAttrModel
    : public TestExternalAttrInterface::ExternalModel<
          TextExternalIntegerAttrModel, IntegerAttr> {
  const Dialect *getDialectPtr(Attribute attr) const {
    return &attr.cast<IntegerAttr>().getDialect();
  }

  static int getSomeNumber() { return 42; }
};

TEST(InterfaceAttachment, Attribute) {
  MLIRContext context;

  // Attribute interfaces use the exact same mechanism as types, so just check
  // that the basics work for attributes.
  IntegerAttr attr = IntegerAttr::get(IntegerType::get(&context, 32), 42);
  ASSERT_FALSE(attr.isa<TestExternalAttrInterface>());
  IntegerAttr::attachInterface<TextExternalIntegerAttrModel>(context);
  auto iface = attr.dyn_cast<TestExternalAttrInterface>();
  ASSERT_TRUE(iface != nullptr);
  EXPECT_EQ(iface.getDialectPtr(), &attr.getDialect());
  EXPECT_EQ(iface.getSomeNumber(), 42);
}

} // end namespace
