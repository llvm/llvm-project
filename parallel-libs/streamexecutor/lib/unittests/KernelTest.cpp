//===-- KernelTest.cpp - Tests for Kernel objects -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for the code in Kernel.
///
//===----------------------------------------------------------------------===//

#include <cassert>

#include "streamexecutor/Executor.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/PlatformInterfaces.h"

#include "llvm/ADT/STLExtras.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

// An Executor that returns a dummy KernelInterface.
//
// During construction it creates a unique_ptr to a dummy KernelInterface and it
// also stores a separate copy of the raw pointer that is stored by that
// unique_ptr.
//
// The expectation is that the code being tested will call the
// getKernelImplementation method and will thereby take ownership of the
// unique_ptr, but the copy of the raw pointer will stay behind in this mock
// object. The raw pointer copy can then be used to identify the unique_ptr in
// its new location (by comparing the raw pointer with unique_ptr::get), to
// verify that the unique_ptr ended up where it was supposed to be.
class MockExecutor : public se::Executor {
public:
  MockExecutor()
      : se::Executor(nullptr), Unique(llvm::make_unique<se::KernelInterface>()),
        Raw(Unique.get()) {}

  // Moves the unique pointer into the returned se::Expected instance.
  //
  // Asserts that it is not called again after the unique pointer has been moved
  // out.
  se::Expected<std::unique_ptr<se::KernelInterface>>
  getKernelImplementation(const se::MultiKernelLoaderSpec &) override {
    assert(Unique && "MockExecutor getKernelImplementation should not be "
                     "called more than once");
    return std::move(Unique);
  }

  // Gets the copy of the raw pointer from the original unique pointer.
  const se::KernelInterface *getRaw() const { return Raw; }

private:
  std::unique_ptr<se::KernelInterface> Unique;
  const se::KernelInterface *Raw;
};

// Test fixture class for typed tests for KernelBase.getImplementation.
//
// The only purpose of this class is to provide a name that types can be bound
// to in the gtest infrastructure.
template <typename T> class GetImplementationTest : public ::testing::Test {};

// Types used with the GetImplementationTest fixture class.
typedef ::testing::Types<se::KernelBase, se::TypedKernel<>,
                         se::TypedKernel<int>>
    GetImplementationTypes;

TYPED_TEST_CASE(GetImplementationTest, GetImplementationTypes);

// Tests that the kernel create functions properly fetch the implementation
// pointers for the kernel objects they construct from the passed-in
// Executor objects.
TYPED_TEST(GetImplementationTest, SetImplementationDuringCreate) {
  se::MultiKernelLoaderSpec Spec;
  MockExecutor MockExecutor;

  auto MaybeKernel = TypeParam::create(&MockExecutor, Spec);
  EXPECT_TRUE(static_cast<bool>(MaybeKernel));
  se::KernelInterface *Implementation = MaybeKernel->getImplementation();
  EXPECT_EQ(MockExecutor.getRaw(), Implementation);
}

} // namespace
