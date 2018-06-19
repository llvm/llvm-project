#include "Target.h"

#include <cassert>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

void InitializeX86ExegesisTarget();

namespace {

using testing::NotNull;

class X86TargetTest : public ::testing::Test {
protected:
  static void SetUpTestCase() { InitializeX86ExegesisTarget(); }
};

TEST_F(X86TargetTest, Lookup) {
  EXPECT_THAT(ExegesisTarget::lookup("x86_64-unknown-linux"), NotNull());
}

} // namespace
} // namespace exegesis
