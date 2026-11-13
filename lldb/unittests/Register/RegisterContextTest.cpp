//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/RegisterContext.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

namespace {
// A value source needs neither a Thread nor a serialized register image.
class ValueSource : public RegisterContext {
public:
  explicit ValueSource(std::shared_ptr<const void> owner = {})
      : RegisterContext(std::move(owner)) {
    RegisterDescriptor reg;
    reg.id = 91;
    reg.name = "value";
    reg.byte_size = 8;
    reg.numbers[lldb::eRegisterKindDWARF] = 3;
    description = std::make_shared<RegisterDescription>(
        cantFail(RegisterDescription::Create({reg})));
  }
  std::shared_ptr<const RegisterDescription> GetDescription() const override {
    return description;
  }
  Expected<RegisterValue> ReadRegister(RegisterID id) override {
    if (!description->Find(id))
      return createStringError("unknown test register");
    return RegisterValue(value);
  }
  Error WriteRegister(RegisterID id, const RegisterValue &input) override {
    if (!description->Find(id))
      return createStringError("unknown test register");
    value = input.GetAsUInt64();
    return Error::success();
  }
  Expected<RegisterCheckpoint> SaveCheckpoint() override {
    return CreateCheckpoint([this, schema = description,
                             saved = value]() -> llvm::Error {
      auto &context = *this;
      if (context.description != schema)
        return createStringError("test register configuration changed");
      context.value = saved;
      return Error::success();
    });
  }
  void ReplaceSchema() {
    description = std::make_shared<RegisterDescription>(*description);
  }

private:
  std::shared_ptr<const RegisterDescription> description;
  uint64_t value = 0;
};
} // namespace

TEST(RegisterContextTest, ResolvesExternalNumbersToIDs) {
  ValueSource source;
  auto id = source.ResolveRegisterNumber(lldb::eRegisterKindDWARF, 3);
  ASSERT_THAT_EXPECTED(id, Succeeded());
  EXPECT_EQ(*id, 91u);
  EXPECT_EQ(source.GetRegisterIDForNumber(lldb::eRegisterKindDWARF, 3), 91u);
  EXPECT_THAT_EXPECTED(source.ReadRegister(0), Failed());
  ASSERT_THAT_ERROR(source.WriteRegister(*id, RegisterValue(uint64_t(42))),
                    Succeeded());
  auto value = source.ReadRegister(*id);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt64(), 42u);
  ASSERT_THAT_ERROR(source.WriteRegisterFromUnsigned(*id, 99), Succeeded());
  EXPECT_EQ(source.ReadRegisterAsUnsigned(*id, 0), 99u);
  EXPECT_EQ(source.ReadRegisterAsUnsigned(3, 123), 123u);
  EXPECT_THAT_ERROR(source.WriteRegisterFromUnsigned(3, 0), Failed());
}

TEST(RegisterContextTest, RestoresOpaqueValueState) {
  ValueSource source;
  ASSERT_THAT_ERROR(source.WriteRegister(91, RegisterValue(uint64_t(42))),
                    Succeeded());
  auto checkpoint = source.SaveCheckpoint();
  ASSERT_THAT_EXPECTED(checkpoint, Succeeded());
  ASSERT_THAT_ERROR(source.WriteRegister(91, RegisterValue(uint64_t(99))),
                    Succeeded());
  ASSERT_THAT_ERROR(source.RestoreCheckpoint(*checkpoint), Succeeded());
  auto value = source.ReadRegister(91);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt64(), 42u);
}

TEST(RegisterContextTest, RejectsAnotherOwnerAndAnotherConfiguration) {
  ValueSource first, second;
  auto checkpoint = first.SaveCheckpoint();
  ASSERT_THAT_EXPECTED(checkpoint, Succeeded());
  EXPECT_THAT_ERROR(second.RestoreCheckpoint(*checkpoint), Failed());
  first.ReplaceSchema();
  EXPECT_THAT_ERROR(first.RestoreCheckpoint(*checkpoint), Failed());
}

TEST(RegisterContextTest, ExpiredOwnerCannotRestoreIntoReplacement) {
  auto source = std::make_unique<ValueSource>();
  auto checkpoint = source->SaveCheckpoint();
  ASSERT_THAT_EXPECTED(checkpoint, Succeeded());
  source.reset();
  ValueSource replacement;
  EXPECT_THAT_ERROR(replacement.RestoreCheckpoint(*checkpoint), Failed());
}

TEST(RegisterContextTest, RestoreOperationStaysBoundToItsBackend) {
  auto owner = std::make_shared<int>(0);
  ValueSource original(owner), another_frame(owner);
  ASSERT_THAT_ERROR(original.WriteRegisterFromUnsigned(91, 42), Succeeded());
  auto checkpoint = original.SaveCheckpoint();
  ASSERT_THAT_EXPECTED(checkpoint, Succeeded());
  ASSERT_THAT_ERROR(original.WriteRegisterFromUnsigned(91, 99), Succeeded());
  ASSERT_THAT_ERROR(another_frame.WriteRegisterFromUnsigned(91, 123),
                    Succeeded());
  ASSERT_THAT_ERROR(another_frame.RestoreCheckpoint(*checkpoint), Succeeded());
  EXPECT_EQ(original.ReadRegisterAsUnsigned(91, 0), 42u);
  EXPECT_EQ(another_frame.ReadRegisterAsUnsigned(91, 0), 123u);
}
