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
  ValueSource() {
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

