//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeRegisterContext.h"
#include "TestingSupport/Host/NativeProcessTestUtils.h"
#include "lldb/Host/common/NativeThreadProtocol.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace {
class ModelThread : public NativeThreadProtocol {
public:
  ModelThread(NativeProcessProtocol &process)
      : NativeThreadProtocol(process, 1) {}
  std::string GetName() override { return "model-thread"; }
  StateType GetState() override { return eStateStopped; }
  NativeRegisterContext &GetRegisterContext() override { return *context; }
  bool GetStopReason(ThreadStopInfo &, std::string &) override { return false; }
  Status SetWatchpoint(addr_t, size_t, uint32_t, bool) override { return {}; }
  Status RemoveWatchpoint(addr_t) override { return {}; }
  Status SetHardwareBreakpoint(addr_t, size_t) override { return {}; }
  Status RemoveHardwareBreakpoint(addr_t) override { return {}; }
  NativeRegisterContext *context = nullptr;
};

RegisterLayout MakeModel(ByteOrder order, uint32_t size = 8) {
  RegisterDescriptor parent;
  parent.id = 90;
  parent.name = "rsp";
  parent.byte_size = size;
  parent.numbers[eRegisterKindGeneric] = LLDB_REGNUM_GENERIC_SP;
  parent.numbers[eRegisterKindLLDB] = 500;
  parent.numbers[eRegisterKindDWARF] = 10;
  auto alias = parent;
  alias.id = 7;
  alias.name = "sp";
  alias.byte_size = 4;
  alias.numbers[eRegisterKindGeneric] = LLDB_INVALID_REGNUM;
  alias.numbers[eRegisterKindLLDB] = 42;
  alias.numbers[eRegisterKindDWARF] = 11;
  alias.value_registers = {90};
  alias.value_byte_offset = 1;
  auto description = std::make_shared<RegisterDescription>(
      cantFail(RegisterDescription::Create({alias, parent},
                                           {{"General", "gpr", {90, 7}}})));
  RegisterLayout layout(order);
  cantFail(layout.AddBank(3, size));
  RegisterLocation location;
  location.kind = RegisterLocation::Stored;
  location.pieces = {{3, 0, size, 0}};
  cantFail(layout.Add(90, location));
  location.pieces = {{3, order == eByteOrderLittle ? 1u : size - 5u, 4, 0}};
  cantFail(layout.Add(7, location));
  return llvm::cantFail(
      RegisterLayout::Create(std::move(description), std::move(layout)));
}

class ModelContext : public NativeRegisterContext {
public:
  ModelContext(ModelThread &thread, const ArchSpec &arch, RegisterLayout layout)
      : NativeRegisterContext(thread, arch, std::move(layout)), bytes(8, 0xaa) {
    thread.context = this;
  }
  Expected<RegisterValue> ReadRegister(RegisterID id) override {
    return ReadBankRegister(id, {{3, bytes}});
  }
  Error WriteRegister(RegisterID id, const RegisterValue &value) override {
    return WriteBankRegister(id, value, {{3, bytes}});
  }
  Status ReadAllRegisterValues(WritableDataBufferSP &) override { return {}; }
  Status WriteAllRegisterValues(const DataBufferSP &) override { return {}; }
  std::vector<uint8_t> bytes;
};
} // namespace

TEST(NativeRegisterContextTest, UsesDescriptionNumbersGroupsAndBankValues) {
  NiceMock<MockDelegate> delegate;
  ArchSpec arch("x86_64-unknown-linux");
  MockProcess<NativeProcessProtocol> process(delegate, arch);
  ModelThread thread(process);
  ModelContext context(thread, arch, MakeModel(eByteOrderLittle));
  EXPECT_EQ(context.GetRegisterCount(), 2u);
  EXPECT_EQ(context.GetRegisterIDForNumber(kNumRegisterKinds, 0),
            LLDB_INVALID_REGNUM);
  EXPECT_EQ(context.GetRegisterIDForNumber(eRegisterKindDWARF, 10), 90u);
  EXPECT_EQ(context.GetRegisterIDForNumber(eRegisterKindLLDB, 42), 7u);
  EXPECT_STREQ(context.GetRegisterByName("sp")->name.c_str(), "rsp");
  EXPECT_EQ(context.GetRegisterGroup(0)->registers[0], 90u);
  EXPECT_EQ(context.GetRegisterValueOffset(7), 1u);
  auto value = context.ReadRegister(7);
  ASSERT_THAT_EXPECTED(value, Succeeded());
  EXPECT_EQ(value->GetAsUInt32(), 0xaaaaaaaau);
  ASSERT_THAT_ERROR(
      context.WriteRegister(7, RegisterValue(uint32_t(0x12345678))),
      Succeeded());
  EXPECT_EQ(context.bytes, (std::vector<uint8_t>{0xaa, 0x78, 0x56, 0x34, 0x12,
                                                 0xaa, 0xaa, 0xaa}));
}

TEST(NativeRegisterContextTest, RetainsOldGenerationAndUsesTargetAliasOffset) {
  NiceMock<MockDelegate> delegate;
  ArchSpec arch("powerpc64-unknown-freebsd");
  MockProcess<NativeProcessProtocol> process(delegate, arch);
  ModelThread thread(process);
  ModelContext context(thread, arch, MakeModel(eByteOrderBig));
  auto old = context.GetDescription();
  EXPECT_EQ(context.GetRegisterValueOffset(7), 3u);
  context.SetRegisterLayout(MakeModel(eByteOrderBig, 16));
  EXPECT_EQ(old->Find(90)->byte_size, 8u);
  EXPECT_EQ(context.GetDescription()->Find(90)->byte_size, 16u);
  EXPECT_EQ(context.GetRegisterValueOffset(7), 11u);
  EXPECT_EQ(context.GetBankSize(3), 16u);
}

TEST(NativeRegisterContextTest, ValueAccessUsesIDsAndPropagatesErrors) {
  NiceMock<MockDelegate> delegate;
  ArchSpec arch("x86_64-unknown-linux");
  MockProcess<NativeProcessProtocol> process(delegate, arch);
  ModelThread thread(process);
  ModelContext context(thread, arch, MakeModel(eByteOrderLittle));

  // Neither a presentation index nor an external number is a logical ID.
  EXPECT_THAT_EXPECTED(context.ReadRegister(0), Failed());
  EXPECT_THAT_EXPECTED(context.ReadRegister(42), Failed());
  EXPECT_THAT_ERROR(context.WriteRegisterFromUnsigned(500, 1), Failed());
  EXPECT_EQ(context.ReadRegisterAsUnsigned(500, 123), 123u);
  EXPECT_EQ(context.GetSP(), 0xaaaaaaaaaaaaaaaaULL);
  EXPECT_EQ(context.GetExpeditedRegisters(ExpeditedRegs::Minimal),
            (std::vector<RegisterID>{90}));
  EXPECT_EQ(context.GetExpeditedRegisters(ExpeditedRegs::Full),
            (std::vector<RegisterID>{90, 7}));

  ASSERT_THAT_ERROR(context.SetSP(0x12345678).ToError(), Succeeded());
  EXPECT_EQ(context.GetSP(), 0x12345678u);
  EXPECT_EQ(context.ReadRegisterAsUnsigned(90, 0), 0x12345678u);

  context.bytes.resize(2);
  EXPECT_THAT_EXPECTED(context.ReadRegister(90), Failed());
  EXPECT_THAT_ERROR(context.WriteRegister(90, RegisterValue(uint64_t(0))),
                    Failed());
}
