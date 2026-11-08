//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Register/RegisterLayout.h"
#include "lldb/Register/RegisterDefinition.h"
#include "lldb/Utility/ArchSpec.h"

#include <functional>

using namespace lldb_private;

class RegisterDefinitionTest : public ::testing::Test {
protected:
  std::vector<RegisterDefinition> m_regs;
  std::shared_ptr<const RegisterLayout> m_layout;

  uint32_t
  AddTestRegister(const char *name, const char *group, uint32_t byte_size,
                  std::function<void(const RegisterDefinition &)> adder,
                  std::vector<uint32_t> value_regs = {},
                  std::vector<uint32_t> invalidate_regs = {}) {
    RegisterDefinition new_reg{ConstString(name),
                               ConstString(),
                               ConstString(group),
                               byte_size,
                               LLDB_INVALID_INDEX32,
                               lldb::eEncodingUint,
                               lldb::eFormatUnsigned,
                               LLDB_INVALID_REGNUM,
                               LLDB_INVALID_REGNUM,
                               LLDB_INVALID_REGNUM,
                               static_cast<uint32_t>(m_regs.size()),
                               value_regs,
                               invalidate_regs};
    adder(new_reg);
    return m_regs.size() - 1;
  }

  void ExpectInRegs(uint32_t reg_num, const char *reg_name,
                    std::vector<uint32_t> value_regs,
                    std::vector<uint32_t> invalidate_regs) {
    ASSERT_GT(m_regs.size(), reg_num);

    const RegisterDefinition &reg = m_regs[reg_num];
    ConstString expected_reg_name{reg_name};
    EXPECT_EQ(reg.name, expected_reg_name);
    EXPECT_EQ(reg.value_registers, value_regs);
    EXPECT_EQ(reg.invalidate_registers, invalidate_regs);
  }

  void ExpectInDynInfo(uint32_t reg_num, const char *reg_name,
                       uint32_t byte_offset,
                       std::vector<uint32_t> value_regs = {},
                       std::vector<uint32_t> invalidate_regs = {}) {
    const RegisterDescriptor *reg =
        m_layout->GetDescription()->GetRegisterAtIndex(reg_num);
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->name, reg_name);
    EXPECT_EQ(GetRegisterImageOffset(*m_layout, reg->id), byte_offset);
    EXPECT_THAT(reg->value_registers, value_regs);
    EXPECT_THAT(reg->invalidate_registers, invalidate_regs);
  }
};

#define EXPECT_IN_REGS(reg, ...)                                               \
  {                                                                            \
    SCOPED_TRACE("at register " #reg);                                         \
    ExpectInRegs(reg, #reg, __VA_ARGS__);                                      \
  }

#define EXPECT_IN_DYNINFO(reg, ...)                                            \
  {                                                                            \
    SCOPED_TRACE("at register " #reg);                                         \
    ExpectInDynInfo(reg, #reg, __VA_ARGS__);                                   \
  }

TEST_F(RegisterDefinitionTest, addSupplementaryRegister) {
  // Add a base register
  uint32_t rax =
      AddTestRegister("rax", "group", 8, [this](const RegisterDefinition &r) {
        m_regs.push_back(r);
      });

  // Add supplementary registers
  auto suppl_adder = [this](const RegisterDefinition &r) {
    addSupplementaryRegister(m_regs, r);
  };
  uint32_t eax = AddTestRegister("eax", "supplementary", 4, suppl_adder, {rax});
  uint32_t ax = AddTestRegister("ax", "supplementary", 2, suppl_adder, {rax});
  uint32_t ah = AddTestRegister("ah", "supplementary", 1, suppl_adder, {rax});
  uint32_t al = AddTestRegister("al", "supplementary", 1, suppl_adder, {rax});
  m_regs[ah].value_reg_offset = 1;

  EXPECT_IN_REGS(rax, {}, {eax, ax, ah, al});
  EXPECT_IN_REGS(eax, {rax}, {rax, ax, ah, al});
  EXPECT_IN_REGS(ax, {rax}, {rax, eax, ah, al});
  EXPECT_IN_REGS(ah, {rax}, {rax, eax, ax, al});
  EXPECT_IN_REGS(al, {rax}, {rax, eax, ax, ah});

  m_layout = CreateRegisterImageLayout(std::move(m_regs), ArchSpec());
  ASSERT_NE(m_layout, nullptr);
  EXPECT_IN_DYNINFO(rax, 0, {}, {eax, ax, ah, al});
  EXPECT_IN_DYNINFO(eax, 0, {rax}, {rax, ax, ah, al});
  EXPECT_IN_DYNINFO(ax, 0, {rax}, {rax, eax, ah, al});
  EXPECT_IN_DYNINFO(ah, 1, {rax}, {rax, eax, ax, al});
  EXPECT_IN_DYNINFO(al, 0, {rax}, {rax, eax, ax, ah});
}

TEST_F(RegisterDefinitionTest, FinalizesRelationships) {
  auto adder = [this](const RegisterDefinition &r) { m_regs.push_back(r); };
  // Add regular registers
  uint32_t b1 = AddTestRegister("b1", "base", 8, adder);
  uint32_t b2 = AddTestRegister("b2", "other", 8, adder);

  // Add a few sub-registers
  uint32_t s1 = AddTestRegister("s1", "base", 4, adder, {b1});
  uint32_t s2 = AddTestRegister("s2", "other", 4, adder, {b2});

  // Add a register with invalidate_regs
  uint32_t i1 = AddTestRegister("i1", "third", 8, adder, {}, {b1});

  // Add a register with indirect invalidate regs to be expanded
  // TODO: why is it done conditionally to value_regs?
  uint32_t i2 = AddTestRegister("i2", "third", 4, adder, {b2}, {i1});

  m_layout = CreateRegisterImageLayout(std::move(m_regs), ArchSpec());
  ASSERT_NE(m_layout, nullptr);
  EXPECT_IN_DYNINFO(b1, 0, {}, {});
  EXPECT_IN_DYNINFO(b2, 8, {}, {});
  EXPECT_IN_DYNINFO(s1, 0, {b1}, {});
  EXPECT_IN_DYNINFO(s2, 8, {b2}, {});
  EXPECT_IN_DYNINFO(i1, 16, {}, {b1});
  EXPECT_IN_DYNINFO(i2, 8, {b2}, {b1, i1});
}

TEST(RegisterImageLayoutTest, OwnsSparseMetadataAndExplicitImageLayout) {
  std::shared_ptr<const RegisterLayout> info;
  RegisterDefinition reg;
  reg.name = ConstString("value");
  reg.set_name = ConstString("gpr");
  reg.byte_size = 8;
  reg.byte_offset = 24;
  reg.regnum_remote = 7;
  reg.regnum_dwarf = 19;
  reg.id = 101;
  info = CreateRegisterImageLayout({reg}, ArchSpec("riscv64-unknown-unknown"));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->GetDescription()->GetRegisters().size(), 1u);
  auto model = *info;
  info.reset();
  const auto *value = model.GetDescription()->Find(101);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->numbers[lldb::eRegisterKindDWARF], 19u);
  EXPECT_EQ(value->numbers[lldb::eRegisterKindLLDB], 0u);
  EXPECT_EQ(model.Find(101)->pieces[0].byte_offset, 24u);
  EXPECT_EQ(model.GetDescription()->GetGroups()[0].registers[0], 101u);
}

TEST(RegisterImageLayoutTest, PublishesImmutableSizeGenerations) {
  std::shared_ptr<const RegisterLayout> info;
  RegisterDefinition parent;
  parent.name = ConstString("z0");
  parent.set_name = ConstString("sve");
  parent.byte_size = 16;
  parent.regnum_remote = 0;
  parent.id = 101;
  auto alias = parent;
  alias.name = ConstString("v0");
  alias.byte_size = 8;
  alias.regnum_remote = 1;
  alias.value_registers = {0};
  alias.id = 202;
  info = CreateRegisterImageLayout({parent, alias}, ArchSpec("aarch64"));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->GetDescription()->GetRegisters().size(), 2u);
  auto old = *info;
  auto resized = ReconfigureRegisterImageLayout(
      *info, [](auto registers) { registers[0].byte_size = 32; });
  ASSERT_TRUE(bool(resized));
  info = std::make_shared<RegisterLayout>(std::move(*resized));
  EXPECT_EQ(old.GetDescription()->Find(101)->byte_size, 16u);
  EXPECT_EQ(old.FindBank(0)->byte_size, 16u);
  EXPECT_EQ(info.get()->GetDescription()->Find(101)->byte_size, 32u);
  EXPECT_EQ(info.get()->FindBank(0)->byte_size, 32u);
  EXPECT_EQ(info.get()->GetDescription()->Find(202)->value_registers[0], 101u);
  EXPECT_EQ(info.get()->Find(202)->pieces[0].byte_offset, 0u);
}

TEST(RegisterImageLayoutTest, RejectedGenerationLeavesCurrentModelIntact) {
  std::shared_ptr<const RegisterLayout> info;
  RegisterDefinition reg;
  reg.name = ConstString("value");
  reg.set_name = ConstString("gpr");
  reg.byte_size = 8;
  reg.regnum_remote = 0;
  info = CreateRegisterImageLayout({reg}, ArchSpec("riscv64"));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->GetDescription()->GetRegisters().size(), 1u);
  auto old = info.get()->GetDescription();
  auto resized = ReconfigureRegisterImageLayout(
      *info, [](auto registers) { registers[0].byte_size = 0; });
  ASSERT_FALSE(bool(resized));
  llvm::consumeError(resized.takeError());
  EXPECT_EQ(info.get()->GetDescription(), old);
}

TEST(RegisterImageLayoutTest, BigEndianSliceUsesLogicalLSBPosition) {
  auto data = StructuredData::ParseJSON(R"({"sets": ["gpr"], "registers": [
      {"name": "parent", "bitsize": 64, "offset": 16, "set": 0},
      {"name": "low", "bitsize": 32, "slice": "parent[31:0]", "set": 0},
      {"name": "high", "bitsize": 32, "slice": "parent[63:32]", "set": 0}
  ]})");
  ASSERT_NE(data, nullptr);
  std::shared_ptr<const RegisterLayout> info;
  info = CreateRegisterImageLayout(*data->GetAsDictionary(),
                                   ArchSpec("powerpc64-unknown-freebsd"));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->GetDescription()->GetRegisters().size(), 3u);
  const auto &model = *info;
  const auto *low = model.GetDescription()->Find("low");
  const auto *high = model.GetDescription()->Find("high");
  ASSERT_NE(low, nullptr);
  ASSERT_NE(high, nullptr);
  EXPECT_EQ(low->value_byte_offset, 0u);
  EXPECT_EQ(high->value_byte_offset, 4u);
  EXPECT_EQ(model.Find(low->id)->pieces[0].byte_offset, 20u);
  EXPECT_EQ(model.Find(high->id)->pieces[0].byte_offset, 16u);
}

TEST(RegisterImageLayoutTest, MapsNoncontiguousCompositeAndNestedAlias) {
  std::shared_ptr<const RegisterLayout> info;
  RegisterDefinition low;
  low.name = ConstString("low");
  low.set_name = ConstString("gpr");
  low.byte_size = 4;
  low.byte_offset = 8;
  low.regnum_remote = 0;
  auto high = low;
  high.name = ConstString("high");
  high.byte_offset = 20;
  high.regnum_remote = 1;
  auto combined = low;
  combined.name = ConstString("combined");
  combined.byte_size = 8;
  combined.byte_offset = LLDB_INVALID_INDEX32;
  combined.regnum_remote = 2;
  combined.value_registers = {0, 1};
  auto slice = combined;
  slice.name = ConstString("slice");
  slice.byte_size = 4;
  slice.regnum_remote = 3;
  slice.value_registers = {2};
  slice.value_reg_offset = 2;
  info = CreateRegisterImageLayout({low, high, combined, slice},
                                   ArchSpec("riscv64"));
  ASSERT_NE(info, nullptr);
  EXPECT_EQ(info->GetDescription()->GetRegisters().size(), 4u);
  const auto &model = *info;
  const auto *reg = model.GetDescription()->Find("slice");
  ASSERT_NE(reg, nullptr);
  const auto *location = model.Find(reg->id);
  ASSERT_EQ(location->pieces.size(), 2u);
  EXPECT_EQ(location->pieces[0].byte_offset, 10u);
  EXPECT_EQ(location->pieces[1].byte_offset, 20u);
  std::vector<uint8_t> image(24, 0);
  image[10] = 0x11;
  image[11] = 0x22;
  image[20] = 0x33;
  image[21] = 0x44;
  auto value = model.Read(reg->id, {{0, image}});
  ASSERT_TRUE(bool(value));
  EXPECT_EQ(value->GetAsUInt32(), 0x44332211u);
  auto error =
      model.Write(reg->id, RegisterValue(uint32_t(0xaabbccdd)), {{0, image}});
  ASSERT_FALSE(bool(error));
  EXPECT_EQ(image[10], 0xddu);
  EXPECT_EQ(image[11], 0xccu);
  EXPECT_EQ(image[20], 0xbbu);
  EXPECT_EQ(image[21], 0xaau);
  EXPECT_EQ(image[8], 0u);
  EXPECT_EQ(image[22], 0u);
}

TEST(RegisterImageLayoutTest, EmptyDefinitionsAreAValidUnavailableSource) {
  auto layout =
      CreateRegisterImageLayout(std::vector<RegisterDefinition>{}, ArchSpec());
  ASSERT_NE(layout, nullptr);
  EXPECT_TRUE(layout->GetDescription()->GetRegisters().empty());
  EXPECT_EQ(layout->GetBankSize(0), 0u);
  auto value = layout->Read(0, {});
  ASSERT_FALSE(bool(value));
  llvm::consumeError(value.takeError());
}

TEST(RegisterImageLayoutTest, RepeatedRemoteNumberDoesNotShiftPacketValues) {
  RegisterDefinition pc;
  pc.name = ConstString("pc");
  pc.byte_size = 8;
  pc.regnum_remote = 0;
  auto x0 = pc;
  x0.name = ConstString("x0");
  auto cpsr = pc;
  cpsr.name = ConstString("cpsr");
  cpsr.byte_size = 4;
  cpsr.regnum_remote = 33;
  auto layout = CreateRegisterImageLayout({pc, x0, cpsr}, ArchSpec("aarch64"));
  ASSERT_NE(layout, nullptr);
  const auto &description = *layout->GetDescription();
  EXPECT_EQ(GetRegisterImageOffset(*layout, description.Find("pc")->id), 0u);
  EXPECT_EQ(GetRegisterImageOffset(*layout, description.Find("x0")->id), 0u);
  EXPECT_EQ(GetRegisterImageOffset(*layout, description.Find("cpsr")->id), 8u);
  EXPECT_EQ(layout->GetBankSize(0), 12u);
  const uint8_t image[] = {0x77, 0x77, 0x77, 0x77, 0xee, 0xee,
                          0xee, 0xee, 0x77, 0x77, 0xee, 0xee};
  auto value = layout->Read(description.Find("cpsr")->id, {{0, image}});
  ASSERT_TRUE(bool(value));
  EXPECT_EQ(value->GetAsUInt32(), 0xeeee7777u);
  // The repeated PC definition must remain readable so the process can stop.
  auto pc_value = layout->Read(description.Find("pc")->id, {{0, image}});
  ASSERT_TRUE(bool(pc_value));
  EXPECT_EQ(pc_value->GetAsUInt64(), 0xeeeeeeee77777777ULL);
}
