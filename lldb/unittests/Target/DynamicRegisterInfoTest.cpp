//===-- DynamicRegisterInfoTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Target/DynamicRegisterInfo.h"
#include "lldb/Utility/ArchSpec.h"

#include <functional>

using namespace lldb_private;

static std::vector<uint32_t> regs_to_vector(uint32_t *regs) {
  std::vector<uint32_t> ret;
  if (regs) {
    while (*regs != LLDB_INVALID_REGNUM)
      ret.push_back(*regs++);
  }
  return ret;
}

class DynamicRegisterInfoRegisterTest : public ::testing::Test {
protected:
  std::vector<DynamicRegisterInfo::Register> m_regs;
  DynamicRegisterInfo m_dyninfo;

  uint32_t AddTestRegister(
      const char *name, const char *group, uint32_t byte_size,
      std::function<void(const DynamicRegisterInfo::Register &)> adder,
      std::vector<uint32_t> value_regs = {},
      std::vector<uint32_t> invalidate_regs = {}) {
    DynamicRegisterInfo::Register new_reg{ConstString(name),
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

    const DynamicRegisterInfo::Register &reg = m_regs[reg_num];
    ConstString expected_reg_name{reg_name};
    EXPECT_EQ(reg.name, expected_reg_name);
    EXPECT_EQ(reg.value_regs, value_regs);
    EXPECT_EQ(reg.invalidate_regs, invalidate_regs);
  }

  void ExpectInDynInfo(uint32_t reg_num, const char *reg_name,
                       uint32_t byte_offset,
                       std::vector<uint32_t> value_regs = {},
                       std::vector<uint32_t> invalidate_regs = {}) {
    const RegisterInfo *reg = m_dyninfo.GetRegisterInfoAtIndex(reg_num);
    ASSERT_NE(reg, nullptr);
    EXPECT_STREQ(reg->name, reg_name);
    EXPECT_EQ(reg->byte_offset, byte_offset);
    EXPECT_THAT(regs_to_vector(reg->value_regs), value_regs);
    EXPECT_THAT(regs_to_vector(reg->invalidate_regs), invalidate_regs);
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

TEST_F(DynamicRegisterInfoRegisterTest, addSupplementaryRegister) {
  // Add a base register
  uint32_t rax = AddTestRegister(
      "rax", "group", 8,
      [this](const DynamicRegisterInfo::Register &r) { m_regs.push_back(r); });

  // Add supplementary registers
  auto suppl_adder = [this](const DynamicRegisterInfo::Register &r) {
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

  EXPECT_EQ(m_dyninfo.SetRegisterInfo(std::move(m_regs), ArchSpec()),
            m_regs.size());
  EXPECT_IN_DYNINFO(rax, 0, {}, {eax, ax, ah, al});
  EXPECT_IN_DYNINFO(eax, 0, {rax}, {rax, ax, ah, al});
  EXPECT_IN_DYNINFO(ax, 0, {rax}, {rax, eax, ah, al});
  EXPECT_IN_DYNINFO(ah, 1, {rax}, {rax, eax, ax, al});
  EXPECT_IN_DYNINFO(al, 0, {rax}, {rax, eax, ax, ah});
}

TEST_F(DynamicRegisterInfoRegisterTest, SetRegisterInfo) {
  auto adder = [this](const DynamicRegisterInfo::Register &r) {
    m_regs.push_back(r);
  };
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

  EXPECT_EQ(m_dyninfo.SetRegisterInfo(std::move(m_regs), ArchSpec()),
            m_regs.size());
  EXPECT_IN_DYNINFO(b1, 0, {}, {});
  EXPECT_IN_DYNINFO(b2, 8, {}, {});
  EXPECT_IN_DYNINFO(s1, 0, {b1}, {});
  EXPECT_IN_DYNINFO(s2, 8, {b2}, {});
  EXPECT_IN_DYNINFO(i1, 16, {}, {b1});
  EXPECT_IN_DYNINFO(i2, 8, {b2}, {b1, i1});
}
