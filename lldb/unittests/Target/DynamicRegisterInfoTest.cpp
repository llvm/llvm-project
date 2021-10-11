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

class DynamicRegisterInfoTest : public ::testing::Test {
protected:
  DynamicRegisterInfo info;
  uint32_t next_regnum = 0;
  ConstString group{"group"};

  uint32_t AddTestRegister(const char *name, uint32_t byte_size,
                           std::vector<uint32_t> value_regs = {},
                           std::vector<uint32_t> invalidate_regs = {}) {
    struct RegisterInfo new_reg {
      name, nullptr, byte_size, LLDB_INVALID_INDEX32, lldb::eEncodingUint,
          lldb::eFormatUnsigned,
          {LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,
           next_regnum, next_regnum},
          nullptr, nullptr
    };

    if (!value_regs.empty()) {
      value_regs.push_back(LLDB_INVALID_REGNUM);
      new_reg.value_regs = value_regs.data();
    }
    if (!invalidate_regs.empty()) {
      invalidate_regs.push_back(LLDB_INVALID_REGNUM);
      new_reg.invalidate_regs = invalidate_regs.data();
    }

    info.AddRegister(new_reg, group);
    return next_regnum++;
  }

  void AssertRegisterInfo(uint32_t reg_num, const char *reg_name,
                          uint32_t byte_offset,
                          std::vector<uint32_t> value_regs = {},
                          std::vector<uint32_t> invalidate_regs = {}) {
    const RegisterInfo *reg = info.GetRegisterInfoAtIndex(reg_num);
    EXPECT_NE(reg, nullptr);
    if (!reg)
      return;

    EXPECT_STREQ(reg->name, reg_name);
    EXPECT_EQ(reg->byte_offset, byte_offset);
    EXPECT_THAT(regs_to_vector(reg->value_regs), value_regs);
    EXPECT_THAT(regs_to_vector(reg->invalidate_regs), invalidate_regs);
  }
};

#define ASSERT_REG(reg, ...)                                                   \
  {                                                                            \
    SCOPED_TRACE("at register " #reg);                                         \
    AssertRegisterInfo(reg, #reg, __VA_ARGS__);                                \
  }

TEST_F(DynamicRegisterInfoTest, finalize_regs) {
  // Add regular registers
  uint32_t b1 = AddTestRegister("b1", 8);
  uint32_t b2 = AddTestRegister("b2", 8);

  // Add a few sub-registers
  uint32_t s1 = AddTestRegister("s1", 4, {b1});
  uint32_t s2 = AddTestRegister("s2", 4, {b2});

  // Add a register with invalidate_regs
  uint32_t i1 = AddTestRegister("i1", 8, {}, {b1});

  // Add a register with indirect invalidate regs to be expanded
  // TODO: why is it done conditionally to value_regs?
  uint32_t i2 = AddTestRegister("i2", 4, {b2}, {i1});

  info.Finalize(lldb_private::ArchSpec());

  ASSERT_REG(b1, 0);
  ASSERT_REG(b2, 8);
  ASSERT_REG(s1, 0, {b1});
  ASSERT_REG(s2, 8, {b2});
  ASSERT_REG(i1, 16, {}, {b1});
  ASSERT_REG(i2, 8, {b2}, {b1, i1});
}

TEST_F(DynamicRegisterInfoTest, no_finalize_regs) {
  // Add regular registers
  uint32_t b1 = AddTestRegister("b1", 8);
  uint32_t b2 = AddTestRegister("b2", 8);

  // Add a few sub-registers
  uint32_t s1 = AddTestRegister("s1", 4, {b1});
  uint32_t s2 = AddTestRegister("s2", 4, {b2});

  // Add a register with invalidate_regs
  uint32_t i1 = AddTestRegister("i1", 8, {}, {b1});

  // Add a register with indirect invalidate regs to be expanded
  // TODO: why is it done conditionally to value_regs?
  uint32_t i2 = AddTestRegister("i2", 4, {b2}, {i1});

  ASSERT_REG(b1, LLDB_INVALID_INDEX32);
  ASSERT_REG(b2, LLDB_INVALID_INDEX32);
  ASSERT_REG(s1, LLDB_INVALID_INDEX32);
  ASSERT_REG(s2, LLDB_INVALID_INDEX32);
  ASSERT_REG(i1, LLDB_INVALID_INDEX32);
  ASSERT_REG(i2, LLDB_INVALID_INDEX32);
}

class DynamicRegisterInfoRegisterTest : public ::testing::Test {
protected:
  std::vector<DynamicRegisterInfo::Register> m_regs;

  uint32_t AddTestRegister(
      const char *name, const char *group, uint32_t byte_size,
      std::function<void(const DynamicRegisterInfo::Register &)> adder,
      std::vector<uint32_t> value_regs = {},
      std::vector<uint32_t> invalidate_regs = {}) {
    DynamicRegisterInfo::Register new_reg{
        ConstString(name),     ConstString(),
        ConstString(group),    byte_size,
        LLDB_INVALID_INDEX32,  lldb::eEncodingUint,
        lldb::eFormatUnsigned, LLDB_INVALID_REGNUM,
        LLDB_INVALID_REGNUM,   LLDB_INVALID_REGNUM,
        LLDB_INVALID_REGNUM,   value_regs,
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
};

#define EXPECT_IN_REGS(reg, ...)                                               \
  {                                                                            \
    SCOPED_TRACE("at register " #reg);                                         \
    ExpectInRegs(reg, #reg, __VA_ARGS__);                                      \
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
  uint32_t al = AddTestRegister("al", "supplementary", 1, suppl_adder, {rax});

  EXPECT_IN_REGS(rax, {}, {eax, ax, al});
  EXPECT_IN_REGS(eax, {rax}, {rax, ax, al});
  EXPECT_IN_REGS(ax, {rax}, {rax, eax, al});
  EXPECT_IN_REGS(al, {rax}, {rax, eax, ax});
}
