//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Register/Arch/lldb-arm32-register-enums.h"
#include "lldb/Register/Arch/lldb-riscv32-register-enums.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_arm32.h"
#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_arm64.h"
#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_riscv32.h"
#include "lldb/Register/Layouts/Darwin/RegisterLayoutDarwin_x86_64.h"
#include "llvm/Support/Endian.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>
#include <cstddef>

using namespace lldb_private;
using llvm::Succeeded;

TEST(RegisterLayoutDarwinTest, Arm32FloatingPointSlotsStartAtS0) {
  using Layout = RegisterLayoutDarwin_arm32;
  const auto &layout = Layout::GetShared();
  auto description = layout.GetDescription();
  const auto s0 = description->Find("s0")->id;
  const auto s31 = description->Find("s31")->id;
  std::array<uint8_t, sizeof(Layout::FPU)> bytes{};
  llvm::support::endian::write32le(bytes.data(), 0x3f800000);
  llvm::support::endian::write32le(bytes.data() + 31 * sizeof(uint32_t),
                                   0x40000000);

  auto first = layout.Read(s0, {{Layout::FPURegSet, bytes}});
  ASSERT_THAT_EXPECTED(first, Succeeded());
  EXPECT_EQ(first->GetAsFloat(), 1.0f);
  auto last = layout.Read(s31, {{Layout::FPURegSet, bytes}});
  ASSERT_THAT_EXPECTED(last, Succeeded());
  EXPECT_EQ(last->GetAsFloat(), 2.0f);

  ASSERT_THAT_ERROR(
      layout.Write(s0, RegisterValue(3.0f), {{Layout::FPURegSet, bytes}}),
      Succeeded());
  EXPECT_EQ(llvm::support::endian::read32le(bytes.data()), 0x40400000u);
  EXPECT_EQ(
      llvm::support::endian::read32le(bytes.data() + 31 * sizeof(uint32_t)),
      0x40000000u);

  const auto *exception = description->Find("exception");
  ASSERT_NE(exception, nullptr);
  EXPECT_EQ(exception->id, registers::arm32::id::exc_exception);
  EXPECT_EQ(exception, description->FindByNumber(lldb::eRegisterKindLLDB, 50));
}

TEST(RegisterLayoutDarwinTest, Arm64AliasesUseTheSameSavedVectorSlot) {
  using Layout = RegisterLayoutDarwin_arm64;
  const auto &layout = Layout::GetShared();
  auto description = layout.GetDescription();
  for (const char *name : {"v31", "s31", "d31"}) {
    const auto *reg = description->Find(name);
    ASSERT_NE(reg, nullptr);
    const auto *location = layout.Find(reg->id);
    ASSERT_NE(location, nullptr);
    ASSERT_EQ(location->pieces.size(), 1u);
    EXPECT_EQ(location->pieces.front().bank, Layout::FPURegSet);
    EXPECT_EQ(location->pieces.front().byte_offset, 31 * sizeof(Layout::VReg));
    EXPECT_EQ(location->pieces.front().byte_size, reg->byte_size);
  }
  EXPECT_EQ(layout.GetBankSize(Layout::DBGRegSet), sizeof(Layout::DBG));
}

TEST(RegisterLayoutDarwinTest, Riscv32UsesTheMachProgramCounterSlot) {
  using Layout = RegisterLayoutDarwin_riscv32;
  const auto &layout = Layout::GetShared();
  auto description = layout.GetDescription();
  std::array<uint8_t, sizeof(Layout::GPR)> bytes{};
  llvm::support::endian::write32le(bytes.data() + offsetof(Layout::GPR, pc),
                                   0x12345678);
  const auto *pc_reg = description->Find("pc");
  ASSERT_NE(pc_reg, nullptr);
  EXPECT_EQ(pc_reg->id, registers::riscv32::id::gpr_pc);
  EXPECT_EQ(pc_reg, description->FindByNumber(lldb::eRegisterKindLLDB, 32));
  auto pc = layout.Read(pc_reg->id, {{Layout::GPRRegSet, bytes}});
  ASSERT_THAT_EXPECTED(pc, Succeeded());
  EXPECT_EQ(pc->GetAsUInt32(), 0x12345678u);
}

TEST(RegisterLayoutDarwinTest, X86KeepsTheMachX87TagWidthAndXmmOffsets) {
  using Layout = RegisterLayoutDarwin_x86_64;
  const auto &layout = Layout::GetShared();
  auto description = layout.GetDescription();
  std::array<uint8_t, sizeof(Layout::FPU)> bytes{};
  bytes[offsetof(Layout::FPU, ftw)] = 0xa5;
  bytes[offsetof(Layout::FPU, pad1)] = 0x5a;
  const auto *tag_reg = description->Find("ftw");
  ASSERT_NE(tag_reg, nullptr);
  EXPECT_EQ(tag_reg->id, registers::x86_64::id::ftag);
  EXPECT_EQ(tag_reg, description->FindByNumber(lldb::eRegisterKindLLDB, 23));
  auto tag = layout.Read(tag_reg->id, {{Layout::FPURegSet, bytes}});
  ASSERT_THAT_EXPECTED(tag, Succeeded());
  EXPECT_EQ(tag->GetAsUInt8(), 0xa5u);

  const auto xmm15 = description->Find("xmm15")->id;
  auto *location = layout.Find(xmm15);
  ASSERT_NE(location, nullptr);
  ASSERT_EQ(location->pieces.size(), 1u);
  EXPECT_EQ(location->pieces.front().byte_offset,
            offsetof(Layout::FPU, xmm) + 15 * sizeof(Layout::XMMReg));
}
