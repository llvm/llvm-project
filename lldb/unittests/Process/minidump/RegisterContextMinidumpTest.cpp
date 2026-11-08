//===-- RegisterContextMinidumpTest.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/minidump/RegisterContextMinidump_x86_32.h"
#include "lldb/Register/Arch/lldb-x86_64-register-enums.h"
#include "lldb/Register/Arch/lldb-x86-register-enums.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_x86_64.h"
#include "Plugins/Process/minidump/RegisterContextMinidump_ARM.h"
#include "lldb/Utility/DataBuffer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::minidump;

static uint64_t reg64(const ArchSpec &arch, const RegisterLayout &model,
                      llvm::ArrayRef<uint8_t> bytes, RegisterID id) {
  const auto *reg = model.GetDescription()->Find(id);
  EXPECT_NE(reg, nullptr);
  if (!reg)
    return 0;
  auto value = ReadMinidumpX86Register(arch, model, bytes, reg->id);
  EXPECT_TRUE(bool(value));
  if (!value) {
    llvm::consumeError(value.takeError());
    return 0;
  }
  return value->GetAsUInt64();
}
static uint32_t reg32(const ArchSpec &arch, const RegisterLayout &model,
                      llvm::ArrayRef<uint8_t> bytes, uint32_t number) {
  return reg64(arch, model, bytes, number);
}

TEST(RegisterContextMinidump, ReadMinidumpContext_x86_32) {
  MinidumpContext_x86_32 Context;
  Context.context_flags =
      static_cast<uint32_t>(MinidumpContext_x86_32_Flags::x86_32_Flag |
                            MinidumpContext_x86_32_Flags::Control |
                            MinidumpContext_x86_32_Flags::Segments |
                            MinidumpContext_x86_32_Flags::Integer);
  Context.eax = 0x00010203;
  Context.ebx = 0x04050607;
  Context.ecx = 0x08090a0b;
  Context.edx = 0x0c0d0e0f;
  Context.edi = 0x10111213;
  Context.esi = 0x14151617;
  Context.ebp = 0x18191a1b;
  Context.esp = 0x1c1d1e1f;
  Context.eip = 0x20212223;
  Context.eflags = 0x24252627;
  Context.cs = 0x2829;
  Context.fs = 0x2a2b;
  Context.gs = 0x2c2d;
  Context.ss = 0x2e2f;
  Context.ds = 0x3031;
  Context.es = 0x3233;
  llvm::ArrayRef<uint8_t> ContextRef(reinterpret_cast<uint8_t *>(&Context),
                                     sizeof(Context));

  ArchSpec arch("i386-pc-linux");
  auto result = CreateMinidumpX86RegisterLayout(arch);
  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  auto model = std::move(*result);

  EXPECT_EQ(Context.eax,
            reg32(arch, model, ContextRef, registers::x86::id::eax));
  EXPECT_EQ(Context.ebx,
            reg32(arch, model, ContextRef, registers::x86::id::ebx));
  EXPECT_EQ(Context.ecx,
            reg32(arch, model, ContextRef, registers::x86::id::ecx));
  EXPECT_EQ(Context.edx,
            reg32(arch, model, ContextRef, registers::x86::id::edx));
  EXPECT_EQ(Context.edi,
            reg32(arch, model, ContextRef, registers::x86::id::edi));
  EXPECT_EQ(Context.esi,
            reg32(arch, model, ContextRef, registers::x86::id::esi));
  EXPECT_EQ(Context.ebp,
            reg32(arch, model, ContextRef, registers::x86::id::ebp));
  EXPECT_EQ(Context.esp,
            reg32(arch, model, ContextRef, registers::x86::id::esp));
  EXPECT_EQ(Context.eip,
            reg32(arch, model, ContextRef, registers::x86::id::eip));
  EXPECT_EQ(Context.eflags,
            reg32(arch, model, ContextRef, registers::x86::id::eflags));
  EXPECT_EQ(Context.cs, reg32(arch, model, ContextRef, registers::x86::id::cs));
  EXPECT_EQ(Context.fs, reg32(arch, model, ContextRef, registers::x86::id::fs));
  EXPECT_EQ(Context.gs, reg32(arch, model, ContextRef, registers::x86::id::gs));
  EXPECT_EQ(Context.ss, reg32(arch, model, ContextRef, registers::x86::id::ss));
  EXPECT_EQ(Context.ds, reg32(arch, model, ContextRef, registers::x86::id::ds));
  EXPECT_EQ(Context.es, reg32(arch, model, ContextRef, registers::x86::id::es));
}

TEST(RegisterContextMinidump, ReadMinidumpContext_x86_64) {
  MinidumpContext_x86_64 Context;
  Context.context_flags =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::x86_64_Flag |
                            MinidumpContext_x86_64_Flags::Control |
                            MinidumpContext_x86_64_Flags::Segments |
                            MinidumpContext_x86_64_Flags::Integer);
  Context.rax = 0x0001020304050607;
  Context.rbx = 0x08090a0b0c0d0e0f;
  Context.rcx = 0x1011121314151617;
  Context.rdx = 0x18191a1b1c1d1e1f;
  Context.rdi = 0x2021222324252627;
  Context.rsi = 0x28292a2b2c2d2e2f;
  Context.rbp = 0x3031323334353637;
  Context.rsp = 0x38393a3b3c3d3e3f;
  Context.r8 = 0x4041424344454647;
  Context.r9 = 0x48494a4b4c4d4e4f;
  Context.r10 = 0x5051525354555657;
  Context.r11 = 0x58595a5b5c5d5e5f;
  Context.r12 = 0x6061626364656667;
  Context.r13 = 0x68696a6b6c6d6e6f;
  Context.r14 = 0x7071727374757677;
  Context.r15 = 0x78797a7b7c7d7e7f;
  Context.rip = 0x8081828384858687;
  Context.eflags = 0x88898a8b;
  Context.cs = 0x8c8d;
  Context.fs = 0x8e8f;
  Context.gs = 0x9091;
  Context.ss = 0x9293;
  Context.ds = 0x9495;
  Context.es = 0x9697;
  llvm::ArrayRef<uint8_t> ContextRef(reinterpret_cast<uint8_t *>(&Context),
                                     sizeof(Context));

  ArchSpec arch("x86_64-pc-linux");
  auto result = CreateMinidumpX86RegisterLayout(arch);
  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  auto model = std::move(*result);

  EXPECT_EQ(Context.rax,
            reg64(arch, model, ContextRef, registers::x86_64::id::rax));
  EXPECT_EQ(Context.rbx,
            reg64(arch, model, ContextRef, registers::x86_64::id::rbx));
  EXPECT_EQ(Context.rcx,
            reg64(arch, model, ContextRef, registers::x86_64::id::rcx));
  EXPECT_EQ(Context.rdx,
            reg64(arch, model, ContextRef, registers::x86_64::id::rdx));
  EXPECT_EQ(Context.rdi,
            reg64(arch, model, ContextRef, registers::x86_64::id::rdi));
  EXPECT_EQ(Context.rsi,
            reg64(arch, model, ContextRef, registers::x86_64::id::rsi));
  EXPECT_EQ(Context.rbp,
            reg64(arch, model, ContextRef, registers::x86_64::id::rbp));
  EXPECT_EQ(Context.rsp,
            reg64(arch, model, ContextRef, registers::x86_64::id::rsp));
  EXPECT_EQ(Context.r8,
            reg64(arch, model, ContextRef, registers::x86_64::id::r8));
  EXPECT_EQ(Context.r9,
            reg64(arch, model, ContextRef, registers::x86_64::id::r9));
  EXPECT_EQ(Context.r10,
            reg64(arch, model, ContextRef, registers::x86_64::id::r10));
  EXPECT_EQ(Context.r11,
            reg64(arch, model, ContextRef, registers::x86_64::id::r11));
  EXPECT_EQ(Context.r12,
            reg64(arch, model, ContextRef, registers::x86_64::id::r12));
  EXPECT_EQ(Context.r13,
            reg64(arch, model, ContextRef, registers::x86_64::id::r13));
  EXPECT_EQ(Context.r14,
            reg64(arch, model, ContextRef, registers::x86_64::id::r14));
  EXPECT_EQ(Context.r15,
            reg64(arch, model, ContextRef, registers::x86_64::id::r15));
  EXPECT_EQ(Context.rip,
            reg64(arch, model, ContextRef, registers::x86_64::id::rip));
  EXPECT_EQ(Context.eflags,
            reg64(arch, model, ContextRef, registers::x86_64::id::rflags));
  EXPECT_EQ(Context.cs,
            reg64(arch, model, ContextRef, registers::x86_64::id::cs));
  EXPECT_EQ(Context.fs,
            reg64(arch, model, ContextRef, registers::x86_64::id::fs));
  EXPECT_EQ(Context.gs,
            reg64(arch, model, ContextRef, registers::x86_64::id::gs));
  EXPECT_EQ(Context.ss,
            reg64(arch, model, ContextRef, registers::x86_64::id::ss));
  EXPECT_EQ(Context.ds,
            reg64(arch, model, ContextRef, registers::x86_64::id::ds));
  EXPECT_EQ(Context.es,
            reg64(arch, model, ContextRef, registers::x86_64::id::es));
}

static void TestARMRegInfo(const lldb_private::RegisterDescriptor *info) {
  // Make sure we have valid register numbers for eRegisterKindEHFrame and
  // eRegisterKindDWARF for GPR registers r0-r15 so that we can unwind
  // correctly when using this information.
  llvm::StringRef name(info->name);
  llvm::StringRef alt_name(info->alt_name);
  if (name.starts_with("r") || alt_name.starts_with("r")) {
    EXPECT_NE(info->numbers[lldb::eRegisterKindEHFrame], LLDB_INVALID_REGNUM);
    EXPECT_NE(info->numbers[lldb::eRegisterKindDWARF], LLDB_INVALID_REGNUM);
  }
  // Verify generic register are set correctly
  if (name == "r0") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG1);
  } else if (name == "r1") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG2);
  } else if (name == "r2") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG3);
  } else if (name == "r3") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_ARG4);
  } else if (name == "sp") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_SP);
  } else if (name == "fp") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_FP);
  } else if (name == "lr") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_RA);
  } else if (name == "pc") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_PC);
  } else if (name == "cpsr") {
    EXPECT_EQ(info->numbers[lldb::eRegisterKindGeneric],
              (uint32_t)LLDB_REGNUM_GENERIC_FLAGS);
  }
}

TEST(RegisterContextMinidump, CheckRegisterContextMinidump_ARM) {
  auto apple = CreateMinidumpArmRegisterLayout(true).GetDescription();
  auto other = CreateMinidumpArmRegisterLayout(false).GetDescription();
  size_t num_regs = apple->GetRegisters().size();
  const lldb_private::RegisterDescriptor *reg_info;
  for (size_t reg=0; reg<num_regs; ++reg) {
    reg_info = apple->GetRegisterAtIndex(reg);
    TestARMRegInfo(reg_info);
    reg_info = other->GetRegisterAtIndex(reg);
    TestARMRegInfo(reg_info);
  }
}

TEST(RegisterContextMinidump, RejectsAbsentRegisterSetsAndTruncatedContexts) {
  ArchSpec arch("x86_64-pc-windows");
  auto result = CreateMinidumpX86RegisterLayout(arch);
  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  auto model = std::move(*result);
  MinidumpContext_x86_64 context{};
  context.context_flags =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::x86_64_Flag);
  llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(&context),
                                sizeof(context));
  const auto *rax = model.GetDescription()->FindByNumber(
      lldb::eRegisterKindLLDB, registers::x86_64::id::rax);
  auto absent = ReadMinidumpX86Register(arch, model, bytes, rax->id);
  EXPECT_FALSE(bool(absent));
  if (!absent)
    llvm::consumeError(absent.takeError());
  auto truncated =
      ReadMinidumpX86Register(arch, model, bytes.take_front(8), rax->id);
  EXPECT_FALSE(bool(truncated));
  if (!truncated)
    llvm::consumeError(truncated.takeError());
}

TEST(RegisterContextMinidump, ResolvesValuesByIDRatherThanExternalNumber) {
  ArchSpec arch("x86_64-pc-windows");
  MinidumpContext_x86_64 context{};
  context.context_flags =
      static_cast<uint32_t>(MinidumpContext_x86_64_Flags::x86_64_Flag |
                            MinidumpContext_x86_64_Flags::Integer);
  context.rax = 0x123456789abcdef0;
  RegisterDescriptor reg;
  reg.id = 101;
  reg.name = "rax";
  reg.byte_size = 8;
  reg.numbers[lldb::eRegisterKindLLDB] = 79;
  auto description = RegisterDescription::Create({reg});
  ASSERT_THAT_EXPECTED(description, llvm::Succeeded());
  RegisterLayout layout(lldb::eByteOrderLittle);
  ASSERT_THAT_ERROR(layout.AddBank(0, sizeof(context)), llvm::Succeeded());
  RegisterLocation location;
  location.kind = RegisterLocation::Stored;
  location.pieces = {{0, offsetof(MinidumpContext_x86_64, rax), 8, 0}};
  ASSERT_THAT_ERROR(layout.Add(101, std::move(location)), llvm::Succeeded());
  llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(&context),
                                sizeof(context));
  layout = llvm::cantFail(RegisterLayout::Create(
      std::make_shared<RegisterDescription>(*description), std::move(layout)));
  auto value = ReadMinidumpX86Register(arch, layout, bytes, 101);
  ASSERT_THAT_EXPECTED(value, llvm::Succeeded());
  EXPECT_EQ(value->GetAsUInt64(), context.rax);
  EXPECT_THAT_EXPECTED(ReadMinidumpX86Register(arch, layout, bytes, 79),
                       llvm::Failed());
}

TEST(RegisterContextMinidump, ARMSourceBytesAreDecodedWithoutHostStructs) {
  auto layout = CreateMinidumpArmRegisterLayout(false);
  const auto *pc = layout.GetDescription()->Find("pc");
  ASSERT_NE(pc, nullptr);
  const auto *location = layout.Find(pc->id);
  ASSERT_NE(location, nullptr);
  std::vector<uint8_t> bytes(layout.GetBankSize(0), 0);
  size_t offset = location->pieces.front().byte_offset;
  bytes[offset] = 0x78;
  bytes[offset + 1] = 0x56;
  bytes[offset + 2] = 0x34;
  bytes[offset + 3] = 0x12;
  auto value = layout.Read(pc->id, {{0, bytes}});
  ASSERT_TRUE(bool(value));
  EXPECT_EQ(value->GetAsUInt32(), 0x12345678u);
}

TEST(RegisterContextMinidump,
     RejectsUnsupportedOperatingSystemsAndArchitectures) {
  for (const char *triple : {"x86_64-unknown-unknown", "i386-unknown-freebsd",
                             "aarch64-pc-windows"}) {
    SCOPED_TRACE(triple);
    EXPECT_THAT_EXPECTED(CreateMinidumpX86RegisterLayout(ArchSpec(triple)),
                         llvm::Failed());
  }
}
