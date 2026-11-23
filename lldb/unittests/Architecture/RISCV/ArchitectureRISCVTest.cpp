//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Architecture/RISCV/ArchitectureRISCV.h"
#include "lldb/Core/Architecture.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/ArchSpec.h"

#include "gtest/gtest.h"

using namespace lldb_private;

class ArchitectureRISCVTest : public testing::Test {
protected:
  static void SetUpTestSuite() { ArchitectureRISCV::Initialize(); }
  static void TearDownTestSuite() { ArchitectureRISCV::Terminate(); }
};

TEST_F(ArchitectureRISCVTest, CreatesPluginForRISCVTargets) {
  EXPECT_TRUE(PluginManager::CreateArchitectureInstance(
      ArchSpec("riscv32-unknown-unknown-elf")));
  EXPECT_TRUE(PluginManager::CreateArchitectureInstance(
      ArchSpec("riscv64-unknown-unknown-elf")));
  EXPECT_FALSE(PluginManager::CreateArchitectureInstance(
      ArchSpec("x86_64-unknown-unknown-elf")));
}

TEST_F(ArchitectureRISCVTest, ValidatesEBreak) {
  std::unique_ptr<Architecture> arch =
      PluginManager::CreateArchitectureInstance(
          ArchSpec("riscv32-unknown-unknown-elf"));
  ASSERT_TRUE(arch);

  const uint8_t ebreak[] = {0x73, 0x00, 0x10, 0x00};
  const uint8_t ebreak_with_extra_bytes[] = {0x73, 0x00, 0x10,
                                             0x00, 0xff, 0xff};
  const uint8_t wrong_immediate[] = {0x73, 0x00, 0x20, 0x00};
  const uint8_t truncated_ebreak[] = {0x73, 0x00};
  const llvm::ArrayRef<uint8_t> empty_reference;
  const uint8_t bad_size_reference[] = {0x73, 0x00, 0x10};

  EXPECT_TRUE(arch->IsValidTrapInstruction(ebreak, ebreak));
  EXPECT_TRUE(arch->IsValidTrapInstruction(ebreak, ebreak_with_extra_bytes));
  EXPECT_FALSE(arch->IsValidTrapInstruction(ebreak, wrong_immediate));
  EXPECT_FALSE(arch->IsValidTrapInstruction(ebreak, truncated_ebreak));
  EXPECT_FALSE(arch->IsValidTrapInstruction(empty_reference, ebreak));
  EXPECT_FALSE(arch->IsValidTrapInstruction(bad_size_reference, ebreak));
}

TEST_F(ArchitectureRISCVTest, ValidatesCompressedEBreak) {
  std::unique_ptr<Architecture> arch =
      PluginManager::CreateArchitectureInstance(
          ArchSpec("riscv64-unknown-unknown-elf"));
  ASSERT_TRUE(arch);

  const uint8_t compressed_ebreak[] = {0x02, 0x90};
  const uint8_t compressed_ebreak_with_extra_bytes[] = {0x02, 0x90, 0xff, 0xff};
  const uint8_t compressed_unimp[] = {0x00, 0x00};
  const uint8_t truncated_compressed_ebreak[] = {0x02};
  const llvm::ArrayRef<uint8_t> empty_reference;
  const uint8_t bad_size_reference[] = {0x02, 0x90, 0xff};

  EXPECT_TRUE(
      arch->IsValidTrapInstruction(compressed_ebreak, compressed_ebreak));
  EXPECT_TRUE(arch->IsValidTrapInstruction(compressed_ebreak,
                                           compressed_ebreak_with_extra_bytes));
  EXPECT_FALSE(
      arch->IsValidTrapInstruction(empty_reference, compressed_ebreak));
  EXPECT_FALSE(
      arch->IsValidTrapInstruction(bad_size_reference, compressed_ebreak));
  EXPECT_FALSE(
      arch->IsValidTrapInstruction(compressed_ebreak, compressed_unimp));
  EXPECT_FALSE(arch->IsValidTrapInstruction(compressed_ebreak,
                                            truncated_compressed_ebreak));
}
