//===-- RegisterTypeBuilderClangTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/RegisterTypeBuilder/RegisterTypeBuilderClang.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/RegisterInfo.h"
#include "lldb/Utility/RegisterTypeFlags.h"
#include "gtest/gtest.h"

#include <mutex>
#include <optional>

using namespace lldb;
using namespace lldb_private;

namespace {

class RegisterTypeBuilderClangTest : public testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, TypeSystemClang,
                platform_linux::PlatformLinux>
      subsystems;

protected:
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    ArchSpec host_arch("x86_64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &host_arch));
    m_debugger_sp = Debugger::CreateInstance();
  }

  static RegisterInfo MakeRegisterInfo(const RegisterType &type,
                                       uint32_t byte_size) {
    RegisterInfo info{};
    info.name = "test";
    info.byte_size = byte_size;
    info.register_type = &type;
    return info;
  }

  DebuggerSP m_debugger_sp;
};

TEST_F(RegisterTypeBuilderClangTest, ReusesCachedType) {
  Target &target = m_debugger_sp->GetDummyTarget();
  RegisterTypeFlags flags("flags", 4,
                          {RegisterTypeFlags::Field("field", 0, 31)});
  RegisterInfo info = MakeRegisterInfo(flags, 4);
  RegisterTypeBuilderClang builder(target);

  CompilerType first = builder.GetRegisterType(info);
  CompilerType second = builder.GetRegisterType(info);

  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_EQ(first, second);
}

// XML type IDs are scoped to a feature, so separate features may define
// different types using the same ID.
TEST_F(RegisterTypeBuilderClangTest, DistinguishesTypesWithTheSameID) {
  Target &target = m_debugger_sp->GetDummyTarget();
  RegisterTypeFlags first_flags("flags", 4,
                                {RegisterTypeFlags::Field("first", 0, 31)});
  RegisterTypeFlags second_flags("flags", 4,
                                 {RegisterTypeFlags::Field("second", 0, 31)});
  RegisterTypeBuilderClang builder(target);

  CompilerType first =
      builder.GetRegisterType(MakeRegisterInfo(first_flags, 4));
  CompilerType second =
      builder.GetRegisterType(MakeRegisterInfo(second_flags, 4));

  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  EXPECT_NE(first, second);

  std::string first_field_name;
  std::string second_field_name;
  ASSERT_TRUE(
      first.GetFieldAtIndex(0, first_field_name, nullptr, nullptr, nullptr));
  ASSERT_TRUE(
      second.GetFieldAtIndex(0, second_field_name, nullptr, nullptr, nullptr));
  EXPECT_EQ(first_field_name, "first");
  EXPECT_EQ(second_field_name, "second");
}

TEST_F(RegisterTypeBuilderClangTest, RegisterSizeIsPartOfCacheKey) {
  Target &target = m_debugger_sp->GetDummyTarget();
  RegisterTypeEnum type("enum", {{0, "zero"}, {1, "one"}});
  RegisterTypeBuilderClang builder(target);

  CompilerType four_byte = builder.GetRegisterType(MakeRegisterInfo(type, 4));
  CompilerType eight_byte = builder.GetRegisterType(MakeRegisterInfo(type, 8));

  ASSERT_TRUE(four_byte);
  ASSERT_TRUE(eight_byte);
  EXPECT_NE(four_byte, eight_byte);
  EXPECT_EQ(llvm::expectedToOptional(four_byte.GetByteSize(nullptr)), 4u);
  EXPECT_EQ(llvm::expectedToOptional(eight_byte.GetByteSize(nullptr)), 8u);
  EXPECT_EQ(four_byte, builder.GetRegisterType(MakeRegisterInfo(type, 4)));
  EXPECT_EQ(eight_byte, builder.GetRegisterType(MakeRegisterInfo(type, 8)));
}

TEST_F(RegisterTypeBuilderClangTest, DistinguishesReusedObjectAddresses) {
  Target &target = m_debugger_sp->GetDummyTarget();
  RegisterTypeBuilderClang builder(target);
  std::optional<RegisterTypeEnum> type;

  type.emplace("enum", RegisterTypeEnum::Enumerators{{0, "first"}});
  const RegisterTypeEnum *first_address = &*type;
  uint64_t first_uid = type->GetUID();
  CompilerType first = builder.GetRegisterType(MakeRegisterInfo(*type, 4));
  ASSERT_TRUE(first);

  type.reset();
  type.emplace("enum", RegisterTypeEnum::Enumerators{{0, "second"}});
  ASSERT_EQ(first_address, &*type);
  ASSERT_NE(first_uid, type->GetUID());
  CompilerType second = builder.GetRegisterType(MakeRegisterInfo(*type, 4));

  ASSERT_TRUE(second);
  EXPECT_NE(first, second);
}

TEST_F(RegisterTypeBuilderClangTest, CacheFollowsScratchTypeSystem) {
  Target &target = m_debugger_sp->GetDummyTarget();
  RegisterTypeEnum type("enum", {{0, "zero"}, {1, "one"}});
  RegisterInfo info = MakeRegisterInfo(type, 4);
  RegisterTypeBuilderClang builder(target);

  CompilerType first = builder.GetRegisterType(info);
  ASSERT_TRUE(first);
  std::shared_ptr<TypeSystemClang> first_type_system =
      first.GetTypeSystem<TypeSystemClang>();
  ASSERT_TRUE(first_type_system);

  target.ClearModules(/*delete_locations=*/false);

  CompilerType second = builder.GetRegisterType(info);
  ASSERT_TRUE(second);
  std::shared_ptr<TypeSystemClang> second_type_system =
      second.GetTypeSystem<TypeSystemClang>();
  ASSERT_TRUE(second_type_system);
  EXPECT_NE(first_type_system, second_type_system);
  EXPECT_NE(first, second);
}

} // namespace
