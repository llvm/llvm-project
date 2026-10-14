//===-- GetPointeeDataTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/ScriptInterpreter/None/ScriptInterpreterNone.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/Symbol/ClangTestUtils.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/Listener.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::clang_utils;

namespace {
/// A process whose live-memory reads return a fixed sentinel byte, so a test
/// can tell whether data came from the live process or from the read-only
/// object file. When \c fail_reads is set the live read fails instead, which
/// models an address that has no backing memory in a core file.
struct SentinelProcess : Process {
  static constexpr uint8_t kSentinel = 0xEE;
  bool fail_reads = false;

  SentinelProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  llvm::StringRef GetPluginName() override { return "sentinel"; }
  bool CanDebug(TargetSP, bool) override { return true; }
  bool IsAlive() override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  size_t DoReadMemory(const ProcessAddress &, void *buf, size_t size,
                      Status &error) override {
    if (fail_reads) {
      error = Status::FromErrorString("no live memory at this address");
      return 0;
    }
    std::memset(buf, kSentinel, size);
    return size;
  }
};

class GetPointeeDataTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF,
                platform_linux::PlatformLinux, SymbolFileSymtab,
                ScriptInterpreterNone>
      m_subsystems;

protected:
  /// Build a target with a read-only .rodata section holding four little-endian
  /// int32 values (10, 20, 30, 40) loaded at an address, attach a process (that
  /// either serves sentinel bytes or fails live reads), and read the four ints
  /// (item_count > 1) through ValueObject::GetPointeeData. The bytes read are
  /// returned in \p out_bytes.
  void ReadFourInts(bool fail_live_reads, size_t &out_bytes_read,
                    std::vector<uint8_t> &out_bytes) {
    ArchSpec arch("x86_64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));

    DebuggerSP debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(debugger_sp);

    PlatformSP platform_sp;
    TargetSP target_sp;
    debugger_sp->GetTargetList().CreateTarget(
        *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
    ASSERT_TRUE(target_sp);

    // A read-only (SHF_ALLOC, no SHF_WRITE) .rodata section holding four
    // little-endian int32 values: 10, 20, 30, 40.
    auto expected_file = TestFile::fromYaml(R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_DYN
  Machine: EM_X86_64
Sections:
  - Name:    .rodata
    Type:    SHT_PROGBITS
    Flags:   [ SHF_ALLOC ]
    Address: 0x1000
    Content: '0a000000140000001e00000028000000'
...
)");
    ASSERT_THAT_EXPECTED(expected_file, llvm::Succeeded());

    auto module_sp = std::make_shared<Module>(expected_file->moduleSpec());
    ASSERT_TRUE(module_sp);

    SectionList *section_list = module_sp->GetSectionList();
    ASSERT_TRUE(section_list);
    SectionSP rodata_sp =
        section_list->FindSectionByName(ConstString(".rodata"));
    ASSERT_TRUE(rodata_sp);

    // Load the read-only section so its bytes are reachable via a load address.
    const addr_t load_base = 0x4000;
    ASSERT_TRUE(target_sp->SetSectionLoadAddress(rodata_sp, load_base));

    // Attach a live process whose memory differs from the file contents (or
    // fails), so the source of the read is unambiguous.
    ListenerSP listener_sp = Listener::MakeListener("dummy");
    auto sentinel = std::make_shared<SentinelProcess>(target_sp, listener_sp);
    ASSERT_TRUE(sentinel);
    sentinel->fail_reads = fail_live_reads;
    ProcessSP process_sp = sentinel;
    struct TargetHack : Target {
      void SetProcess(ProcessSP process) { m_process_sp = std::move(process); }
    };
    static_cast<TargetHack *>(target_sp.get())->SetProcess(process_sp);

    // Build an `int *` whose value is the load address of the read-only array.
    TypeSystemClangHolder holder("test");
    TypeSystemClang *ast = holder.GetAST();
    ASSERT_TRUE(ast);
    CompilerType int_ptr_type =
        ast->GetBasicType(lldb::BasicType::eBasicTypeInt).GetPointerType();
    ASSERT_TRUE(int_ptr_type.IsPointerType());

    addr_t ptr_value = load_base;
    DataExtractor ptr_data(&ptr_value, sizeof(ptr_value),
                           endian::InlHostByteOrder(), sizeof(void *));
    ExecutionContext exe_ctx(process_sp);
    ValueObjectSP ptr_sp = ValueObjectConstResult::Create(
        exe_ctx.GetBestExecutionContextScope(), int_ptr_type, ConstString("p"),
        ptr_data);
    ASSERT_TRUE(ptr_sp);

    // Read four ints (item_count > 1) through GetPointeeData.
    DataExtractor result;
    out_bytes_read = ptr_sp->GetPointeeData(result, 0, /*item_count=*/4);
    out_bytes.assign(result.GetDataStart(), result.GetDataEnd());
  }
};
} // namespace

// GetPointeeData reads multi-element pointee data (item_count > 1) through the
// eAddressTypeLoad path. It must prefer live process memory so that any
// modifications the process made to otherwise read-only memory are observed.
TEST_F(GetPointeeDataTest, MultiElementReadPrefersLiveMemory) {
  size_t bytes_read = 0;
  std::vector<uint8_t> bytes;
  ReadFourInts(/*fail_live_reads=*/false, bytes_read, bytes);

  ASSERT_EQ(bytes_read, 4u * sizeof(uint32_t));
  // The bytes must come from the live process (sentinel), NOT the read-only
  // section, i.e. the read must use force_live_memory=true first.
  EXPECT_EQ(bytes, std::vector<uint8_t>(4 * sizeof(uint32_t),
                                        SentinelProcess::kSentinel));
}

// When the live read fails (e.g. read-only data that is not backed by a core
// file), GetPointeeData must fall back to reading the object file's read-only
// section rather than returning nothing.
TEST_F(GetPointeeDataTest, MultiElementReadFallsBackToReadOnlySection) {
  size_t bytes_read = 0;
  std::vector<uint8_t> bytes;
  ReadFourInts(/*fail_live_reads=*/true, bytes_read, bytes);

  ASSERT_EQ(bytes_read, 4u * sizeof(uint32_t));
  DataExtractor result(bytes.data(), bytes.size(), endian::InlHostByteOrder(),
                       /*addr_size=*/8);
  lldb::offset_t offset = 0;
  for (uint32_t want : {10u, 20u, 30u, 40u})
    EXPECT_EQ(result.GetU32(&offset), want);
}
