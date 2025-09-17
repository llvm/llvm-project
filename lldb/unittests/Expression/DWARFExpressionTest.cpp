//===-- DWARFExpressionTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"
#include "Plugins/ObjectFile/wasm/ObjectFileWasm.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDwo.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileWasm.h"
#include "Plugins/SymbolVendor/wasm/SymbolVendorWasm.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private::plugin::dwarf;
using namespace lldb_private::wasm;
using namespace lldb_private;
using namespace llvm::dwarf;

namespace {
struct MockProcess : Process {
  MockProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}

  llvm::StringRef GetPluginName() override { return "mock process"; }

  bool CanDebug(lldb::TargetSP target, bool plugin_specified_by_name) override {
    return false;
  };

  Status DoDestroy() override { return {}; }

  void RefreshStateAfterStop() override {}

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  };

  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    for (size_t i = 0; i < size; ++i)
      ((char *)buf)[i] = (vm_addr + i) & 0xff;
    error.Clear();
    return size;
  }
};

class MockThread : public Thread {
public:
  MockThread(Process &process) : Thread(process, /*tid=*/1), m_reg_ctx_sp() {}
  ~MockThread() override { DestroyThread(); }

  void RefreshStateAfterStop() override {}

  lldb::RegisterContextSP GetRegisterContext() override { return m_reg_ctx_sp; }

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(StackFrame *frame) override {
    return m_reg_ctx_sp;
  }

  bool CalculateStopInfo() override { return false; }

  void SetRegisterContext(lldb::RegisterContextSP reg_ctx_sp) {
    m_reg_ctx_sp = reg_ctx_sp;
  }

private:
  lldb::RegisterContextSP m_reg_ctx_sp;
};

class MockRegisterContext : public RegisterContext {
public:
  MockRegisterContext(Thread &thread, const RegisterValue &reg_value)
      : RegisterContext(thread, 0 /*concrete_frame_idx*/),
        m_reg_value(reg_value) {}

  void InvalidateAllRegisters() override {}

  size_t GetRegisterCount() override { return 0; }

  const RegisterInfo *GetRegisterInfoAtIndex(size_t reg) override {
    return &m_reg_info;
  }

  size_t GetRegisterSetCount() override { return 0; }

  const RegisterSet *GetRegisterSet(size_t reg_set) override { return nullptr; }

  lldb::ByteOrder GetByteOrder() override {
    return lldb::ByteOrder::eByteOrderLittle;
  }

  bool ReadRegister(const RegisterInfo *reg_info,
                    RegisterValue &reg_value) override {
    reg_value = m_reg_value;
    return true;
  }

  bool WriteRegister(const RegisterInfo *reg_info,
                     const RegisterValue &reg_value) override {
    return false;
  }

  uint32_t ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind,
                                               uint32_t num) override {
    return num;
  }

private:
  RegisterInfo m_reg_info{};
  RegisterValue m_reg_value{};
};
} // namespace

static llvm::Expected<Scalar> Evaluate(llvm::ArrayRef<uint8_t> expr,
                                       lldb::ModuleSP module_sp = {},
                                       DWARFUnit *unit = nullptr,
                                       ExecutionContext *exe_ctx = nullptr,
                                       RegisterContext *reg_ctx = nullptr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);

  llvm::Expected<Value> result = DWARFExpression::Evaluate(
      exe_ctx, reg_ctx, module_sp, extractor, unit, lldb::eRegisterKindLLDB,
      /*initial_value_ptr=*/nullptr,
      /*object_address_ptr=*/nullptr);
  if (!result)
    return result.takeError();

  switch (result->GetValueType()) {
  case Value::ValueType::Scalar:
    return result->GetScalar();
  case Value::ValueType::LoadAddress:
    return LLDB_INVALID_ADDRESS;
  case Value::ValueType::HostAddress: {
    // Convert small buffers to scalars to simplify the tests.
    DataBufferHeap &buf = result->GetBuffer();
    if (buf.GetByteSize() <= 8) {
      uint64_t val = 0;
      memcpy(&val, buf.GetBytes(), buf.GetByteSize());
      return Scalar(llvm::APInt(buf.GetByteSize() * 8, val, false));
    }
  }
    [[fallthrough]];
  default:
    break;
  }
  return llvm::createStringError("unsupported value type");
}

class DWARFExpressionTester : public YAMLModuleTester {
public:
  DWARFExpressionTester(llvm::StringRef yaml_data, size_t cu_index)
      : YAMLModuleTester(yaml_data, cu_index) {}

  using YAMLModuleTester::YAMLModuleTester;
  llvm::Expected<Scalar> Eval(llvm::ArrayRef<uint8_t> expr) {
    return ::Evaluate(expr, m_module_sp, m_dwarf_unit);
  }
};

/// Unfortunately Scalar's operator==() is really picky.
static Scalar GetScalar(unsigned bits, uint64_t value, bool sign) {
  Scalar scalar(value);
  scalar.TruncOrExtendTo(bits, sign);
  return scalar;
}

/// This is needed for the tests that use a mock process.
class DWARFExpressionMockProcessTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
  }
  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

// NB: This class doesn't use the override keyword to avoid
// -Winconsistent-missing-override warnings from the compiler. The
// inconsistency comes from the overriding definitions in the MOCK_*** macros.
class MockTarget : public Target {
public:
  MockTarget(Debugger &debugger, const ArchSpec &target_arch,
             const lldb::PlatformSP &platform_sp)
      : Target(debugger, target_arch, platform_sp, true) {}

  MOCK_METHOD2(ReadMemory,
               llvm::Expected<std::vector<uint8_t>>(lldb::addr_t addr,
                                                    size_t size));

  size_t ReadMemory(const Address &addr, void *dst, size_t dst_len,
                    Status &error, bool force_live_memory = false,
                    lldb::addr_t *load_addr_ptr = nullptr,
                    bool *did_read_live_memory = nullptr) /*override*/ {
    auto expected_memory = this->ReadMemory(addr.GetOffset(), dst_len);
    if (!expected_memory) {
      llvm::consumeError(expected_memory.takeError());
      return 0;
    }
    const size_t bytes_read = expected_memory->size();
    assert(bytes_read <= dst_len);
    std::memcpy(dst, expected_memory->data(), bytes_read);
    return bytes_read;
  }
};

TEST(DWARFExpression, DW_OP_pick) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 0}),
                       llvm::HasValue(0));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 1}),
                       llvm::HasValue(1));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit1, DW_OP_lit0, DW_OP_pick, 2}),
                       llvm::Failed());
}

TEST(DWARFExpression, DW_OP_const) {
  // Extend to address size.
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const1u, 0x88}), llvm::HasValue(0x88));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const1s, 0x88}),
                       llvm::HasValue(0xffffff88));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const2u, 0x47, 0x88}),
                       llvm::HasValue(0x8847));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const2s, 0x47, 0x88}),
                       llvm::HasValue(0xffff8847));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const4u, 0x44, 0x42, 0x47, 0x88}),
                       llvm::HasValue(0x88474244));
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const4s, 0x44, 0x42, 0x47, 0x88}),
                       llvm::HasValue(0x88474244));

  // Truncate to address size.
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_const8u, 0x00, 0x11, 0x22, 0x33, 0x44, 0x42, 0x47, 0x88}),
      llvm::HasValue(0x33221100));
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_const8s, 0x00, 0x11, 0x22, 0x33, 0x44, 0x42, 0x47, 0x88}),
      llvm::HasValue(0x33221100));

  // Don't truncate to address size for compatibility with clang (pr48087).
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_constu, 0x81, 0x82, 0x84, 0x88, 0x90, 0xa0, 0x40}),
      llvm::HasValue(0x01010101010101));
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_consts, 0x81, 0x82, 0x84, 0x88, 0x90, 0xa0, 0x40}),
      llvm::HasValue(0xffff010101010101));
}

TEST(DWARFExpression, DW_OP_skip) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const1u, 0x42, DW_OP_skip, 0x02, 0x00,
                                 DW_OP_const1u, 0xff}),
                       llvm::HasValue(0x42));
}

TEST(DWARFExpression, DW_OP_bra) {
  EXPECT_THAT_EXPECTED(
      // clang-format off
      Evaluate({
        DW_OP_const1u, 0x42,     // push 0x42
        DW_OP_const1u, 0x1,      // push 0x1
        DW_OP_bra, 0x02, 0x00,   // if 0x1 > 0, then skip 0x0002 opcodes
        DW_OP_const1u, 0xff,     // push 0xff
      }),
      // clang-format on
      llvm::HasValue(0x42));

  EXPECT_THAT_ERROR(Evaluate({DW_OP_bra, 0x01, 0x00}).takeError(),
                    llvm::Failed());
}

TEST(DWARFExpression, DW_OP_convert) {
  /// Auxiliary debug info.
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_yes
          Attributes:
            - Attribute:       DW_AT_language
              Form:            DW_FORM_data2
        - Code:            0x00000002
          Tag:             DW_TAG_base_type
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_encoding
              Form:            DW_FORM_data1
            - Attribute:       DW_AT_byte_size
              Form:            DW_FORM_data1
  debug_info:
    - Version:         4
      AddrSize:        8
      AbbrevTableID:   0
      AbbrOffset:      0x0
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
        - AbbrCode:        0x00000000
    - Version:         4
      AddrSize:        8
      AbbrevTableID:   0
      AbbrOffset:      0x0
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x000000000000000C
        # 0x0000000e:
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000004
        # 0x00000011:
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000007 # DW_ATE_unsigned
            - Value:           0x0000000000000008
        # 0x00000014:
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000005 # DW_ATE_signed
            - Value:           0x0000000000000008
        # 0x00000017:
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000008 # DW_ATE_unsigned_char
            - Value:           0x0000000000000001
        # 0x0000001a:
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x0000000000000006 # DW_ATE_signed_char
            - Value:           0x0000000000000001
        # 0x0000001d:
        - AbbrCode:        0x00000002
          Values:
            - Value:           0x000000000000000b # DW_ATE_numeric_string
            - Value:           0x0000000000000001
        - AbbrCode:        0x00000000

)";
  // Compile unit relative offsets to each DW_TAG_base_type
  uint8_t offs_uint32_t = 0x0000000e;
  uint8_t offs_uint64_t = 0x00000011;
  uint8_t offs_sint64_t = 0x00000014;
  uint8_t offs_uchar = 0x00000017;
  uint8_t offs_schar = 0x0000001a;

  DWARFExpressionTester t(yamldata, /*cu_index=*/1);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  // Constant is given as little-endian.
  bool is_signed = true;
  bool not_signed = false;

  //
  // Positive tests.
  //

  // Leave as is.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4u, 0x11, 0x22, 0x33, 0x44, //
              DW_OP_convert, offs_uint32_t, DW_OP_stack_value}),
      llvm::HasValue(GetScalar(64, 0x44332211, not_signed)));

  // Zero-extend to 64 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4u, 0x11, 0x22, 0x33, 0x44, //
              DW_OP_convert, offs_uint64_t, DW_OP_stack_value}),
      llvm::HasValue(GetScalar(64, 0x44332211, not_signed)));

  // Sign-extend to 64 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
              DW_OP_convert, offs_sint64_t, DW_OP_stack_value}),
      llvm::HasValue(GetScalar(64, 0xffffffffffeeddcc, is_signed)));

  // Sign-extend, then truncate.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
              DW_OP_convert, offs_sint64_t,          //
              DW_OP_convert, offs_uint32_t, DW_OP_stack_value}),
      llvm::HasValue(GetScalar(32, 0xffeeddcc, not_signed)));

  // Truncate to default unspecified (pointer-sized) type.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
                               DW_OP_convert, offs_sint64_t,          //
                               DW_OP_convert, 0x00, DW_OP_stack_value}),
                       llvm::HasValue(GetScalar(32, 0xffeeddcc, not_signed)));

  // Truncate to 8 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', DW_OP_convert,
                               offs_uchar, DW_OP_stack_value}),
                       llvm::HasValue(GetScalar(8, 'A', not_signed)));

  // Also truncate to 8 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', DW_OP_convert,
                               offs_schar, DW_OP_stack_value}),
                       llvm::HasValue(GetScalar(8, 'A', is_signed)));

  //
  // Errors.
  //

  // No Module.
  EXPECT_THAT_ERROR(Evaluate({DW_OP_const1s, 'X', DW_OP_convert, 0x00}, nullptr,
                             t.GetDwarfUnit())
                        .takeError(),
                    llvm::Failed());

  // No DIE.
  EXPECT_THAT_ERROR(
      t.Eval({DW_OP_const1s, 'X', DW_OP_convert, 0x01}).takeError(),
      llvm::Failed());

  // Unsupported.
  EXPECT_THAT_ERROR(
      t.Eval({DW_OP_const1s, 'X', DW_OP_convert, 0x1d}).takeError(),
      llvm::Failed());
}

TEST(DWARFExpression, DW_OP_stack_value) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_stack_value}), llvm::Failed());
}

TEST(DWARFExpression, DW_OP_piece) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_const2u, 0x11, 0x22, DW_OP_piece, 2,
                                 DW_OP_const2u, 0x33, 0x44, DW_OP_piece, 2}),
                       llvm::HasValue(GetScalar(32, 0x44332211, true)));
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_piece, 1, DW_OP_const1u, 0xff, DW_OP_piece, 1}),
      // Note that the "00" should really be "undef", but we can't
      // represent that yet.
      llvm::HasValue(GetScalar(16, 0xff00, true)));
}

TEST(DWARFExpression, DW_OP_piece_host_address) {
  static const uint8_t expr_data[] = {DW_OP_lit2, DW_OP_stack_value,
                                      DW_OP_piece, 40};
  llvm::ArrayRef<uint8_t> expr(expr_data, sizeof(expr_data));
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle, 4);

  // This tests if ap_int is extended to the right width.
  // expect 40*8 = 320 bits size.
  llvm::Expected<Value> result =
      DWARFExpression::Evaluate(nullptr, nullptr, nullptr, extractor, nullptr,
                                lldb::eRegisterKindDWARF, nullptr, nullptr);
  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  ASSERT_EQ(result->GetValueType(), Value::ValueType::HostAddress);
  ASSERT_EQ(result->GetBuffer().GetByteSize(), 40ul);
  const uint8_t *data = result->GetBuffer().GetBytes();
  ASSERT_EQ(data[0], 2);
  for (int i = 1; i < 40; i++) {
    ASSERT_EQ(data[i], 0);
  }
}

TEST(DWARFExpression, DW_OP_implicit_value) {
  unsigned char bytes = 4;

  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_implicit_value, bytes, 0x11, 0x22, 0x33, 0x44}),
      llvm::HasValue(GetScalar(8 * bytes, 0x44332211, true)));
}

TEST(DWARFExpression, DW_OP_unknown) {
  EXPECT_THAT_EXPECTED(
      Evaluate({0xff}),
      llvm::FailedWithMessage(
          "Unhandled opcode DW_OP_unknown_ff in DWARFExpression"));
}

TEST_F(DWARFExpressionMockProcessTest, DW_OP_deref) {
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit0, DW_OP_deref}), llvm::Failed());

  // Set up a mock process.
  ArchSpec arch("i386-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  lldb::TargetSP target_sp;
  lldb::PlatformSP platform_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  ASSERT_TRUE(target_sp);
  ASSERT_TRUE(target_sp->GetArchitecture().IsValid());
  ASSERT_TRUE(platform_sp);
  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<MockProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  ExecutionContext exe_ctx(process_sp);
  // Implicit location: *0x4.
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_lit4, DW_OP_deref, DW_OP_stack_value}, {}, {}, &exe_ctx),
      llvm::HasValue(GetScalar(32, 0x07060504, false)));
  // Memory location: *(*0x4).
  // Evaluate returns LLDB_INVALID_ADDRESS for all load addresses.
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit4, DW_OP_deref}, {}, {}, &exe_ctx),
                       llvm::HasValue(Scalar(LLDB_INVALID_ADDRESS)));
  // Memory location: *0x4.
  // Evaluate returns LLDB_INVALID_ADDRESS for all load addresses.
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit4}, {}, {}, &exe_ctx),
                       llvm::HasValue(Scalar(4)));
  // Implicit location: *0x4.
  // Evaluate returns LLDB_INVALID_ADDRESS for all load addresses.
  EXPECT_THAT_EXPECTED(
      Evaluate({DW_OP_lit4, DW_OP_deref, DW_OP_stack_value}, {}, {}, &exe_ctx),
      llvm::HasValue(GetScalar(32, 0x07060504, false)));
}

TEST_F(DWARFExpressionMockProcessTest, WASM_DW_OP_addr) {
  // Set up a wasm target
  ArchSpec arch("wasm32-unknown-unknown-wasm");
  lldb::PlatformSP host_platform_sp =
      platform_linux::PlatformLinux::CreateInstance(true, &arch);
  ASSERT_TRUE(host_platform_sp);
  Platform::SetHostPlatform(host_platform_sp);
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  lldb::TargetSP target_sp;
  lldb::PlatformSP platform_sp;
  debugger_sp->GetTargetList().CreateTarget(*debugger_sp, "", arch,
                                            lldb_private::eLoadDependentsNo,
                                            platform_sp, target_sp);

  ExecutionContext exe_ctx(target_sp, false);
  // DW_OP_addr takes a single operand of address size width:
  uint8_t expr[] = {DW_OP_addr, 0x40, 0x0, 0x0, 0x0};
  DataExtractor extractor(expr, sizeof(expr), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);

  llvm::Expected<Value> result = DWARFExpression::Evaluate(
      &exe_ctx, /*reg_ctx*/ nullptr, /*module_sp*/ {}, extractor,
      /*unit*/ nullptr, lldb::eRegisterKindLLDB,
      /*initial_value_ptr*/ nullptr,
      /*object_address_ptr*/ nullptr);

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  ASSERT_EQ(result->GetValueType(), Value::ValueType::LoadAddress);
}

TEST_F(DWARFExpressionMockProcessTest, WASM_DW_OP_addr_index) {
  const char *yamldata = R"(
--- !ELF
FileHeader:
  Class:   ELFCLASS64
  Data:    ELFDATA2LSB
  Type:    ET_EXEC
  Machine: EM_386
DWARF:
  debug_abbrev:
    - Table:
        - Code:            0x00000001
          Tag:             DW_TAG_compile_unit
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_addr_base
              Form:            DW_FORM_sec_offset

  debug_info:
    - Version:         5
      AddrSize:        4
      UnitType:        DW_UT_compile
      Entries:
        - AbbrCode:        0x00000001
          Values:
            - Value:           0x8 # Offset of the first Address past the header
        - AbbrCode:        0x0

  debug_addr:
    - Version: 5
      AddressSize: 4
      Entries:
        - Address: 0x1234
        - Address: 0x5678
)";

  // Can't use DWARFExpressionTester from above because subsystems overlap with
  // the fixture.
  SubsystemRAII<ObjectFileELF, SymbolFileDWARF> subsystems;
  llvm::Expected<TestFile> file = TestFile::fromYaml(yamldata);
  EXPECT_THAT_EXPECTED(file, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(file->moduleSpec());
  auto *dwarf_cu = llvm::cast<SymbolFileDWARF>(module_sp->GetSymbolFile())
                       ->DebugInfo()
                       .GetUnitAtIndex(0);
  ASSERT_TRUE(dwarf_cu);
  dwarf_cu->ExtractDIEsIfNeeded();

  // Set up a wasm target
  ArchSpec arch("wasm32-unknown-unknown-wasm");
  lldb::PlatformSP host_platform_sp =
      platform_linux::PlatformLinux::CreateInstance(true, &arch);
  ASSERT_TRUE(host_platform_sp);
  Platform::SetHostPlatform(host_platform_sp);
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  lldb::TargetSP target_sp;
  lldb::PlatformSP platform_sp;
  debugger_sp->GetTargetList().CreateTarget(*debugger_sp, "", arch,
                                            lldb_private::eLoadDependentsNo,
                                            platform_sp, target_sp);

  ExecutionContext exe_ctx(target_sp, false);

  auto evaluate = [&](DWARFExpression &expr) -> llvm::Expected<Value> {
    DataExtractor extractor;
    expr.GetExpressionData(extractor);
    return DWARFExpression::Evaluate(&exe_ctx, /*reg_ctx*/ nullptr,
                                     /*module_sp*/ {}, extractor, dwarf_cu,
                                     lldb::eRegisterKindLLDB,
                                     /*initial_value_ptr*/ nullptr,
                                     /*object_address_ptr*/ nullptr);
  };

  // DW_OP_addrx takes a single leb128 operand, the index in the addr table:
  uint8_t expr_data[] = {DW_OP_addrx, 0x01};
  DataExtractor extractor(expr_data, sizeof(expr_data), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);
  DWARFExpression expr(extractor);

  llvm::Expected<Value> result = evaluate(expr);
  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  ASSERT_EQ(result->GetValueType(), Value::ValueType::LoadAddress);
  ASSERT_EQ(result->GetScalar().UInt(), 0x5678u);

  ASSERT_TRUE(expr.Update_DW_OP_addr(dwarf_cu, 0xdeadbeef));
  result = evaluate(expr);
  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  ASSERT_EQ(result->GetValueType(), Value::ValueType::LoadAddress);
  ASSERT_EQ(result->GetScalar().UInt(), 0xdeadbeefu);
}

class CustomSymbolFileDWARF : public SymbolFileDWARF {
  static char ID;

public:
  using SymbolFileDWARF::SymbolFileDWARF;

  bool isA(const void *ClassID) const override {
    return ClassID == &ID || SymbolFile::isA(ClassID);
  }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }

  static llvm::StringRef GetPluginNameStatic() { return "custom_dwarf"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Symbol file reader with expression extensions.";
  }

  static void Initialize() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance,
                                  SymbolFileDWARF::DebuggerInitialize);
  }

  static void Terminate() { PluginManager::UnregisterPlugin(CreateInstance); }

  static lldb_private::SymbolFile *
  CreateInstance(lldb::ObjectFileSP objfile_sp) {
    return new CustomSymbolFileDWARF(std::move(objfile_sp),
                                     /*dwo_section_list*/ nullptr);
  }

  lldb::offset_t
  GetVendorDWARFOpcodeSize(const lldb_private::DataExtractor &data,
                           const lldb::offset_t data_offset,
                           const uint8_t op) const final {
    auto offset = data_offset;
    if (op != DW_OP_WASM_location) {
      return LLDB_INVALID_OFFSET;
    }

    // DW_OP_WASM_location WASM_GLOBAL:0x03 index:u32
    // Called with "arguments" 0x03 and 0x04
    // Location type:
    if (data.GetU8(&offset) != /* global */ 0x03) {
      return LLDB_INVALID_OFFSET;
    }

    // Index
    if (data.GetU32(&offset) != 0x04) {
      return LLDB_INVALID_OFFSET;
    }

    // Report the skipped distance:
    return offset - data_offset;
  }

  virtual bool ParseVendorDWARFOpcode(
      uint8_t op, const lldb_private::DataExtractor &opcodes,
      lldb::offset_t &offset,

      RegisterContext *reg_ctx, lldb::RegisterKind reg_kind,
      std::vector<lldb_private::Value> &stack) const override {
    if (op != DW_OP_WASM_location) {
      return false;
    }

    // DW_OP_WASM_location WASM_GLOBAL:0x03 index:u32
    // Called with "arguments" 0x03 and  0x04
    // Location type:
    if (opcodes.GetU8(&offset) != /* global */ 0x03) {
      return false;
    }

    // Index:
    if (opcodes.GetU32(&offset) != 0x04) {
      return false;
    }

    // Return some value:
    stack.push_back({GetScalar(32, 42, false)});
    return true;
  }
};

char CustomSymbolFileDWARF::ID;

static auto testExpressionVendorExtensions(lldb::ModuleSP module_sp,
                                           DWARFUnit &dwarf_unit,
                                           RegisterContext *reg_ctx) {
  // Test that expression extensions can be evaluated, for example
  // DW_OP_WASM_location which is not currently handled by DWARFExpression:
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_WASM_location, 0x03, // WASM_GLOBAL:0x03
                                 0x04, 0x00, 0x00,          // index:u32
                                 0x00, DW_OP_stack_value},
                                module_sp, &dwarf_unit, nullptr, reg_ctx),
                       llvm::HasValue(GetScalar(32, 42, false)));

  // Test that searches for opcodes work in the presence of extensions:
  uint8_t expr[] = {DW_OP_WASM_location,   0x03, 0x04, 0x00, 0x00, 0x00,
                    DW_OP_form_tls_address};
  DataExtractor extractor(expr, sizeof(expr), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);
  DWARFExpression dwarf_expr(extractor);
  ASSERT_TRUE(dwarf_expr.ContainsThreadLocalStorage(&dwarf_unit));
}

TEST(DWARFExpression, Extensions) {
  const char *yamldata = R"(
--- !WASM
FileHeader:
  Version:         0x1
Sections:
  - Type:            TYPE
    Signatures:
      - Index:           0
        ParamTypes:
          - I32
        ReturnTypes:
          - I32
  - Type:            FUNCTION
    FunctionTypes:   [ 0 ]
  - Type:            TABLE
    Tables:
      - Index:           0
        ElemType:        FUNCREF
        Limits:
          Flags:           [ HAS_MAX ]
          Minimum:         0x1
          Maximum:         0x1
  - Type:            MEMORY
    Memories:
      - Flags:           [ HAS_MAX ]
        Minimum:         0x100
        Maximum:         0x100
  - Type:            GLOBAL
    Globals:
      - Index:           0
        Type:            I32
        Mutable:         true
        InitExpr:
          Opcode:          I32_CONST
          Value:           65536
  - Type:            EXPORT
    Exports:
      - Name:            memory
        Kind:            MEMORY
        Index:           0
      - Name:            square
        Kind:            FUNCTION
        Index:           0
      - Name:            __indirect_function_table
        Kind:            TABLE
        Index:           0
  - Type:            CODE
    Functions:
      - Index:           0
        Locals:
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
        Body:            2300210141102102200120026B21032003200036020C200328020C2104200328020C2105200420056C210620060F0B
  - Type:            CUSTOM
    Name:            name
    FunctionNames:
      - Index:           0
        Name:            square
    GlobalNames:
      - Index:           0
        Name:            __stack_pointer
  - Type:            CUSTOM
    Name:            .debug_abbrev
    Payload:         011101250E1305030E10171B0E110112060000022E01110112064018030E3A0B3B0B271949133F1900000305000218030E3A0B3B0B49130000042400030E3E0B0B0B000000
  - Type:            CUSTOM
    Name:            .debug_info
    Payload:         510000000400000000000401670000001D005E000000000000000A000000020000003C00000002020000003C00000004ED00039F5700000001014D0000000302910C0400000001014D000000000400000000050400
  - Type:            CUSTOM
    Name:            .debug_str
    Payload:         696E740076616C756500513A5C70616F6C6F7365764D5346545C6C6C766D2D70726F6A6563745C6C6C64625C746573745C4150495C66756E6374696F6E616C69746965735C6764625F72656D6F74655F636C69656E745C737175617265007371756172652E6300636C616E672076657273696F6E2031382E302E30202868747470733A2F2F6769746875622E636F6D2F6C6C766D2F6C6C766D2D70726F6A65637420373535303166353336323464653932616166636532663164613639386232343961373239336463372900
  - Type:            CUSTOM
    Name:            .debug_line
    Payload:         64000000040020000000010101FB0E0D000101010100000001000001007371756172652E6300000000000005020200000001000502250000000301050A0A010005022C00000005120601000502330000000510010005023A0000000503010005023E000000000101
)";

  SubsystemRAII<FileSystem, HostInfo, ObjectFileWasm, SymbolVendorWasm>
      subsystems;

  // Set up a wasm target.
  ArchSpec arch("wasm32-unknown-unknown-wasm");
  lldb::PlatformSP host_platform_sp =
      platform_linux::PlatformLinux::CreateInstance(true, &arch);
  ASSERT_TRUE(host_platform_sp);
  Platform::SetHostPlatform(host_platform_sp);
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  lldb::TargetSP target_sp;
  lldb::PlatformSP platform_sp;
  debugger_sp->GetTargetList().CreateTarget(*debugger_sp, "", arch,
                                            lldb_private::eLoadDependentsNo,
                                            platform_sp, target_sp);
  // Set up a mock process and thread.
  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<MockProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  MockThread thread(*process_sp);
  const uint32_t kExpectedValue = 42;
  lldb::RegisterContextSP reg_ctx_sp = std::make_shared<MockRegisterContext>(
      thread, RegisterValue(kExpectedValue));
  thread.SetRegisterContext(reg_ctx_sp);

  llvm::Expected<TestFile> file = TestFile::fromYaml(yamldata);
  EXPECT_THAT_EXPECTED(file, llvm::Succeeded());
  auto module_sp = std::make_shared<Module>(file->moduleSpec());
  auto obj_file_sp = module_sp->GetObjectFile()->shared_from_this();
  SymbolFileWasm sym_file_wasm(obj_file_sp, nullptr);
  auto *dwarf_unit = sym_file_wasm.DebugInfo().GetUnitAtIndex(0);

  testExpressionVendorExtensions(module_sp, *dwarf_unit, reg_ctx_sp.get());
}

TEST(DWARFExpression, ExtensionsSplitSymbols) {
  const char *skeleton_yamldata = R"(
--- !WASM
FileHeader:
  Version:         0x1
Sections:
  - Type:            TYPE
    Signatures:
      - Index:           0
        ParamTypes:
          - I32
        ReturnTypes:
          - I32
  - Type:            FUNCTION
    FunctionTypes:   [ 0 ]
  - Type:            TABLE
    Tables:
      - Index:           0
        ElemType:        FUNCREF
        Limits:
          Flags:           [ HAS_MAX ]
          Minimum:         0x1
          Maximum:         0x1
  - Type:            MEMORY
    Memories:
      - Flags:           [ HAS_MAX ]
        Minimum:         0x100
        Maximum:         0x100
  - Type:            GLOBAL
    Globals:
      - Index:           0
        Type:            I32
        Mutable:         true
        InitExpr:
          Opcode:          I32_CONST
          Value:           65536
  - Type:            EXPORT
    Exports:
      - Name:            memory
        Kind:            MEMORY
        Index:           0
      - Name:            square
        Kind:            FUNCTION
        Index:           0
      - Name:            __indirect_function_table
        Kind:            TABLE
        Index:           0
  - Type:            CODE
    Functions:
      - Index:           0
        Locals:
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
        Body:            2300210141102102200120026B21032003200036020C200328020C2104200328020C2105200420056C210620060F0B
  - Type:            CUSTOM
    Name:            name
    FunctionNames:
      - Index:           0
        Name:            square
    GlobalNames:
      - Index:           0
        Name:            __stack_pointer
  - Type:            CUSTOM
    Name:            external_debug_info
    Payload:         167371756172652E7761736D2E64656275672E7761736D
)";

  const char *sym_yamldata = R"(
--- !WASM
FileHeader:
  Version:         0x1
Sections:
  - Type:            TYPE
    Signatures:
      - Index:           0
        ParamTypes:
          - I32
        ReturnTypes:
          - I32
  - Type:            FUNCTION
    FunctionTypes:   [ 0 ]
  - Type:            TABLE
    Tables:
      - Index:           0
        ElemType:        FUNCREF
        Limits:
          Flags:           [ HAS_MAX ]
          Minimum:         0x1
          Maximum:         0x1
  - Type:            MEMORY
    Memories:
      - Flags:           [ HAS_MAX ]
        Minimum:         0x100
        Maximum:         0x100
  - Type:            GLOBAL
    Globals:
      - Index:           0
        Type:            I32
        Mutable:         true
        InitExpr:
          Opcode:          I32_CONST
          Value:           65536
  - Type:            EXPORT
    Exports:
      - Name:            memory
        Kind:            MEMORY
        Index:           0
      - Name:            square
        Kind:            FUNCTION
        Index:           0
      - Name:            __indirect_function_table
        Kind:            TABLE
        Index:           0
  - Type:            CODE
    Functions:
      - Index:           0
        Locals:
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
          - Type:            I32
            Count:           1
        Body:            2300210141102102200120026B21032003200036020C200328020C2104200328020C2105200420056C210620060F0B
  - Type:            CUSTOM
    Name:            name
    FunctionNames:
      - Index:           0
        Name:            square
    GlobalNames:
      - Index:           0
        Name:            __stack_pointer
  - Type:            CUSTOM
    Name:            .debug_abbrev
    Payload:         011101250E1305030E10171B0E110112060000022E01110112064018030E3A0B3B0B271949133F1900000305000218030E3A0B3B0B49130000042400030E3E0B0B0B000000
  - Type:            CUSTOM
    Name:            .debug_info
    Payload:         510000000400000000000401670000001D005E0000000000000004000000020000003C00000002020000003C00000004ED00039F5700000001014D0000000302910C5100000001014D000000000400000000050400
  - Type:            CUSTOM
    Name:            .debug_str
    Payload:         696E7400513A5C70616F6C6F7365764D5346545C6C6C766D2D70726F6A6563745C6C6C64625C746573745C4150495C66756E6374696F6E616C69746965735C6764625F72656D6F74655F636C69656E740076616C756500737175617265007371756172652E6300636C616E672076657273696F6E2031382E302E30202868747470733A2F2F6769746875622E636F6D2F6C6C766D2F6C6C766D2D70726F6A65637420373535303166353336323464653932616166636532663164613639386232343961373239336463372900
  - Type:            CUSTOM
    Name:            .debug_line
    Payload:         64000000040020000000010101FB0E0D000101010100000001000001007371756172652E6300000000000005020200000001000502250000000301050A0A010005022C00000005120601000502330000000510010005023A0000000503010005023E000000000101
)";

  SubsystemRAII<FileSystem, HostInfo, ObjectFileWasm, SymbolVendorWasm>
      subsystems;

  // Set up a wasm target.
  ArchSpec arch("wasm32-unknown-unknown-wasm");
  lldb::PlatformSP host_platform_sp =
      platform_linux::PlatformLinux::CreateInstance(true, &arch);
  ASSERT_TRUE(host_platform_sp);
  Platform::SetHostPlatform(host_platform_sp);
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  lldb::TargetSP target_sp;
  lldb::PlatformSP platform_sp;
  debugger_sp->GetTargetList().CreateTarget(*debugger_sp, "", arch,
                                            lldb_private::eLoadDependentsNo,
                                            platform_sp, target_sp);
  // Set up a mock process and thread.
  lldb::ListenerSP listener_sp(Listener::MakeListener("dummy"));
  lldb::ProcessSP process_sp =
      std::make_shared<MockProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  MockThread thread(*process_sp);
  const uint32_t kExpectedValue = 42;
  lldb::RegisterContextSP reg_ctx_sp = std::make_shared<MockRegisterContext>(
      thread, RegisterValue(kExpectedValue));
  thread.SetRegisterContext(reg_ctx_sp);

  llvm::Expected<TestFile> skeleton_file =
      TestFile::fromYaml(skeleton_yamldata);
  EXPECT_THAT_EXPECTED(skeleton_file, llvm::Succeeded());
  auto skeleton_module_sp =
      std::make_shared<Module>(skeleton_file->moduleSpec());

  llvm::Expected<TestFile> sym_file = TestFile::fromYaml(sym_yamldata);
  EXPECT_THAT_EXPECTED(sym_file, llvm::Succeeded());
  auto sym_module_sp = std::make_shared<Module>(sym_file->moduleSpec());

  auto obj_file_sp = sym_module_sp->GetObjectFile()->shared_from_this();
  SymbolFileWasm sym_file_wasm(obj_file_sp, nullptr);
  auto *dwarf_unit = sym_file_wasm.DebugInfo().GetUnitAtIndex(0);

  testExpressionVendorExtensions(sym_module_sp, *dwarf_unit, reg_ctx_sp.get());
}

TEST_F(DWARFExpressionMockProcessTest, DW_OP_piece_file_addr) {
  using ::testing::ByMove;
  using ::testing::ElementsAre;
  using ::testing::Return;

  // Set up a mock process.
  ArchSpec arch("i386-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  lldb::PlatformSP platform_sp;
  auto target_sp =
      std::make_shared<MockTarget>(*debugger_sp, arch, platform_sp);
  ASSERT_TRUE(target_sp);
  ASSERT_TRUE(target_sp->GetArchitecture().IsValid());

  EXPECT_CALL(*target_sp, ReadMemory(0x40, 1))
      .WillOnce(Return(ByMove(std::vector<uint8_t>{0x11})));
  EXPECT_CALL(*target_sp, ReadMemory(0x50, 1))
      .WillOnce(Return(ByMove(std::vector<uint8_t>{0x22})));

  ExecutionContext exe_ctx(static_cast<lldb::TargetSP>(target_sp), false);

  uint8_t expr[] = {DW_OP_addr, 0x40, 0x0, 0x0, 0x0, DW_OP_piece, 1,
                    DW_OP_addr, 0x50, 0x0, 0x0, 0x0, DW_OP_piece, 1};
  DataExtractor extractor(expr, sizeof(expr), lldb::eByteOrderLittle,
                          /*addr_size=*/4);
  llvm::Expected<Value> result = DWARFExpression::Evaluate(
      &exe_ctx, /*reg_ctx=*/nullptr, /*module_sp=*/{}, extractor,
      /*unit=*/nullptr, lldb::eRegisterKindLLDB,
      /*initial_value_ptr=*/nullptr,
      /*object_address_ptr=*/nullptr);

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  ASSERT_EQ(result->GetValueType(), Value::ValueType::HostAddress);
  ASSERT_THAT(result->GetBuffer().GetData(), ElementsAre(0x11, 0x22));
}
