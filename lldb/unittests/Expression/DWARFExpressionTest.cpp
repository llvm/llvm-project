//===-- DWARFExpressionTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDwo.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;

static llvm::Expected<Scalar> Evaluate(llvm::ArrayRef<uint8_t> expr,
                                       lldb::ModuleSP module_sp = {},
                                       DWARFUnit *unit = nullptr,
                                       ExecutionContext *exe_ctx = nullptr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);

  llvm::Expected<Value> result =
      DWARFExpression::Evaluate(exe_ctx, /*reg_ctx*/ nullptr, module_sp,
                                extractor, unit, lldb::eRegisterKindLLDB,
                                /*initial_value_ptr*/ nullptr,
                                /*object_address_ptr*/ nullptr);
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
      return Scalar(llvm::APInt(buf.GetByteSize()*8, val, false));
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
  DWARFExpressionTester(llvm::StringRef yaml_data, size_t cu_index) :
      YAMLModuleTester(yaml_data, cu_index) {}

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
                    lldb::addr_t *load_addr_ptr = nullptr) /*override*/ {
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

  struct MockProcess : Process {
    MockProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
        : Process(target_sp, listener_sp) {}

    llvm::StringRef GetPluginName() override { return "mock process"; }
    bool CanDebug(lldb::TargetSP target,
                  bool plugin_specified_by_name) override {
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
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_lit4, DW_OP_deref, DW_OP_stack_value},
                                {}, {}, &exe_ctx),
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

  bool
  ParseVendorDWARFOpcode(uint8_t op, const lldb_private::DataExtractor &opcodes,
                         lldb::offset_t &offset,
                         std::vector<lldb_private::Value> &stack) const final {
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
                                           DWARFUnit &dwarf_unit) {
  // Test that expression extensions can be evaluated, for example
  // DW_OP_WASM_location which is not currently handled by DWARFExpression:
  EXPECT_THAT_EXPECTED(Evaluate({DW_OP_WASM_location, 0x03, // WASM_GLOBAL:0x03
                                 0x04, 0x00, 0x00,          // index:u32
                                 0x00, DW_OP_stack_value},
                                module_sp, &dwarf_unit),
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
  debug_info:
    - Version:         4
      AddrSize:        4
      Entries:
        - AbbrCode:        0x1
        - AbbrCode:        0x0
)";

  SubsystemRAII<FileSystem, HostInfo, TypeSystemClang, ObjectFileELF,
                CustomSymbolFileDWARF>
      subsystems;

  llvm::Expected<TestFile> file = TestFile::fromYaml(yamldata);
  EXPECT_THAT_EXPECTED(file, llvm::Succeeded());

  auto module_sp = std::make_shared<Module>(file->moduleSpec());
  auto &symfile =
      *llvm::cast<CustomSymbolFileDWARF>(module_sp->GetSymbolFile());
  auto *dwarf_unit = symfile.DebugInfo().GetUnitAtIndex(0);

  testExpressionVendorExtensions(module_sp, *dwarf_unit);
}

TEST(DWARFExpression, ExtensionsDWO) {
  const char *skeleton_yamldata = R"(
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
          Tag:             DW_TAG_skeleton_unit
          Children:        DW_CHILDREN_no
          Attributes:
            - Attribute:       DW_AT_dwo_name
              Form:            DW_FORM_string
            - Attribute:       DW_AT_dwo_id
              Form:            DW_FORM_data4
  debug_info:
    - Version:         4
      AddrSize:        4
      Entries:
        - AbbrCode:        0x1
          Values:
            - CStr:           "dwo_unit"
            - Value:           0x01020304
        - AbbrCode:        0x0
)";

  // .dwo sections aren't currently supported by dwarfyaml. The dwo_yamldata
  // contents where generated by roundtripping the following yaml through
  // yaml2obj | obj2yaml and renaming the sections. This works because the
  // structure of the .dwo and non-.dwo sections is identical.
  //
  // --- !ELF
  // FileHeader:
  //   Class:   ELFCLASS64
  //   Data:    ELFDATA2LSB
  //   Type:    ET_EXEC
  //   Machine: EM_386
  // DWARF:
  //   debug_abbrev: #.dwo
  //     - Table:
  //         - Code:            0x00000001
  //           Tag:             DW_TAG_compile_unit
  //           Children:        DW_CHILDREN_no
  //           Attributes:
  //             - Attribute:       DW_AT_dwo_id
  //               Form:            DW_FORM_data4
  //   debug_info: #.dwo
  //     - Version:         4
  //       AddrSize:        4
  //       Entries:
  //         - AbbrCode:        0x1
  //           Values:
  //             - Value:           0x0120304
  //         - AbbrCode:        0x0
  const char *dwo_yamldata = R"(
--- !ELF
FileHeader:
  Class:           ELFCLASS64
  Data:            ELFDATA2LSB
  Type:            ET_EXEC
  Machine:         EM_386
Sections:
  - Name:            .debug_abbrev.dwo
    Type:            SHT_PROGBITS
    AddressAlign:    0x1
    Content:         '0111007506000000'
  - Name:            .debug_info.dwo
    Type:            SHT_PROGBITS
    AddressAlign:    0x1
    Content:         0D00000004000000000004010403020100
)";

  SubsystemRAII<FileSystem, HostInfo, ObjectFileELF, CustomSymbolFileDWARF>
      subsystems;

  llvm::Expected<TestFile> skeleton_file =
      TestFile::fromYaml(skeleton_yamldata);
  EXPECT_THAT_EXPECTED(skeleton_file, llvm::Succeeded());
  llvm::Expected<TestFile> dwo_file = TestFile::fromYaml(dwo_yamldata);
  EXPECT_THAT_EXPECTED(dwo_file, llvm::Succeeded());

  auto skeleton_module_sp =
      std::make_shared<Module>(skeleton_file->moduleSpec());
  auto &skeleton_symfile =
      *llvm::cast<CustomSymbolFileDWARF>(skeleton_module_sp->GetSymbolFile());

  auto dwo_module_sp = std::make_shared<Module>(dwo_file->moduleSpec());
  SymbolFileDWARFDwo dwo_symfile(
      skeleton_symfile, dwo_module_sp->GetObjectFile()->shared_from_this(),
      0x0120304);
  auto *dwo_dwarf_unit = dwo_symfile.DebugInfo().GetUnitAtIndex(0);

  testExpressionVendorExtensions(dwo_module_sp, *dwo_dwarf_unit);
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
                          /*addr_size*/ 4);
  llvm::Expected<Value> result = DWARFExpression::Evaluate(
      &exe_ctx, /*reg_ctx*/ nullptr, /*module_sp*/ {}, extractor,
      /*unit*/ nullptr, lldb::eRegisterKindLLDB,
      /*initial_value_ptr*/ nullptr,
      /*object_address_ptr*/ nullptr);

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  ASSERT_EQ(result->GetValueType(), Value::ValueType::HostAddress);
  ASSERT_THAT(result->GetBuffer().GetData(), ElementsAre(0x11, 0x22));
}
