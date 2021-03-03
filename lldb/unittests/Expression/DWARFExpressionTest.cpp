//===-- DWARFExpressionTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DWARFExpression.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "TestingSupport/Symbol/YAMLModuleTester.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Reproducer.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

static llvm::Expected<Scalar> Evaluate(llvm::ArrayRef<uint8_t> expr,
                                       lldb::ModuleSP module_sp = {},
                                       DWARFUnit *unit = nullptr,
                                       ExecutionContext *exe_ctx = nullptr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);
  Value result;
  Status status;
  if (!DWARFExpression::Evaluate(exe_ctx, /*reg_ctx*/ nullptr, module_sp,
                                 extractor, unit, lldb::eRegisterKindLLDB,
                                 /*initial_value_ptr*/ nullptr,
                                 /*object_address_ptr*/ nullptr, result,
                                 &status))
    return status.ToError();

  switch (result.GetValueType()) {
  case Value::ValueType::Scalar:
    return result.GetScalar();
  case Value::ValueType::LoadAddress:
    return LLDB_INVALID_ADDRESS;
  case Value::ValueType::HostAddress: {
    // Convert small buffers to scalars to simplify the tests.
    DataBufferHeap &buf = result.GetBuffer();
    if (buf.GetByteSize() <= 8) {
      uint64_t val = 0;
      memcpy(&val, buf.GetBytes(), buf.GetByteSize());
      return Scalar(llvm::APInt(buf.GetByteSize()*8, val, false));
    }
  }
    LLVM_FALLTHROUGH;
  default:
    return status.ToError();
  }
}

class DWARFExpressionTester : public YAMLModuleTester {
public:
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
    llvm::cantFail(repro::Reproducer::Initialize(repro::ReproducerMode::Off, {}));
    FileSystem::Initialize();
    HostInfo::Initialize();
    platform_linux::PlatformLinux::Initialize();
  }
  void TearDown() override {
    platform_linux::PlatformLinux::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
    repro::Reproducer::Terminate();
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
  uint8_t offs_uint32_t = 0x0000000e;
  uint8_t offs_uint64_t = 0x00000011;
  uint8_t offs_sint64_t = 0x00000014;
  uint8_t offs_uchar = 0x00000017;
  uint8_t offs_schar = 0x0000001a;

  DWARFExpressionTester t(yamldata);
  ASSERT_TRUE((bool)t.GetDwarfUnit());

  // Constant is given as little-endian.
  bool is_signed = true;
  bool not_signed = false;

  //
  // Positive tests.
  //

  // Leave as is.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4u, 0x11, 0x22, 0x33, 0x44, //
                               DW_OP_convert, offs_uint32_t}),
                       llvm::HasValue(GetScalar(64, 0x44332211, not_signed)));

  // Zero-extend to 64 bits.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4u, 0x11, 0x22, 0x33, 0x44, //
                               DW_OP_convert, offs_uint64_t}),
                       llvm::HasValue(GetScalar(64, 0x44332211, not_signed)));

  // Sign-extend to 64 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
              DW_OP_convert, offs_sint64_t}),
      llvm::HasValue(GetScalar(64, 0xffffffffffeeddcc, is_signed)));

  // Sign-extend, then truncate.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
                               DW_OP_convert, offs_sint64_t,          //
                               DW_OP_convert, offs_uint32_t}),
                       llvm::HasValue(GetScalar(32, 0xffeeddcc, not_signed)));

  // Truncate to default unspecified (pointer-sized) type.
  EXPECT_THAT_EXPECTED(t.Eval({DW_OP_const4s, 0xcc, 0xdd, 0xee, 0xff, //
                               DW_OP_convert, offs_sint64_t,          //
                               DW_OP_convert, 0x00}),
                       llvm::HasValue(GetScalar(32, 0xffeeddcc, not_signed)));

  // Truncate to 8 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', DW_OP_convert, offs_uchar}),
      llvm::HasValue(GetScalar(8, 'A', not_signed)));

  // Also truncate to 8 bits.
  EXPECT_THAT_EXPECTED(
      t.Eval({DW_OP_const4s, 'A', 'B', 'C', 'D', DW_OP_convert, offs_schar}),
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
    using Process::Process;
    ConstString GetPluginName() override { return ConstString("mock process"); }
    uint32_t GetPluginVersion() override { return 0; }
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
}
