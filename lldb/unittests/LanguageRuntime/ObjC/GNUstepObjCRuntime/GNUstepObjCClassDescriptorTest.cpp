//===-- GNUstepObjCClassDescriptorTest.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/LanguageRuntime/ObjC/GNUstepObjCRuntime/GNUstepObjCClassDescriptor.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Platform/Windows/PlatformWindows.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Listener.h"

#include "gtest/gtest.h"

#include <cstring>

using namespace lldb;
using namespace lldb_private;

namespace {

/// Serves memory reads out of one contiguous block of fake inferior memory, so
/// that class structures can be laid out byte by byte and handed to the
/// descriptor without a live process.
class FakeProcess : public Process {
public:
  static constexpr addr_t g_base_addr = 0x100000;
  static constexpr size_t g_size = 0x2000;

  FakeProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp), m_memory(g_size, 0) {}

  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  bool IsAlive() override { return true; }
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "fake"; }

  size_t DoReadMemory(const ProcessAddress &process_addr, void *buf,
                      size_t size, Status &error) override {
    const addr_t vm_addr = process_addr.GetValue();
    if (vm_addr < g_base_addr || vm_addr >= g_base_addr + m_memory.size()) {
      error = Status::FromErrorString("address is not mapped");
      return 0;
    }
    const size_t offset = vm_addr - g_base_addr;
    const size_t bytes = std::min(size, m_memory.size() - offset);
    std::memcpy(buf, m_memory.data() + offset, bytes);
    return bytes;
  }

  // The targets libobjc2 supports are all little-endian.
  void WriteInteger(addr_t addr, uint64_t value, uint32_t byte_size) {
    const size_t offset = addr - g_base_addr;
    for (uint32_t i = 0; i < byte_size; i++)
      m_memory[offset + i] = (value >> (8 * i)) & 0xff;
  }

  void WriteCString(addr_t addr, llvm::StringRef str) {
    const size_t offset = addr - g_base_addr;
    std::memcpy(m_memory.data() + offset, str.data(), str.size());
    m_memory[offset + str.size()] = '\0';
  }

  void Fill(addr_t addr, uint8_t byte, size_t count) {
    const size_t offset = addr - g_base_addr;
    std::memset(m_memory.data() + offset, byte, count);
  }

  std::vector<uint8_t> m_memory;
};

/// The parts of libobjc2's `struct objc_class` this test lays out. The first
/// three fields are pointers and the rest are `long`, which is why the two
/// sizes are tracked separately.
struct DataModel {
  const char *triple;
  uint32_t pointer_size;
  uint32_t long_size;
};

class GNUstepClassDescriptorTest : public ::testing::TestWithParam<DataModel> {
public:
  void SetUp() override {
    ArchSpec arch(GetParam().triple);
    PlatformSP platform_sp =
        arch.GetTriple().isOSWindows()
            ? PlatformWindows::CreateInstance(true, &arch)
            : platform_linux::PlatformLinux::CreateInstance(true, &arch);
    Platform::SetHostPlatform(platform_sp);

    m_debugger_sp = Debugger::CreateInstance();
    m_debugger_sp->GetTargetList().CreateTarget(
        *m_debugger_sp, "", arch, eLoadDependentsNo, platform_sp, m_target_sp);
    ASSERT_TRUE(m_target_sp);

    ListenerSP listener_sp(Listener::MakeListener("fake"));
    m_process_sp = std::make_shared<FakeProcess>(m_target_sp, listener_sp);
    struct TargetHack : public Target {
      void SetProcess(ProcessSP process) { m_process_sp = process; }
    };
    static_cast<TargetHack *>(m_target_sp.get())->SetProcess(m_process_sp);
  }

  void TearDown() override {
    m_process_sp.reset();
    m_target_sp.reset();
    m_debugger_sp.reset();
  }

  FakeProcess &GetProcess() {
    return *static_cast<FakeProcess *>(m_process_sp.get());
  }

  uint32_t PointerSize() const { return GetParam().pointer_size; }
  uint32_t LongSize() const { return GetParam().long_size; }

  addr_t InfoOffset() const { return 3 * PointerSize() + LongSize(); }
  addr_t InstanceSizeOffset() const {
    return 3 * PointerSize() + 2 * LongSize();
  }
  addr_t ClassSize() const { return 3 * PointerSize() + 6 * LongSize(); }

  /// Lays out a class structure, returning its address.
  addr_t WriteClass(addr_t addr, addr_t metaclass, addr_t superclass,
                    addr_t name_addr, uint64_t info, int64_t instance_size) {
    FakeProcess &process = GetProcess();
    process.WriteInteger(addr, metaclass, PointerSize());
    process.WriteInteger(addr + PointerSize(), superclass, PointerSize());
    process.WriteInteger(addr + 2 * PointerSize(), name_addr, PointerSize());
    process.WriteInteger(addr + InfoOffset(), info, LongSize());
    process.WriteInteger(addr + InstanceSizeOffset(),
                         static_cast<uint64_t>(instance_size), LongSize());
    return addr;
  }

  // Flags from libobjc2's enum objc_class_flags.
  static constexpr uint64_t g_flag_meta = 1ULL << 0;
  static constexpr uint64_t g_flag_resolved = 1ULL << 9;

  SubsystemRAII<FileSystem, HostInfo, platform_linux::PlatformLinux,
                PlatformWindows>
      m_subsystems;
  DebuggerSP m_debugger_sp;
  TargetSP m_target_sp;
  ProcessSP m_process_sp;
};

// Addresses inside the fake memory block. Kept far from its edges so a cache
// read around a structure stays mapped.
constexpr addr_t g_class_addr = FakeProcess::g_base_addr + 0x100;
constexpr addr_t g_metaclass_addr = FakeProcess::g_base_addr + 0x200;
constexpr addr_t g_superclass_addr = FakeProcess::g_base_addr + 0x300;
constexpr addr_t g_superclass_meta_addr = FakeProcess::g_base_addr + 0x380;
constexpr addr_t g_name_addr = FakeProcess::g_base_addr + 0x400;
constexpr addr_t g_super_name_addr = FakeProcess::g_base_addr + 0x480;

/// A well-formed class parses on every data model. This is what proves the
/// field offsets track the target's `long` size rather than its pointer size:
/// on Windows the instance size sits four bytes earlier than on Linux.
TEST_P(GNUstepClassDescriptorTest, ParsesWellFormedClass) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "Derived");
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, g_name_addr,
             g_flag_resolved, 42);
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr,
             g_flag_meta | g_flag_resolved, 0);

  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  ASSERT_TRUE(descriptor.IsValid());
  EXPECT_EQ(descriptor.GetClassName(), ConstString("Derived"));
  EXPECT_EQ(descriptor.GetInstanceSize(), 42u);
  EXPECT_EQ(descriptor.GetISA(), g_class_addr);
}

/// A descriptor built from a class object's own ISA (its metaclass) must say
/// so. GetDynamicTypeAndAddress relies on this to refuse a dynamic type for
/// values that are Class rather than instances: libobjc2 gives a metaclass the
/// same name as its class, so nothing else distinguishes the two.
TEST_P(GNUstepClassDescriptorTest, IdentifiesMetaclass) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "Derived");
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, g_name_addr,
             g_flag_resolved, 42);
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr,
             g_flag_meta | g_flag_resolved, 0);

  GNUstepObjCClassDescriptor instance_class(m_process_sp, g_class_addr);
  ASSERT_TRUE(instance_class.IsValid());
  EXPECT_FALSE(instance_class.IsMetaclass());

  GNUstepObjCClassDescriptor metaclass(m_process_sp, g_metaclass_addr);
  ASSERT_TRUE(metaclass.IsValid());
  EXPECT_TRUE(metaclass.IsMetaclass());
  EXPECT_EQ(metaclass.GetClassName(), ConstString("Derived"));
}

TEST_P(GNUstepClassDescriptorTest, WalksSuperclassChain) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "Derived");
  process.WriteCString(g_super_name_addr, "Base");
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, g_name_addr,
             g_flag_resolved, 42);
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr,
             g_flag_meta | g_flag_resolved, 0);
  WriteClass(g_superclass_addr, g_superclass_meta_addr, 0, g_super_name_addr,
             g_flag_resolved, 16);
  WriteClass(g_superclass_meta_addr, g_superclass_meta_addr, 0,
             g_super_name_addr, g_flag_meta | g_flag_resolved, 0);

  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  ASSERT_TRUE(descriptor.IsValid());
  auto superclass_sp = descriptor.GetSuperclass();
  ASSERT_TRUE(superclass_sp);
  EXPECT_EQ(superclass_sp->GetClassName(), ConstString("Base"));
  EXPECT_EQ(superclass_sp->GetInstanceSize(), 16u);
}

/// Before the runtime resolves a class, `super_class` still holds a name
/// pointer and `instance_size` the negated size of only this class's ivars,
/// so neither may be reported.
TEST_P(GNUstepClassDescriptorTest, UnresolvedClassHidesSuperclassAndSize) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "Derived");
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, g_name_addr,
             /*info=*/0, -8);
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr, g_flag_meta,
             0);

  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  ASSERT_TRUE(descriptor.IsValid());
  EXPECT_EQ(descriptor.GetClassName(), ConstString("Derived"));
  EXPECT_EQ(descriptor.GetInstanceSize(), 0u);
  EXPECT_FALSE(descriptor.GetSuperclass());
}

TEST_P(GNUstepClassDescriptorTest, RejectsUnmappedAddress) {
  GNUstepObjCClassDescriptor descriptor(m_process_sp, 0xdead0000);
  EXPECT_FALSE(descriptor.IsValid());
}

TEST_P(GNUstepClassDescriptorTest, RejectsMisalignedAddress) {
  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr + 1);
  EXPECT_FALSE(descriptor.IsValid());
}

TEST_P(GNUstepClassDescriptorTest, RejectsNullNamePointer) {
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, /*name=*/0,
             g_flag_resolved, 42);
  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  EXPECT_FALSE(descriptor.IsValid());
}

/// A name that never terminates is not a class name, and must not drag in
/// unbounded amounts of inferior memory.
TEST_P(GNUstepClassDescriptorTest, RejectsUnterminatedName) {
  FakeProcess &process = GetProcess();
  process.Fill(g_name_addr, 'A', 0x800);
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, g_name_addr,
             g_flag_resolved, 42);
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr,
             g_flag_meta | g_flag_resolved, 0);

  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  EXPECT_FALSE(descriptor.IsValid());
}

/// Arbitrary readable memory must not be accepted as a class: a class and its
/// metaclass have to disagree about the meta flag.
TEST_P(GNUstepClassDescriptorTest, RejectsClassWhoseMetaclassIsNotMeta) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "NotAClass");
  WriteClass(g_class_addr, g_metaclass_addr, g_superclass_addr, g_name_addr,
             g_flag_resolved, 42);
  // The "metaclass" is missing the meta flag.
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr,
             g_flag_resolved, 0);

  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  EXPECT_FALSE(descriptor.IsValid());
}

TEST_P(GNUstepClassDescriptorTest, RejectsNullMetaclass) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "Derived");
  WriteClass(g_class_addr, /*metaclass=*/0, g_superclass_addr, g_name_addr,
             g_flag_resolved, 42);
  GNUstepObjCClassDescriptor descriptor(m_process_sp, g_class_addr);
  EXPECT_FALSE(descriptor.IsValid());
}

/// Tagged pointer payloads are shifted by the tag width, and a signed payload
/// has to be sign-extended from the target's pointer width.
TEST_P(GNUstepClassDescriptorTest, DecodesTaggedPointerPayload) {
  FakeProcess &process = GetProcess();
  process.WriteCString(g_name_addr, "NSSmallInt");
  WriteClass(g_class_addr, g_metaclass_addr, 0, g_name_addr, g_flag_resolved,
             0);
  WriteClass(g_metaclass_addr, g_metaclass_addr, 0, g_name_addr,
             g_flag_meta | g_flag_resolved, 0);

  const bool is_64_bit = PointerSize() == 8;
  const uint64_t tag = is_64_bit ? 3 : 1;
  const uint32_t shift = is_64_bit ? 3 : 1;
  const uint64_t pointer_mask =
      is_64_bit ? UINT64_MAX : std::numeric_limits<uint32_t>::max();

  // A positive payload of 42.
  const addr_t positive = ((42ULL << shift) | tag) & pointer_mask;
  GNUstepObjCTaggedPointerClassDescriptor positive_descriptor(
      m_process_sp, g_class_addr, positive, tag, shift, PointerSize());
  ASSERT_TRUE(positive_descriptor.IsValid());
  uint64_t info_bits = 0;
  uint64_t value_bits = 0;
  ASSERT_TRUE(
      positive_descriptor.GetTaggedPointerInfo(&info_bits, &value_bits));
  EXPECT_EQ(info_bits, tag);
  EXPECT_EQ(value_bits, 42u);

  // A negative payload of -42 encoded in the target's pointer width.
  const addr_t negative =
      ((static_cast<uint64_t>(-42LL) << shift) | tag) & pointer_mask;
  GNUstepObjCTaggedPointerClassDescriptor negative_descriptor(
      m_process_sp, g_class_addr, negative, tag, shift, PointerSize());
  int64_t signed_value = 0;
  ASSERT_TRUE(negative_descriptor.GetTaggedPointerInfoSigned(&info_bits,
                                                             &signed_value));
  EXPECT_EQ(info_bits, tag);
  EXPECT_EQ(signed_value, -42);
}

INSTANTIATE_TEST_SUITE_P(
    DataModels, GNUstepClassDescriptorTest,
    ::testing::Values(DataModel{"x86_64-pc-linux", 8, 8},
                      // Windows is LLP64: pointers are 64 bits but long stays
                      // 32, moving every field after the class name.
                      DataModel{"x86_64-pc-windows-msvc", 8, 4},
                      DataModel{"i386-pc-linux", 4, 4}),
    [](const ::testing::TestParamInfo<DataModel> &info) {
      std::string name = info.param.triple;
      for (char &c : name)
        if (!std::isalnum(static_cast<unsigned char>(c)))
          c = '_';
      return name;
    });

} // namespace
