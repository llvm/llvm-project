
#include "lldb/Target/StackID.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

static std::once_flag initialize_flag;

// Initialize the bare minimum to enable defining a mock Process class.
class StackIDTest : public ::testing::Test {
public:
  void SetUp() override {
    std::call_once(initialize_flag, []() {
      HostInfo::Initialize();
      PlatformMacOSX::Initialize();
      FileSystem::Initialize();
    });
    ArchSpec arch("x86_64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    m_debugger_sp->GetTargetList().CreateTarget(*m_debugger_sp, "", arch,
                                                eLoadDependentsNo,
                                                m_platform_sp, m_target_sp);
    ASSERT_TRUE(m_target_sp);
    ASSERT_TRUE(m_target_sp->GetArchitecture().IsValid());
    ASSERT_TRUE(m_platform_sp);
  }

  PlatformSP m_platform_sp;
  TargetSP m_target_sp;
  DebuggerSP m_debugger_sp;
};

struct MockProcess : Process {
  llvm::DenseMap<addr_t, addr_t> memory_map;

  MockProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  MockProcess(TargetSP target_sp, ListenerSP listener_sp,
              llvm::DenseMap<addr_t, addr_t> &&memory_map)
      : Process(target_sp, listener_sp), memory_map(memory_map) {}
  size_t DoReadMemory(addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    assert(memory_map.contains(vm_addr));
    assert(size == sizeof(addr_t));
    *reinterpret_cast<addr_t *>(buf) = memory_map[vm_addr];
    return sizeof(addr_t);
  }
  size_t ReadMemory(addr_t addr, void *buf, size_t size,
                    Status &status) override {
    return DoReadMemory(addr, buf, size, status);
  }
  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return Status(); }
  llvm::StringRef GetPluginName() override { return ""; }
  void RefreshStateAfterStop() override {}
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
};

enum OnStack { Yes, No };
/// Helper class to enable testing StackID::IsYounger.
struct MockStackID : StackID {
  MockStackID(addr_t cfa, OnStack on_stack) : StackID() {
    SetPC(0);
    SetCFA(cfa);
    m_cfa_on_stack = on_stack == OnStack::Yes ? LazyBool::eLazyBoolYes
                                              : LazyBool::eLazyBoolNo;
  }
};

TEST_F(StackIDTest, StackStackCFAComparison) {
  auto process = MockProcess(m_target_sp, Listener::MakeListener("dummy"));

  MockStackID small_cfa_on_stack(/*cfa*/ 10, OnStack::Yes);
  MockStackID big_cfa_on_stack(/*cfa*/ 100, OnStack::Yes);

  EXPECT_TRUE(
      StackID::IsYounger(small_cfa_on_stack, big_cfa_on_stack, process));
  EXPECT_FALSE(
      StackID::IsYounger(big_cfa_on_stack, small_cfa_on_stack, process));
}

TEST_F(StackIDTest, StackHeapCFAComparison) {
  auto process = MockProcess(m_target_sp, Listener::MakeListener("dummy"));

  MockStackID cfa_on_stack(/*cfa*/ 100, OnStack::Yes);
  MockStackID cfa_on_heap(/*cfa*/ 10, OnStack::No);

  EXPECT_TRUE(StackID::IsYounger(cfa_on_stack, cfa_on_heap, process));
  EXPECT_FALSE(StackID::IsYounger(cfa_on_heap, cfa_on_stack, process));
}

TEST_F(StackIDTest, HeapHeapCFAComparison) {
  // Create a mock async continuation chain:
  // 100 -> 108 -> 116 -> 0
  // This should be read as:
  // "Async context whose address is 100 has a continuation context whose
  // address is 108", etc.
  llvm::DenseMap<addr_t, addr_t> memory_map;
  memory_map[100] = 108;
  memory_map[108] = 116;
  memory_map[116] = 0;
  auto process = MockProcess(m_target_sp, Listener::MakeListener("dummy"),
                             std::move(memory_map));

  MockStackID oldest_cfa(/*cfa*/ 116, OnStack::No);
  MockStackID middle_cfa(/*cfa*/ 108, OnStack::No);
  MockStackID youngest_cfa(/*cfa*/ 100, OnStack::No);

  EXPECT_TRUE(StackID::IsYounger(youngest_cfa, oldest_cfa, process));
  EXPECT_FALSE(StackID::IsYounger(oldest_cfa, youngest_cfa, process));

  EXPECT_TRUE(StackID::IsYounger(youngest_cfa, middle_cfa, process));
  EXPECT_FALSE(StackID::IsYounger(middle_cfa, youngest_cfa, process));

  EXPECT_TRUE(StackID::IsYounger(middle_cfa, oldest_cfa, process));
  EXPECT_FALSE(StackID::IsYounger(oldest_cfa, middle_cfa, process));
}

TEST_F(StackIDTest, HeapHeapCFAComparisonDecreasing) {
  // Create a mock async continuation chain:
  // 100 -> 90 -> 80 -> 0
  // This should be read as:
  // "Async context whose address is 100 has a continuation context whose
  // address is 90", etc.
  llvm::DenseMap<addr_t, addr_t> memory_map;
  memory_map[100] = 90;
  memory_map[90] = 80;
  memory_map[80] = 0;
  auto process = MockProcess(m_target_sp, Listener::MakeListener("dummy"),
                             std::move(memory_map));

  MockStackID oldest_cfa(/*cfa*/ 80, OnStack::No);
  MockStackID middle_cfa(/*cfa*/ 90, OnStack::No);
  MockStackID youngest_cfa(/*cfa*/ 100, OnStack::No);

  EXPECT_TRUE(StackID::IsYounger(youngest_cfa, oldest_cfa, process));
  EXPECT_FALSE(StackID::IsYounger(oldest_cfa, youngest_cfa, process));

  EXPECT_TRUE(StackID::IsYounger(youngest_cfa, middle_cfa, process));
  EXPECT_FALSE(StackID::IsYounger(middle_cfa, youngest_cfa, process));

  EXPECT_TRUE(StackID::IsYounger(middle_cfa, oldest_cfa, process));
  EXPECT_FALSE(StackID::IsYounger(oldest_cfa, middle_cfa, process));
}
