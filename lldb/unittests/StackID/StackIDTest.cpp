
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
  MockProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  size_t DoReadMemory(addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    return 0;
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
