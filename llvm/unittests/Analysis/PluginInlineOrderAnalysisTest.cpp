#include "llvm/Analysis/CallGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include "llvm/Analysis/InlineOrder.h"

namespace llvm {

namespace {

void anchor() {}

std::string libPath(const std::string Name = "InlineOrderPlugin") {
  const auto &Argvs = testing::internal::GetArgvs();
  const char *Argv0 =
      Argvs.size() > 0 ? Argvs[0].c_str() : "PluginInlineOrderAnalysisTest";
  void *Ptr = (void *)(intptr_t)anchor;
  std::string Path = sys::fs::getMainExecutable(Argv0, Ptr);
  llvm::SmallString<256> Buf{sys::path::parent_path(Path)};
  sys::path::append(Buf, (Name + LLVM_PLUGIN_EXT).c_str());
  return std::string(Buf.str());
}

struct CompilerInstance {
  LLVMContext Ctx;
  ModulePassManager MPM;
  InlineParams IP;

  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  SMDiagnostic Error;

  // Connect the plugin to our compiler instance.
  void setupPlugin() {
    auto PluginPath = libPath();
    ASSERT_NE("", PluginPath);
    Expected<PassPlugin> Plugin = PassPlugin::Load(PluginPath);
    ASSERT_TRUE(!!Plugin) << "Plugin path: " << PluginPath;
    Plugin->registerPassBuilderCallbacks(PB);
  }

  CompilerInstance() {
    IP = getInlineParams(3, 0);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    MPM.addPass(ModuleInlinerPass(IP, InliningAdvisorMode::Default,
                                  ThinOrFullLTOPhase::None));
  }

  std::string Output;
  std::unique_ptr<Module> OutputM;

  // Run with the dynamic inline order.
  auto run(StringRef IR) {
    OutputM = parseAssemblyString(IR, Error, Ctx);
    MPM.run(*OutputM, MAM);
    ASSERT_TRUE(OutputM);
    Output.clear();
    raw_string_ostream OStream{Output};
    OutputM->print(OStream, nullptr);
    ASSERT_TRUE(true);
  }
};

StringRef TestIRS[] = {
    // Simple 3 function inline case.
    R"(
define void @f1() {
  call void @foo()
  ret void
}
define void @foo() {
  call void @f3()
  ret void
}
define void @f3() {
  ret void
}
  )",
    // Test that has 5 functions of which 2 are recursive.
    R"(
define void @f1() {
  call void @foo()
  ret void
}
define void @f2() {
  call void @foo()
  ret void
}
define void @foo() {
  call void @f4()
  call void @f5()
  ret void
}
define void @f4() {
  ret void
}
define void @f5() {
  call void @foo()
  ret void
}
  )",
    // Test with 2 mutually recursive functions and 1 function with a loop.
    R"(
define void @f1() {
  call void @f2()
  ret void
}
define void @f2() {
  call void @foo()
  ret void
}
define void @foo() {
  call void @f1()
  ret void
}
define void @f4() {
  br label %loop
loop:
  call void @f5()
  br label %loop
}
define void @f5() {
  ret void
}
  )",
    // Test that has a function that computes fibonacci in a loop, one in a
    // recursive manner, and one that calls both and compares them.
    R"(
define i32 @fib_loop(i32 %n){
    %curr = alloca i32
    %last = alloca i32
    %i = alloca i32
    store i32 1, i32* %curr
    store i32 1, i32* %last
    store i32 2, i32* %i
    br label %loop_cond
  loop_cond:
    %i_val = load i32, i32* %i
    %cmp = icmp slt i32 %i_val, %n
    br i1 %cmp, label %loop_body, label %loop_end
  loop_body:
    %curr_val = load i32, i32* %curr
    %last_val = load i32, i32* %last
    %add = add i32 %curr_val, %last_val
    store i32 %add, i32* %last
    store i32 %curr_val, i32* %curr
    %i_val2 = load i32, i32* %i
    %add2 = add i32 %i_val2, 1
    store i32 %add2, i32* %i
    br label %loop_cond
  loop_end:
    %curr_val3 = load i32, i32* %curr
    ret i32 %curr_val3
}

define i32 @foo(i32 %n){
    %cmp = icmp eq i32 %n, 0
    %cmp2 = icmp eq i32 %n, 1
    %or = or i1 %cmp, %cmp2
    br i1 %or, label %if_true, label %if_false
  if_true:
    ret i32 1
  if_false:
    %sub = sub i32 %n, 1
    %call = call i32 @foo(i32 %sub)
    %sub2 = sub i32 %n, 2
    %call2 = call i32 @foo(i32 %sub2)
    %add = add i32 %call, %call2
    ret i32 %add
}

define i32 @fib_check(){
    %correct = alloca i32
    %i = alloca i32
    store i32 1, i32* %correct
    store i32 0, i32* %i
    br label %loop_cond
  loop_cond:
    %i_val = load i32, i32* %i
    %cmp = icmp slt i32 %i_val, 10
    br i1 %cmp, label %loop_body, label %loop_end
  loop_body:
    %i_val2 = load i32, i32* %i
    %call = call i32 @fib_loop(i32 %i_val2)
    %i_val3 = load i32, i32* %i
    %call2 = call i32 @foo(i32 %i_val3)
    %cmp2 = icmp ne i32 %call, %call2
    br i1 %cmp2, label %if_true, label %if_false
  if_true:
    store i32 0, i32* %correct
    br label %if_end
  if_false:
    br label %if_end
  if_end:
    %i_val4 = load i32, i32* %i
    %add = add i32 %i_val4, 1
    store i32 %add, i32* %i
    br label %loop_cond
  loop_end:
    %correct_val = load i32, i32* %correct
    ret i32 %correct_val
}
  )"};

} // namespace

// Check that the behaviour of a custom inline order is correct.
// The custom order drops any functions named "foo" so all tests
// should contain at least one function named foo.
TEST(PluginInlineOrderTest, NoInlineFoo) {
#if !defined(LLVM_ENABLE_PLUGINS)
  // Skip the test if plugins are disabled.
  GTEST_SKIP();
#endif
  CompilerInstance CI{};
  CI.setupPlugin();

  for (StringRef IR : TestIRS) {
    bool FoundFoo = false;
    CI.run(IR);
    CallGraph CGraph = CallGraph(*CI.OutputM);
    for (auto &Node : CGraph) {
      for (auto &Edge : *Node.second) {
        FoundFoo |= Edge.second->getFunction()->getName() == "foo";
      }
    }
    ASSERT_TRUE(FoundFoo);
  }
}

} // namespace llvm
