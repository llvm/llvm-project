#include "llvm/Analysis/TargetTransformInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#include "gtest/gtest.h"
#include <initializer_list>
#include <memory>

using namespace llvm;

namespace {

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print(__FILE__, errs());
  return Mod;
}

TEST(TargetTransformInfo, isOffsetFoldingLegal) {
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();

  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    
    target triple = "x86_64-unknown-linux-gnu"
    @Base1 = dso_local constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @Base1_foo, ptr @Base1_bar] }
    @Base2 = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @Base1_foo, ptr @Base1_bar] }
    
    define void @Base1_bar(ptr %this) {
      ret void
    }

    declare i32 @Base1_foo(ptr)
  )");

  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(M->getTargetTriple(), Error);
  std::unique_ptr<TargetMachine> TM(T->createTargetMachine(
      M->getTargetTriple(), "generic", "", TargetOptions(), std::nullopt,
      std::nullopt, CodeGenOptLevel::Default));
  ASSERT_FALSE(TM->isPositionIndependent());

  Function *Func = M->getFunction("Base1_bar");

  TargetTransformInfo TTI = TM->getTargetTransformInfo(*Func);

  // Base1 is dso_local.
  EXPECT_TRUE(TTI.isOffsetFoldingLegal(M->getNamedValue("Base1")));

  // Base2 is not dso_local.
  EXPECT_FALSE(TTI.isOffsetFoldingLegal(M->getNamedValue("Base2")));
}
} // namespace
