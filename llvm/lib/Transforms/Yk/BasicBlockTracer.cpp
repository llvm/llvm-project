#include "llvm/Transforms/Yk/BasicBlockTracer.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Yk/ModuleClone.h"

#define DEBUG_TYPE "yk-basicblock-tracer-pass"

using namespace llvm;

namespace llvm {
void initializeYkBasicBlockTracerPass(PassRegistry &);
} // namespace llvm

namespace {
struct YkBasicBlockTracer : public ModulePass {
  static char ID;

  YkBasicBlockTracer() : ModulePass(ID) {
    initializeYkBasicBlockTracerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    LLVMContext &Context = M.getContext();
    // Create externally linked function declaration:
    //   void __yk_trace_basicblock(int functionIndex, int blockIndex)
    Type *ReturnType = Type::getVoidTy(Context);
    Type *FunctionIndexArgType = Type::getInt32Ty(Context);
    Type *BlockIndexArgType = Type::getInt32Ty(Context);

    FunctionType *FType = FunctionType::get(
        ReturnType, {FunctionIndexArgType, BlockIndexArgType}, false);

    Function *TraceFunc = Function::Create(
        FType, GlobalVariable::ExternalLinkage, YK_TRACE_FUNCTION, M);

    Function *DummyTraceFunc = Function::Create(
        FType, GlobalVariable::ExternalLinkage, YK_TRACE_FUNCTION_DUMMY, M);

    IRBuilder<> builder(Context);
    uint32_t FunctionIndex = 0;
    for (auto &F : M) {
      uint32_t BlockIndex = 0;
      for (auto &BB : F) {
        builder.SetInsertPoint(&*BB.getFirstInsertionPt());

        if (F.getName().startswith(YK_UNOPT_PREFIX)) {
          // Add dummy tracing calls to unoptimised functions
          // TODO: remove these calls once we get rid of the error:
          // #0  core::sync::atomic::AtomicUsize::fetch_sub (self=0xfffffffffffffff0) at /home/pd/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/sync/atomic.rs:2720
          // #1  alloc::sync::{impl#37}::drop<lock_api::mutex::Mutex<parking_lot::raw_mutex::RawMutex, ykrt::location::HotLocation>, alloc::alloc::Global> (self=0x7fffffffd2d0) at /home/pd/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/alloc/src/sync.rs:2529
          // #2  0x00007ffff79fa5bb in core::ptr::drop_in_place<alloc::sync::Arc<lock_api::mutex::Mutex<parking_lot::raw_mutex::RawMutex, ykrt::location::HotLocation>, alloc::alloc::Global>> () at /home/pd/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/mod.rs:521
          // #3  0x00007ffff79372de in core::mem::drop<alloc::sync::Arc<lock_api::mutex::Mutex<parking_lot::raw_mutex::RawMutex, ykrt::location::HotLocation>, alloc::alloc::Global>> (_x=...) at /home/pd/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/mem/mod.rs:942
          // #4  0x00007ffff7a06870 in ykrt::location::{impl#2}::drop (self=0x7fffffffd328) at ykrt/src/location.rs:211
          // #5  0x00007ffff792078b in core::ptr::drop_in_place<ykrt::location::Location> () at /home/pd/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/ptr/mod.rs:521
          // #6  0x00007ffff79207ad in core::mem::drop<ykrt::location::Location> (_x=...) at /home/pd/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/mem/mod.rs:942
          // #7  0x00007ffff79201c5 in ykcapi::yk_location_drop (loc=...) at ykcapi/src/lib.rs:155
          // #8  0x0000000000202b66 in main () at /home/pd/yk-fork/tests/c/simple3.c:36
          // #9  0x00007ffff704624a in __libc_start_call_main (main=main@entry=0x202790 <main>, argc=argc@entry=1, argv=0x7fffffffde98, argv@entry=0x7fffffffdea8) at ../sysdeps/nptl/libc_start_call_main.h:58
          // #10 0x00007ffff7046305 in __libc_start_main_impl (main=0x202790 <main>, argc=1, argv=0x7fffffffdea8, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7fffffffde88) at ../csu/libc-start.c:360
          // #11 0x0000000000202661 in _start ()
          // if (F.getName().startswith(YK_CLONE_PREFIX)) {
          //   continue;
          // }
          builder.CreateCall(TraceFunc, {builder.getInt32(FunctionIndex),
                                         builder.getInt32(BlockIndex)});
        } else {
          // Add tracing calls to unoptimised functions
          builder.CreateCall(DummyTraceFunc, {builder.getInt32(FunctionIndex),
                                              builder.getInt32(BlockIndex)});
        }
        assert(BlockIndex != UINT32_MAX &&
               "Expected BlockIndex to not overflow");
        BlockIndex++;
      }
      assert(FunctionIndex != UINT32_MAX &&
             "Expected FunctionIndex to not overflow");
      FunctionIndex++;
    }
    return true;
  }
   void getAnalysisUsage(AnalysisUsage &AU) const override {
      // AU.setPreservesCFG();
      AU.setPreservesAll(); //if appropriate
    }
};
} // namespace

char YkBasicBlockTracer::ID = 0;

INITIALIZE_PASS(YkBasicBlockTracer, DEBUG_TYPE, "yk basicblock tracer", false,
                false)

ModulePass *llvm::createYkBasicBlockTracerPass() {
  return new YkBasicBlockTracer();
}
