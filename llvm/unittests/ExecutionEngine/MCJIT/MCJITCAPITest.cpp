//===- MCJITTest.cpp - Unit tests for the MCJIT ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This test suite verifies basic MCJIT functionality when invoked form the C
// API.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Analysis.h"
#include "llvm-c/Core.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/Target.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm/Support/Host.h"
#include "MCJITTestAPICommon.h"
#include "gtest/gtest.h"

using namespace llvm;

class MCJITCAPITest : public testing::Test, public MCJITTestAPICommon {
protected:
  MCJITCAPITest() {
    // The architectures below are known to be compatible with MCJIT as they
    // are copied from test/ExecutionEngine/MCJIT/lit.local.cfg and should be
    // kept in sync.
    SupportedArchs.push_back(Triple::arm);
    SupportedArchs.push_back(Triple::mips);
    SupportedArchs.push_back(Triple::x86);
    SupportedArchs.push_back(Triple::x86_64);

    // The operating systems below are known to be sufficiently incompatible
    // that they will fail the MCJIT C API tests.
    UnsupportedOSs.push_back(Triple::Cygwin);
  }
};

TEST_F(MCJITCAPITest, simple_function) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  char *error = 0;
  
  // Creates a function that returns 42, compiles it, and runs it.
  
  LLVMModuleRef module = LLVMModuleCreateWithName("simple_module");

  LLVMSetTarget(module, HostTriple.c_str());
  
  LLVMValueRef function = LLVMAddFunction(
    module, "simple_function", LLVMFunctionType(LLVMInt32Type(), 0, 0, 0));
  LLVMSetFunctionCallConv(function, LLVMCCallConv);
  
  LLVMBasicBlockRef entry = LLVMAppendBasicBlock(function, "entry");
  LLVMBuilderRef builder = LLVMCreateBuilder();
  LLVMPositionBuilderAtEnd(builder, entry);
  LLVMBuildRet(builder, LLVMConstInt(LLVMInt32Type(), 42, 0));
  
  LLVMVerifyModule(module, LLVMAbortProcessAction, &error);
  LLVMDisposeMessage(error);
  
  LLVMDisposeBuilder(builder);
  
  LLVMMCJITCompilerOptions options;
  LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
  options.OptLevel = 2;
  
  // Just ensure that this field still exists.
  options.NoFramePointerElim = false;
  
  LLVMExecutionEngineRef engine;
  ASSERT_EQ(
    0, LLVMCreateMCJITCompilerForModule(&engine, module, &options,
                                        sizeof(options), &error));
  
  LLVMPassManagerRef pass = LLVMCreatePassManager();
  LLVMAddTargetData(LLVMGetExecutionEngineTargetData(engine), pass);
  LLVMAddConstantPropagationPass(pass);
  LLVMAddInstructionCombiningPass(pass);
  LLVMRunPassManager(pass, module);
  LLVMDisposePassManager(pass);
  
  union {
    void *raw;
    int (*usable)();
  } functionPointer;
  functionPointer.raw = LLVMGetPointerToGlobal(engine, function);
  
  EXPECT_EQ(42, functionPointer.usable());
  
  LLVMDisposeExecutionEngine(engine);
}

