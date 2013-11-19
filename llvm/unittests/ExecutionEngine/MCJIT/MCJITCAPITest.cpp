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
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Host.h"
#include "MCJITTestAPICommon.h"
#include "gtest/gtest.h"

using namespace llvm;

static bool didCallAllocateCodeSection;

static uint8_t *roundTripAllocateCodeSection(void *object, uintptr_t size,
                                             unsigned alignment,
                                             unsigned sectionID,
                                             const char *sectionName) {
  didCallAllocateCodeSection = true;
  return static_cast<SectionMemoryManager*>(object)->allocateCodeSection(
    size, alignment, sectionID, sectionName);
}

static uint8_t *roundTripAllocateDataSection(void *object, uintptr_t size,
                                             unsigned alignment,
                                             unsigned sectionID,
                                             const char *sectionName,
                                             LLVMBool isReadOnly) {
  return static_cast<SectionMemoryManager*>(object)->allocateDataSection(
    size, alignment, sectionID, sectionName, isReadOnly);
}

static LLVMBool roundTripFinalizeMemory(void *object, char **errMsg) {
  std::string errMsgString;
  bool result =
    static_cast<SectionMemoryManager*>(object)->finalizeMemory(&errMsgString);
  if (result) {
    *errMsg = LLVMCreateMessage(errMsgString.c_str());
    return 1;
  }
  return 0;
}

static void roundTripDestroy(void *object) {
  delete static_cast<SectionMemoryManager*>(object);
}

namespace {
class MCJITCAPITest : public testing::Test, public MCJITTestAPICommon {
protected:
  MCJITCAPITest() {
    // The architectures below are known to be compatible with MCJIT as they
    // are copied from test/ExecutionEngine/MCJIT/lit.local.cfg and should be
    // kept in sync.
    SupportedArchs.push_back(Triple::aarch64);
    SupportedArchs.push_back(Triple::arm);
    SupportedArchs.push_back(Triple::mips);
    SupportedArchs.push_back(Triple::x86);
    SupportedArchs.push_back(Triple::x86_64);

    // Some architectures have sub-architectures in which tests will fail, like
    // ARM. These two vectors will define if they do have sub-archs (to avoid
    // extra work for those who don't), and if so, if they are listed to work
    HasSubArchs.push_back(Triple::arm);
    SupportedSubArchs.push_back("armv6");
    SupportedSubArchs.push_back("armv7");

    // The operating systems below are known to be sufficiently incompatible
    // that they will fail the MCJIT C API tests.
    UnsupportedOSs.push_back(Triple::Cygwin);
  }
  
  virtual void SetUp() {
    didCallAllocateCodeSection = false;
    Module = 0;
    Function = 0;
    Engine = 0;
    Error = 0;
  }
  
  virtual void TearDown() {
    if (Engine)
      LLVMDisposeExecutionEngine(Engine);
    else if (Module)
      LLVMDisposeModule(Module);
  }
  
  void buildSimpleFunction() {
    Module = LLVMModuleCreateWithName("simple_module");
    
    LLVMSetTarget(Module, HostTriple.c_str());
    
    Function = LLVMAddFunction(
      Module, "simple_function", LLVMFunctionType(LLVMInt32Type(), 0, 0, 0));
    LLVMSetFunctionCallConv(Function, LLVMCCallConv);
    
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(Function, "entry");
    LLVMBuilderRef builder = LLVMCreateBuilder();
    LLVMPositionBuilderAtEnd(builder, entry);
    LLVMBuildRet(builder, LLVMConstInt(LLVMInt32Type(), 42, 0));
    
    LLVMVerifyModule(Module, LLVMAbortProcessAction, &Error);
    LLVMDisposeMessage(Error);
    
    LLVMDisposeBuilder(builder);
  }
  
  void buildMCJITOptions() {
    LLVMInitializeMCJITCompilerOptions(&Options, sizeof(Options));
    Options.OptLevel = 2;
    
    // Just ensure that this field still exists.
    Options.NoFramePointerElim = false;
  }
  
  void useRoundTripSectionMemoryManager() {
    Options.MCJMM = LLVMCreateSimpleMCJITMemoryManager(
      new SectionMemoryManager(),
      roundTripAllocateCodeSection,
      roundTripAllocateDataSection,
      roundTripFinalizeMemory,
      roundTripDestroy);
  }
  
  void buildMCJITEngine() {
    ASSERT_EQ(
      0, LLVMCreateMCJITCompilerForModule(&Engine, Module, &Options,
                                          sizeof(Options), &Error));
  }
  
  void buildAndRunPasses() {
    LLVMPassManagerRef pass = LLVMCreatePassManager();
    LLVMAddTargetData(LLVMGetExecutionEngineTargetData(Engine), pass);
    LLVMAddConstantPropagationPass(pass);
    LLVMAddInstructionCombiningPass(pass);
    LLVMRunPassManager(pass, Module);
    LLVMDisposePassManager(pass);
  }
  
  LLVMModuleRef Module;
  LLVMValueRef Function;
  LLVMMCJITCompilerOptions Options;
  LLVMExecutionEngineRef Engine;
  char *Error;
};
} // end anonymous namespace

TEST_F(MCJITCAPITest, simple_function) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  buildSimpleFunction();
  buildMCJITOptions();
  buildMCJITEngine();
  buildAndRunPasses();
  
  union {
    void *raw;
    int (*usable)();
  } functionPointer;
  functionPointer.raw = LLVMGetPointerToGlobal(Engine, Function);
  
  EXPECT_EQ(42, functionPointer.usable());
}

TEST_F(MCJITCAPITest, custom_memory_manager) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  buildSimpleFunction();
  buildMCJITOptions();
  useRoundTripSectionMemoryManager();
  buildMCJITEngine();
  buildAndRunPasses();
  
  union {
    void *raw;
    int (*usable)();
  } functionPointer;
  functionPointer.raw = LLVMGetPointerToGlobal(Engine, Function);
  
  EXPECT_EQ(42, functionPointer.usable());
  EXPECT_TRUE(didCallAllocateCodeSection);
}
