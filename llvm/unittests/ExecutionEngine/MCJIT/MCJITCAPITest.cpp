//===- MCJITTest.cpp - Unit tests for the MCJIT -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This test suite verifies basic MCJIT functionality when invoked form the C
// API.
//
//===----------------------------------------------------------------------===//

#include "MCJITTestAPICommon.h"
#include "llvm-c/Analysis.h"
#include "llvm-c/Core.h"
#include "llvm-c/ExecutionEngine.h"
#include "llvm-c/Target.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "gtest/gtest.h"

using namespace llvm;

static bool didCallAllocateCodeSection;
static bool didAllocateCompactUnwindSection;
static bool didCallYield;

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
  if (!strcmp(sectionName, "__compact_unwind"))
    didAllocateCompactUnwindSection = true;
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

static void yield(LLVMContextRef, void *) {
  didCallYield = true;
}

namespace {

// memory manager to test reserve allocation space callback
class TestReserveAllocationSpaceMemoryManager: public SectionMemoryManager {
public:
  uintptr_t ReservedCodeSize;
  uintptr_t UsedCodeSize;
  uintptr_t ReservedDataSizeRO;
  uintptr_t UsedDataSizeRO;
  uintptr_t ReservedDataSizeRW;
  uintptr_t UsedDataSizeRW;
  
  TestReserveAllocationSpaceMemoryManager() : 
    ReservedCodeSize(0), UsedCodeSize(0), ReservedDataSizeRO(0), 
    UsedDataSizeRO(0), ReservedDataSizeRW(0), UsedDataSizeRW(0) {    
  }

  bool needsToReserveAllocationSpace() override { return true; }

  void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                              uintptr_t DataSizeRO, uint32_t RODataAlign,
                              uintptr_t DataSizeRW,
                              uint32_t RWDataAlign) override {
    ReservedCodeSize = CodeSize;
    ReservedDataSizeRO = DataSizeRO;
    ReservedDataSizeRW = DataSizeRW;
  }

  void useSpace(uintptr_t* UsedSize, uintptr_t Size, unsigned Alignment) {
    uintptr_t AlignedSize = (Size + Alignment - 1) / Alignment * Alignment;
    uintptr_t AlignedBegin = (*UsedSize + Alignment - 1) / Alignment * Alignment;
    *UsedSize = AlignedBegin + AlignedSize;
  }

  uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID, StringRef SectionName,
                               bool IsReadOnly) override {
    useSpace(IsReadOnly ? &UsedDataSizeRO : &UsedDataSizeRW, Size, Alignment);
    return SectionMemoryManager::allocateDataSection(Size, Alignment, 
      SectionID, SectionName, IsReadOnly);
  }

  uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                               unsigned SectionID,
                               StringRef SectionName) override {
    useSpace(&UsedCodeSize, Size, Alignment);
    return SectionMemoryManager::allocateCodeSection(Size, Alignment, 
      SectionID, SectionName);
  }
};

class MCJITCAPITest : public testing::Test, public MCJITTestAPICommon {
protected:
  MCJITCAPITest() {
    // The architectures below are known to be compatible with MCJIT as they
    // are copied from test/ExecutionEngine/MCJIT/lit.local.cfg and should be
    // kept in sync.
    SupportedArchs.push_back(Triple::aarch64);
    SupportedArchs.push_back(Triple::arm);
    SupportedArchs.push_back(Triple::mips);
    SupportedArchs.push_back(Triple::mips64);
    SupportedArchs.push_back(Triple::mips64el);
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
    UnsupportedEnvironments.push_back(Triple::Cygnus);
  }

  void SetUp() override {
    didCallAllocateCodeSection = false;
    didAllocateCompactUnwindSection = false;
    didCallYield = false;
    Module = nullptr;
    Function = nullptr;
    Engine = nullptr;
    Error = nullptr;
  }

  void TearDown() override {
    if (Engine)
      LLVMDisposeExecutionEngine(Engine);
    else if (Module)
      LLVMDisposeModule(Module);
  }
  
  void buildSimpleFunction() {
    Module = LLVMModuleCreateWithName("simple_module");
    
    LLVMSetTarget(Module, HostTriple.c_str());
    
    Function = LLVMAddFunction(Module, "simple_function",
                               LLVMFunctionType(LLVMInt32Type(), nullptr,0, 0));
    LLVMSetFunctionCallConv(Function, LLVMCCallConv);
    
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(Function, "entry");
    LLVMBuilderRef builder = LLVMCreateBuilder();
    LLVMPositionBuilderAtEnd(builder, entry);
    LLVMBuildRet(builder, LLVMConstInt(LLVMInt32Type(), 42, 0));
    
    LLVMVerifyModule(Module, LLVMAbortProcessAction, &Error);
    LLVMDisposeMessage(Error);
    
    LLVMDisposeBuilder(builder);
  }
  
  void buildFunctionThatUsesStackmap() {
    Module = LLVMModuleCreateWithName("simple_module");
    
    LLVMSetTarget(Module, HostTriple.c_str());
    
    LLVMTypeRef stackmapParamTypes[] = { LLVMInt64Type(), LLVMInt32Type() };
    LLVMTypeRef stackmapTy =
        LLVMFunctionType(LLVMVoidType(), stackmapParamTypes, 2, 1);
    LLVMValueRef stackmap = LLVMAddFunction(
      Module, "llvm.experimental.stackmap", stackmapTy);
    LLVMSetLinkage(stackmap, LLVMExternalLinkage);
    
    Function = LLVMAddFunction(Module, "simple_function",
                              LLVMFunctionType(LLVMInt32Type(), nullptr, 0, 0));
    
    LLVMBasicBlockRef entry = LLVMAppendBasicBlock(Function, "entry");
    LLVMBuilderRef builder = LLVMCreateBuilder();
    LLVMPositionBuilderAtEnd(builder, entry);
    LLVMValueRef stackmapArgs[] = {
      LLVMConstInt(LLVMInt64Type(), 0, 0), LLVMConstInt(LLVMInt32Type(), 5, 0),
      LLVMConstInt(LLVMInt32Type(), 42, 0)
    };
    LLVMBuildCall2(builder, stackmapTy, stackmap, stackmapArgs, 3, "");
    LLVMBuildRet(builder, LLVMConstInt(LLVMInt32Type(), 42, 0));
    
    LLVMVerifyModule(Module, LLVMAbortProcessAction, &Error);
    LLVMDisposeMessage(Error);
    
    LLVMDisposeBuilder(builder);
  }
  
  void buildModuleWithCodeAndData() {
    Module = LLVMModuleCreateWithName("simple_module");
    
    LLVMSetTarget(Module, HostTriple.c_str());
    
    // build a global int32 variable initialized to 42.
    LLVMValueRef GlobalVar = LLVMAddGlobal(Module, LLVMInt32Type(), "intVal");    
    LLVMSetInitializer(GlobalVar, LLVMConstInt(LLVMInt32Type(), 42, 0));
    
    {
        Function = LLVMAddFunction(Module, "getGlobal",
                              LLVMFunctionType(LLVMInt32Type(), nullptr, 0, 0));
        LLVMSetFunctionCallConv(Function, LLVMCCallConv);
        
        LLVMBasicBlockRef Entry = LLVMAppendBasicBlock(Function, "entry");
        LLVMBuilderRef Builder = LLVMCreateBuilder();
        LLVMPositionBuilderAtEnd(Builder, Entry);
        
        LLVMValueRef IntVal =
            LLVMBuildLoad2(Builder, LLVMInt32Type(), GlobalVar, "intVal");
        LLVMBuildRet(Builder, IntVal);
        
        LLVMVerifyModule(Module, LLVMAbortProcessAction, &Error);
        LLVMDisposeMessage(Error);
        
        LLVMDisposeBuilder(Builder);
    }
    
    {
        LLVMTypeRef ParamTypes[] = { LLVMInt32Type() };
        Function2 = LLVMAddFunction(
          Module, "setGlobal", LLVMFunctionType(LLVMVoidType(), ParamTypes, 1, 0));
        LLVMSetFunctionCallConv(Function2, LLVMCCallConv);
        
        LLVMBasicBlockRef Entry = LLVMAppendBasicBlock(Function2, "entry");
        LLVMBuilderRef Builder = LLVMCreateBuilder();
        LLVMPositionBuilderAtEnd(Builder, Entry);
        
        LLVMValueRef Arg = LLVMGetParam(Function2, 0);
        LLVMBuildStore(Builder, Arg, GlobalVar);
        LLVMBuildRetVoid(Builder);
        
        LLVMVerifyModule(Module, LLVMAbortProcessAction, &Error);
        LLVMDisposeMessage(Error);
        
        LLVMDisposeBuilder(Builder);
    }
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
    LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
    if (LLVMErrorRef E =
            LLVMRunPasses(Module, "instcombine", nullptr, Options)) {
        char *Msg = LLVMGetErrorMessage(E);
        LLVMConsumeError(E);
        LLVMDisposePassBuilderOptions(Options);
        FAIL() << "Failed to run passes: " << Msg;
    }

    LLVMDisposePassBuilderOptions(Options);
  }
  
  void buildAndRunOptPasses() {
    LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
    if (LLVMErrorRef E =
            LLVMRunPasses(Module, "default<O2>", nullptr, Options)) {
        char *Msg = LLVMGetErrorMessage(E);
        LLVMConsumeError(E);
        LLVMDisposePassBuilderOptions(Options);
        FAIL() << "Failed to run passes: " << Msg;
    }

    LLVMDisposePassBuilderOptions(Options);
  }
  
  LLVMModuleRef Module;
  LLVMValueRef Function;
  LLVMValueRef Function2;
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

  auto *functionPointer = reinterpret_cast<int (*)()>(
      reinterpret_cast<uintptr_t>(LLVMGetPointerToGlobal(Engine, Function)));

  EXPECT_EQ(42, functionPointer());
}

TEST_F(MCJITCAPITest, gva) {
  SKIP_UNSUPPORTED_PLATFORM;

  Module = LLVMModuleCreateWithName("simple_module");
  LLVMSetTarget(Module, HostTriple.c_str());
  LLVMValueRef GlobalVar = LLVMAddGlobal(Module, LLVMInt32Type(), "simple_value");
  LLVMSetInitializer(GlobalVar, LLVMConstInt(LLVMInt32Type(), 42, 0));

  buildMCJITOptions();
  buildMCJITEngine();
  buildAndRunPasses();

  uint64_t raw = LLVMGetGlobalValueAddress(Engine, "simple_value");
  int32_t *usable  = (int32_t *) raw;

  EXPECT_EQ(42, *usable);
}

TEST_F(MCJITCAPITest, gfa) {
  SKIP_UNSUPPORTED_PLATFORM;

  buildSimpleFunction();
  buildMCJITOptions();
  buildMCJITEngine();
  buildAndRunPasses();

  uint64_t raw = LLVMGetFunctionAddress(Engine, "simple_function");
  int (*usable)() = (int (*)()) raw;

  EXPECT_EQ(42, usable());
}

TEST_F(MCJITCAPITest, custom_memory_manager) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  buildSimpleFunction();
  buildMCJITOptions();
  useRoundTripSectionMemoryManager();
  buildMCJITEngine();
  buildAndRunPasses();

  auto *functionPointer = reinterpret_cast<int (*)()>(
      reinterpret_cast<uintptr_t>(LLVMGetPointerToGlobal(Engine, Function)));

  EXPECT_EQ(42, functionPointer());
  EXPECT_TRUE(didCallAllocateCodeSection);
}

TEST_F(MCJITCAPITest, stackmap_creates_compact_unwind_on_darwin) {
  SKIP_UNSUPPORTED_PLATFORM;
  
  // This test is also not supported on non-x86 platforms.
  if (Triple(HostTriple).getArch() != Triple::x86_64)
    return;
  
  buildFunctionThatUsesStackmap();
  buildMCJITOptions();
  useRoundTripSectionMemoryManager();
  buildMCJITEngine();
  buildAndRunOptPasses();

  auto *functionPointer = reinterpret_cast<int (*)()>(
      reinterpret_cast<uintptr_t>(LLVMGetPointerToGlobal(Engine, Function)));

  EXPECT_EQ(42, functionPointer());
  EXPECT_TRUE(didCallAllocateCodeSection);
  
  // Up to this point, the test is specific only to X86-64. But this next
  // expectation is only valid on Darwin because it assumes that unwind
  // data is made available only through compact_unwind. It would be
  // worthwhile to extend this to handle non-Darwin platforms, in which
  // case you'd want to look for an eh_frame or something.
  //
  // FIXME: Currently, MCJIT relies on a configure-time check to determine which
  // sections to emit. The JIT client should have runtime control over this.
  EXPECT_TRUE(
    Triple(HostTriple).getOS() != Triple::Darwin ||
    Triple(HostTriple).isMacOSXVersionLT(10, 7) ||
    didAllocateCompactUnwindSection);
}

#if defined(__APPLE__) && defined(__aarch64__)
// FIXME: Figure out why this fails on mac/arm, PR46647
#define MAYBE_reserve_allocation_space DISABLED_reserve_allocation_space
#else
#define MAYBE_reserve_allocation_space reserve_allocation_space
#endif
TEST_F(MCJITCAPITest, MAYBE_reserve_allocation_space) {
  SKIP_UNSUPPORTED_PLATFORM;

  TestReserveAllocationSpaceMemoryManager* MM = new TestReserveAllocationSpaceMemoryManager();
  
  buildModuleWithCodeAndData();
  buildMCJITOptions();
  Options.MCJMM = wrap(MM);
  buildMCJITEngine();
  buildAndRunPasses();

  auto GetGlobalFct = reinterpret_cast<int (*)()>(
      reinterpret_cast<uintptr_t>(LLVMGetPointerToGlobal(Engine, Function)));

  auto SetGlobalFct = reinterpret_cast<void (*)(int)>(
      reinterpret_cast<uintptr_t>(LLVMGetPointerToGlobal(Engine, Function2)));

  SetGlobalFct(789);
  EXPECT_EQ(789, GetGlobalFct());
  EXPECT_LE(MM->UsedCodeSize, MM->ReservedCodeSize);
  EXPECT_LE(MM->UsedDataSizeRO, MM->ReservedDataSizeRO);
  EXPECT_LE(MM->UsedDataSizeRW, MM->ReservedDataSizeRW);
  EXPECT_TRUE(MM->UsedCodeSize > 0); 
  EXPECT_TRUE(MM->UsedDataSizeRW > 0);
}

TEST_F(MCJITCAPITest, yield) {
  SKIP_UNSUPPORTED_PLATFORM;

  buildSimpleFunction();
  buildMCJITOptions();
  buildMCJITEngine();
  LLVMContextRef C = LLVMGetGlobalContext();
  LLVMContextSetYieldCallback(C, yield, nullptr);
  buildAndRunPasses();

  auto *functionPointer = reinterpret_cast<int (*)()>(
      reinterpret_cast<uintptr_t>(LLVMGetPointerToGlobal(Engine, Function)));

  EXPECT_EQ(42, functionPointer());
  EXPECT_TRUE(didCallYield);
}

static int localTestFunc() {
  return 42;
}

TEST_F(MCJITCAPITest, addGlobalMapping) {
  SKIP_UNSUPPORTED_PLATFORM;

  Module = LLVMModuleCreateWithName("testModule");
  LLVMSetTarget(Module, HostTriple.c_str());
  LLVMTypeRef FunctionType = LLVMFunctionType(LLVMInt32Type(), nullptr, 0, 0);
  LLVMValueRef MappedFn = LLVMAddFunction(Module, "mapped_fn", FunctionType);

  Function = LLVMAddFunction(Module, "test_fn", FunctionType);
  LLVMBasicBlockRef Entry = LLVMAppendBasicBlock(Function, "");
  LLVMBuilderRef Builder = LLVMCreateBuilder();
  LLVMPositionBuilderAtEnd(Builder, Entry);
  LLVMValueRef RetVal =
      LLVMBuildCall2(Builder, FunctionType, MappedFn, nullptr, 0, "");
  LLVMBuildRet(Builder, RetVal);
  LLVMDisposeBuilder(Builder);

  LLVMVerifyModule(Module, LLVMAbortProcessAction, &Error);
  LLVMDisposeMessage(Error);

  buildMCJITOptions();
  buildMCJITEngine();

  LLVMAddGlobalMapping(
      Engine, MappedFn,
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(&localTestFunc)));

  buildAndRunPasses();

  uint64_t raw = LLVMGetFunctionAddress(Engine, "test_fn");
  int (*usable)() = (int (*)()) raw;

  EXPECT_EQ(42, usable());
}
