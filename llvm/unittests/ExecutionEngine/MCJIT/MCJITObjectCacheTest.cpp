//===- MCJITObjectCacheTest.cpp - Unit tests for MCJIT object caching -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCJITTestBase.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestObjectCache : public ObjectCache {
public:
  TestObjectCache() : DuplicateInserted(false) { }

  void notifyObjectCompiled(const Module *M, MemoryBufferRef Obj) override {
    // If we've seen this module before, note that.
    const std::string ModuleID = M->getModuleIdentifier();
    if (ObjMap.find(ModuleID) != ObjMap.end())
      DuplicateInserted = true;
    // Store a copy of the buffer in our map.
    ObjMap[ModuleID] = copyBuffer(Obj);
  }

  std::unique_ptr<MemoryBuffer> getObject(const Module *M) override {
    const MemoryBuffer* BufferFound = getObjectInternal(M);
    ModulesLookedUp.insert(M->getModuleIdentifier());
    if (!BufferFound)
      return nullptr;
    // Our test cache wants to maintain ownership of its object buffers
    // so we make a copy here for the execution engine.
    return MemoryBuffer::getMemBufferCopy(BufferFound->getBuffer());
  }

  // Test-harness-specific functions
  bool wereDuplicatesInserted() { return DuplicateInserted; }

  bool wasModuleLookedUp(const Module *M) {
    return ModulesLookedUp.find(M->getModuleIdentifier())
                                      != ModulesLookedUp.end();
  }

  const MemoryBuffer* getObjectInternal(const Module* M) {
    // Look for the module in our map.
    const std::string ModuleID = M->getModuleIdentifier();
    StringMap<const MemoryBuffer *>::iterator it = ObjMap.find(ModuleID);
    if (it == ObjMap.end())
      return nullptr;
    return it->second;
  }

private:
  MemoryBuffer *copyBuffer(MemoryBufferRef Buf) {
    // Create a local copy of the buffer.
    std::unique_ptr<MemoryBuffer> NewBuffer =
        MemoryBuffer::getMemBufferCopy(Buf.getBuffer());
    MemoryBuffer *Ret = NewBuffer.get();
    AllocatedBuffers.push_back(std::move(NewBuffer));
    return Ret;
  }

  StringMap<const MemoryBuffer *> ObjMap;
  StringSet<>                     ModulesLookedUp;
  SmallVector<std::unique_ptr<MemoryBuffer>, 2> AllocatedBuffers;
  bool                            DuplicateInserted;
};

class MCJITObjectCacheTest : public testing::Test, public MCJITTestBase {
protected:
  enum {
    OriginalRC = 6,
    ReplacementRC = 7
  };

  void SetUp() override {
    M.reset(createEmptyModule("<main>"));
    Main = insertMainFunction(M.get(), OriginalRC);
  }

  void compileAndRun(int ExpectedRC = OriginalRC) {
    // This function shouldn't be called until after SetUp.
    ASSERT_TRUE(bool(TheJIT));
    ASSERT_TRUE(nullptr != Main);

    // We may be using a null cache, so ensure compilation is valid.
    TheJIT->finalizeObject();
    void *vPtr = TheJIT->getPointerToFunction(Main);

    EXPECT_TRUE(nullptr != vPtr)
      << "Unable to get pointer to main() from JIT";

    int (*FuncPtr)() = (int(*)())(intptr_t)vPtr;
    int returnCode = FuncPtr();
    EXPECT_EQ(returnCode, ExpectedRC);
  }

  Function *Main;
};

TEST_F(MCJITObjectCacheTest, SetNullObjectCache) {
  SKIP_UNSUPPORTED_PLATFORM;

  createJIT(std::move(M));

  TheJIT->setObjectCache(nullptr);

  compileAndRun();
}

TEST_F(MCJITObjectCacheTest, VerifyBasicObjectCaching) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<TestObjectCache> Cache(new TestObjectCache);

  // Save a copy of the module pointer before handing it off to MCJIT.
  const Module * SavedModulePointer = M.get();

  createJIT(std::move(M));

  TheJIT->setObjectCache(Cache.get());

  // Verify that our object cache does not contain the module yet.
  const MemoryBuffer *ObjBuffer = Cache->getObjectInternal(SavedModulePointer);
  EXPECT_EQ(nullptr, ObjBuffer);

  compileAndRun();

  // Verify that MCJIT tried to look-up this module in the cache.
  EXPECT_TRUE(Cache->wasModuleLookedUp(SavedModulePointer));

  // Verify that our object cache now contains the module.
  ObjBuffer = Cache->getObjectInternal(SavedModulePointer);
  EXPECT_TRUE(nullptr != ObjBuffer);

  // Verify that the cache was only notified once.
  EXPECT_FALSE(Cache->wereDuplicatesInserted());
}

TEST_F(MCJITObjectCacheTest, VerifyLoadFromCache) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<TestObjectCache> Cache(new TestObjectCache);

  // Compile this module with an MCJIT engine
  createJIT(std::move(M));
  TheJIT->setObjectCache(Cache.get());
  TheJIT->finalizeObject();

  // Destroy the MCJIT engine we just used
  TheJIT.reset();

  // Create a new memory manager.
  MM.reset(new SectionMemoryManager());

  // Create a new module and save it. Use a different return code so we can
  // tell if MCJIT compiled this module or used the cache.
  M.reset(createEmptyModule("<main>"));
  Main = insertMainFunction(M.get(), ReplacementRC);
  const Module * SecondModulePointer = M.get();

  // Create a new MCJIT instance to load this module then execute it.
  createJIT(std::move(M));
  TheJIT->setObjectCache(Cache.get());
  compileAndRun();

  // Verify that MCJIT tried to look-up this module in the cache.
  EXPECT_TRUE(Cache->wasModuleLookedUp(SecondModulePointer));

  // Verify that MCJIT didn't try to cache this again.
  EXPECT_FALSE(Cache->wereDuplicatesInserted());
}

TEST_F(MCJITObjectCacheTest, VerifyNonLoadFromCache) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<TestObjectCache> Cache(new TestObjectCache);

  // Compile this module with an MCJIT engine
  createJIT(std::move(M));
  TheJIT->setObjectCache(Cache.get());
  TheJIT->finalizeObject();

  // Destroy the MCJIT engine we just used
  TheJIT.reset();

  // Create a new memory manager.
  MM.reset(new SectionMemoryManager());

  // Create a new module and save it. Use a different return code so we can
  // tell if MCJIT compiled this module or used the cache. Note that we use
  // a new module name here so the module shouldn't be found in the cache.
  M.reset(createEmptyModule("<not-main>"));
  Main = insertMainFunction(M.get(), ReplacementRC);
  const Module * SecondModulePointer = M.get();

  // Create a new MCJIT instance to load this module then execute it.
  createJIT(std::move(M));
  TheJIT->setObjectCache(Cache.get());

  // Verify that our object cache does not contain the module yet.
  const MemoryBuffer *ObjBuffer = Cache->getObjectInternal(SecondModulePointer);
  EXPECT_EQ(nullptr, ObjBuffer);

  // Run the function and look for the replacement return code.
  compileAndRun(ReplacementRC);

  // Verify that MCJIT tried to look-up this module in the cache.
  EXPECT_TRUE(Cache->wasModuleLookedUp(SecondModulePointer));

  // Verify that our object cache now contains the module.
  ObjBuffer = Cache->getObjectInternal(SecondModulePointer);
  EXPECT_TRUE(nullptr != ObjBuffer);

  // Verify that MCJIT didn't try to cache this again.
  EXPECT_FALSE(Cache->wereDuplicatesInserted());
}

} // end anonymous namespace
