//===- MCJITMemoryManagerTest.cpp - Unit tests for the JIT memory manager -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MCJITMemoryManagerTest, BasicAllocations) {
  std::unique_ptr<SectionMemoryManager> MemMgr(new SectionMemoryManager());

  uint8_t *code1 = MemMgr->allocateCodeSection(256, 0, 1, "");
  uint8_t *data1 = MemMgr->allocateDataSection(256, 0, 2, "", true);
  uint8_t *code2 = MemMgr->allocateCodeSection(256, 0, 3, "");
  uint8_t *data2 = MemMgr->allocateDataSection(256, 0, 4, "", false);

  EXPECT_NE((uint8_t*)nullptr, code1);
  EXPECT_NE((uint8_t*)nullptr, code2);
  EXPECT_NE((uint8_t*)nullptr, data1);
  EXPECT_NE((uint8_t*)nullptr, data2);

  // Initialize the data
  for (unsigned i = 0; i < 256; ++i) {
    code1[i] = 1;
    code2[i] = 2;
    data1[i] = 3;
    data2[i] = 4;
  }

  // Verify the data (this is checking for overlaps in the addresses)
  for (unsigned i = 0; i < 256; ++i) {
    EXPECT_EQ(1, code1[i]);
    EXPECT_EQ(2, code2[i]);
    EXPECT_EQ(3, data1[i]);
    EXPECT_EQ(4, data2[i]);
  }

  std::string Error;
  EXPECT_FALSE(MemMgr->finalizeMemory(&Error));
}

TEST(MCJITMemoryManagerTest, LargeAllocations) {
  std::unique_ptr<SectionMemoryManager> MemMgr(new SectionMemoryManager());

  uint8_t *code1 = MemMgr->allocateCodeSection(0x100000, 0, 1, "");
  uint8_t *data1 = MemMgr->allocateDataSection(0x100000, 0, 2, "", true);
  uint8_t *code2 = MemMgr->allocateCodeSection(0x100000, 0, 3, "");
  uint8_t *data2 = MemMgr->allocateDataSection(0x100000, 0, 4, "", false);

  EXPECT_NE((uint8_t*)nullptr, code1);
  EXPECT_NE((uint8_t*)nullptr, code2);
  EXPECT_NE((uint8_t*)nullptr, data1);
  EXPECT_NE((uint8_t*)nullptr, data2);

  // Initialize the data
  for (unsigned i = 0; i < 0x100000; ++i) {
    code1[i] = 1;
    code2[i] = 2;
    data1[i] = 3;
    data2[i] = 4;
  }

  // Verify the data (this is checking for overlaps in the addresses)
  for (unsigned i = 0; i < 0x100000; ++i) {
    EXPECT_EQ(1, code1[i]);
    EXPECT_EQ(2, code2[i]);
    EXPECT_EQ(3, data1[i]);
    EXPECT_EQ(4, data2[i]);
  }

  std::string Error;
  EXPECT_FALSE(MemMgr->finalizeMemory(&Error));
}

TEST(MCJITMemoryManagerTest, ManyAllocations) {
  std::unique_ptr<SectionMemoryManager> MemMgr(new SectionMemoryManager());

  uint8_t* code[10000];
  uint8_t* data[10000];

  for (unsigned i = 0; i < 10000; ++i) {
    const bool isReadOnly = i % 2 == 0;

    code[i] = MemMgr->allocateCodeSection(32, 0, 1, "");
    data[i] = MemMgr->allocateDataSection(32, 0, 2, "", isReadOnly);

    for (unsigned j = 0; j < 32; j++) {
      code[i][j] = 1 + (i % 254);
      data[i][j] = 2 + (i % 254);
    }

    EXPECT_NE((uint8_t *)nullptr, code[i]);
    EXPECT_NE((uint8_t *)nullptr, data[i]);
  }

  // Verify the data (this is checking for overlaps in the addresses)
  for (unsigned i = 0; i < 10000; ++i) {
    for (unsigned j = 0; j < 32;j++ ) {
      uint8_t ExpectedCode = 1 + (i % 254);
      uint8_t ExpectedData = 2 + (i % 254);
      EXPECT_EQ(ExpectedCode, code[i][j]);
      EXPECT_EQ(ExpectedData, data[i][j]);
    }
  }

  std::string Error;
  EXPECT_FALSE(MemMgr->finalizeMemory(&Error));
}

TEST(MCJITMemoryManagerTest, ManyVariedAllocations) {
  std::unique_ptr<SectionMemoryManager> MemMgr(new SectionMemoryManager());

  uint8_t* code[10000];
  uint8_t* data[10000];

  for (unsigned i = 0; i < 10000; ++i) {
    uintptr_t CodeSize = i % 16 + 1;
    uintptr_t DataSize = i % 8 + 1;

    bool isReadOnly = i % 3 == 0;
    unsigned Align = 8 << (i % 4);

    code[i] = MemMgr->allocateCodeSection(CodeSize, Align, i, "");
    data[i] = MemMgr->allocateDataSection(DataSize, Align, i + 10000, "",
                                          isReadOnly);

    for (unsigned j = 0; j < CodeSize; j++) {
      code[i][j] = 1 + (i % 254);
    }

    for (unsigned j = 0; j < DataSize; j++) {
      data[i][j] = 2 + (i % 254);
    }

    EXPECT_NE((uint8_t *)nullptr, code[i]);
    EXPECT_NE((uint8_t *)nullptr, data[i]);

    uintptr_t CodeAlign = Align ? (uintptr_t)code[i] % Align : 0;
    uintptr_t DataAlign = Align ? (uintptr_t)data[i] % Align : 0;

    EXPECT_EQ((uintptr_t)0, CodeAlign);
    EXPECT_EQ((uintptr_t)0, DataAlign);
  }

  for (unsigned i = 0; i < 10000; ++i) {
    uintptr_t CodeSize = i % 16 + 1;
    uintptr_t DataSize = i % 8 + 1;

    for (unsigned j = 0; j < CodeSize; j++) {
      uint8_t ExpectedCode = 1 + (i % 254);
      EXPECT_EQ(ExpectedCode, code[i][j]);
    }

    for (unsigned j = 0; j < DataSize; j++) {
      uint8_t ExpectedData = 2 + (i % 254);
      EXPECT_EQ(ExpectedData, data[i][j]); 
    }
  }
}

} // Namespace

