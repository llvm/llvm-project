//===- MCJITMemoryManagerTest.cpp - Unit tests for the JIT memory manager -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(MCJITMemoryManagerTest, BasicAllocations) {
  std::unique_ptr<SectionMemoryManager> MemMgr(new SectionMemoryManager());

  EXPECT_FALSE(MemMgr->needsToReserveAllocationSpace());

  uint8_t *code1 = MemMgr->allocateCodeSection(256, 0, 1, "");
  uint8_t *data1 = MemMgr->allocateDataSection(256, 0, 2, "", true);
  uint8_t *code2 = MemMgr->allocateCodeSection(256, 0, 3, "");
  uint8_t *data2 = MemMgr->allocateDataSection(256, 0, 4, "", false);

  EXPECT_NE((uint8_t *)nullptr, code1);
  EXPECT_NE((uint8_t *)nullptr, code2);
  EXPECT_NE((uint8_t *)nullptr, data1);
  EXPECT_NE((uint8_t *)nullptr, data2);

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

  EXPECT_NE((uint8_t *)nullptr, code1);
  EXPECT_NE((uint8_t *)nullptr, code2);
  EXPECT_NE((uint8_t *)nullptr, data1);
  EXPECT_NE((uint8_t *)nullptr, data2);

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

  uint8_t *code[10000];
  uint8_t *data[10000];

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

  uint8_t *code[10000];
  uint8_t *data[10000];

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

TEST(MCJITMemoryManagerTest, PreAllocation) {
  std::unique_ptr<SectionMemoryManager> MemMgr(
      new SectionMemoryManager(nullptr, true));

  EXPECT_TRUE(MemMgr->needsToReserveAllocationSpace());

  llvm::Align Align{16};
  MemMgr->reserveAllocationSpace(512, Align, 256, Align, 256, Align);

  uint8_t *code1 = MemMgr->allocateCodeSection(256, 0, 1, "");
  uint8_t *data1 = MemMgr->allocateDataSection(256, 0, 2, "", true);
  uint8_t *code2 = MemMgr->allocateCodeSection(256, 0, 3, "");
  uint8_t *data2 = MemMgr->allocateDataSection(256, 0, 4, "", false);

  uint8_t *minAddr = std::min({code1, data1, code2, data2});
  uint8_t *maxAddr = std::max({code1, data1, code2, data2});

  EXPECT_NE((uint8_t *)nullptr, code1);
  EXPECT_NE((uint8_t *)nullptr, code2);
  EXPECT_NE((uint8_t *)nullptr, data1);
  EXPECT_NE((uint8_t *)nullptr, data2);

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

  MemMgr->reserveAllocationSpace(512, Align, 256, Align, 256, Align);

  code1 = MemMgr->allocateCodeSection(256, 0, 1, "");
  data1 = MemMgr->allocateDataSection(256, 0, 2, "", true);
  code2 = MemMgr->allocateCodeSection(256, 0, 3, "");
  data2 = MemMgr->allocateDataSection(256, 0, 4, "", false);

  EXPECT_NE((uint8_t *)nullptr, code1);
  EXPECT_NE((uint8_t *)nullptr, code2);
  EXPECT_NE((uint8_t *)nullptr, data1);
  EXPECT_NE((uint8_t *)nullptr, data2);

  // Validate difference is more than 3x PageSize (the original reservation).
  minAddr = std::min({minAddr, code1, data1, code2, data2});
  maxAddr = std::max({maxAddr, code1, data1, code2, data2});
  EXPECT_GT(maxAddr - minAddr, 3 * sys::Process::getPageSizeEstimate());

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

  EXPECT_FALSE(MemMgr->finalizeMemory(&Error));
}

TEST(MCJITMemoryManagerTest, PreAllocationReuse) {
  std::unique_ptr<SectionMemoryManager> MemMgr(
      new SectionMemoryManager(nullptr, true));

  EXPECT_TRUE(MemMgr->needsToReserveAllocationSpace());

  // Reserve PageSize, because finalizeMemory eliminates blocks that aren't a
  // full page size. Alignment adjustment will ensure that 2 pages are
  // allocated for each section.
  const unsigned PageSize = sys::Process::getPageSizeEstimate();
  EXPECT_GE(PageSize, 512u);
  llvm::Align Align{16};
  MemMgr->reserveAllocationSpace(PageSize, Align, PageSize, Align, PageSize,
                                 Align);

  uint8_t *code1 = MemMgr->allocateCodeSection(256, 0, 1, "");
  uint8_t *data1 = MemMgr->allocateDataSection(256, 0, 2, "", true);
  uint8_t *code2 = MemMgr->allocateCodeSection(256, 0, 3, "");
  uint8_t *data2 = MemMgr->allocateDataSection(256, 0, 4, "", false);

  uint8_t *minAddr = std::min({code1, data1, code2, data2});
  uint8_t *maxAddr = std::max({code1, data1, code2, data2});

  EXPECT_NE((uint8_t *)nullptr, code1);
  EXPECT_NE((uint8_t *)nullptr, code2);
  EXPECT_NE((uint8_t *)nullptr, data1);
  EXPECT_NE((uint8_t *)nullptr, data2);

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

  // Each type of data is allocated on PageSize (usually 4KB). Allocate again
  // and guarantee we get requests in the same block.
  MemMgr->reserveAllocationSpace(512, Align, 256, Align, 256, Align);

  code1 = MemMgr->allocateCodeSection(256, 0, 5, "");
  data1 = MemMgr->allocateDataSection(256, 0, 6, "", true);
  code2 = MemMgr->allocateCodeSection(256, 0, 7, "");
  data2 = MemMgr->allocateDataSection(256, 0, 8, "", false);

  EXPECT_NE((uint8_t *)nullptr, code1);
  EXPECT_NE((uint8_t *)nullptr, code2);
  EXPECT_NE((uint8_t *)nullptr, data1);
  EXPECT_NE((uint8_t *)nullptr, data2);

  // Validate difference is less than 6x PageSize
  minAddr = std::min({minAddr, code1, data1, code2, data2});
  maxAddr = std::max({maxAddr, code1, data1, code2, data2});
  EXPECT_LT(maxAddr - minAddr, 6 * PageSize);

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

  EXPECT_FALSE(MemMgr->finalizeMemory(&Error));
}

TEST(MCJITMemoryManagerTest, ManyPreAllocation) {
  std::unique_ptr<SectionMemoryManager> MemMgr(
      new SectionMemoryManager(nullptr, true));

  uint8_t *code[10000];
  uint8_t *data[10000];

  // Total size computation needs to take into account how much memory will be
  // used including alignment.
  uintptr_t CodeSize = 0, RODataSize = 0, RWDataSize = 0;
  for (unsigned i = 0; i < 10000; ++i) {
    unsigned Align = 8 << (i % 4);
    CodeSize += alignTo(i % 16 + 1, Align);
    if (i % 3 == 0) {
      RODataSize += alignTo(i % 8 + 1, Align);
    } else {
      RWDataSize += alignTo(i % 8 + 1, Align);
    }
  }
  llvm::Align Align = llvm::Align(8);
  MemMgr->reserveAllocationSpace(CodeSize, Align, RODataSize, Align, RWDataSize,
                                 Align);
  uint8_t *minAddr = (uint8_t *)std::numeric_limits<uintptr_t>::max();
  uint8_t *maxAddr = (uint8_t *)std::numeric_limits<uintptr_t>::min();

  for (unsigned i = 0; i < 10000; ++i) {
    uintptr_t CodeSize = i % 16 + 1;
    uintptr_t DataSize = i % 8 + 1;

    bool isReadOnly = i % 3 == 0;
    unsigned Align = 8 << (i % 4);

    code[i] = MemMgr->allocateCodeSection(CodeSize, Align, i, "");
    data[i] =
        MemMgr->allocateDataSection(DataSize, Align, i + 10000, "", isReadOnly);
    minAddr = std::min({minAddr, code[i], data[i]});
    maxAddr = std::max({maxAddr, code[i], data[i]});

    EXPECT_NE((uint8_t *)nullptr, code[i]);
    EXPECT_NE((uint8_t *)nullptr, data[i]);

    for (unsigned j = 0; j < CodeSize; j++) {
      code[i][j] = 1 + (i % 254);
    }

    for (unsigned j = 0; j < DataSize; j++) {
      data[i][j] = 2 + (i % 254);
    }

    uintptr_t CodeAlign = Align ? (uintptr_t)code[i] % Align : 0;
    uintptr_t DataAlign = Align ? (uintptr_t)data[i] % Align : 0;

    EXPECT_EQ((uintptr_t)0, CodeAlign);
    EXPECT_EQ((uintptr_t)0, DataAlign);
  }

  EXPECT_LT(maxAddr - minAddr, 1024 * 1024 * 1024);

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

