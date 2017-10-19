//===- llvm/unittest/Support/AllocatorTest.cpp - BumpPtrAllocator tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Memory.h"
#include "llvm/Support/Process.h"
#include "gtest/gtest.h"
#include <cstdlib>

using namespace llvm;
using namespace sys;

namespace {

class MappedMemoryTest : public ::testing::TestWithParam<unsigned> {
public:
  MappedMemoryTest() {
    Flags = GetParam();
    PageSize = sys::Process::getPageSize();
  }

protected:
  // Adds RW flags to permit testing of the resulting memory
  unsigned getTestableEquivalent(unsigned RequestedFlags) {
    switch (RequestedFlags) {
    case Memory::MF_READ:
    case Memory::MF_WRITE:
    case Memory::MF_READ|Memory::MF_WRITE:
      return Memory::MF_READ|Memory::MF_WRITE;
    case Memory::MF_READ|Memory::MF_EXEC:
    case Memory::MF_READ|Memory::MF_WRITE|Memory::MF_EXEC:
    case Memory::MF_EXEC:
      return Memory::MF_READ|Memory::MF_WRITE|Memory::MF_EXEC;
    }
    // Default in case values are added to the enum, as required by some compilers
    return Memory::MF_READ|Memory::MF_WRITE;
  }

  // Returns true if the memory blocks overlap
  bool doesOverlap(MemoryBlock M1, MemoryBlock M2) {
    if (M1.base() == M2.base())
      return true;

    if (M1.base() > M2.base())
      return (unsigned char *)M2.base() + M2.size() > M1.base();

    return (unsigned char *)M1.base() + M1.size() > M2.base();
  }

  unsigned Flags;
  size_t   PageSize;
};

TEST_P(MappedMemoryTest, AllocAndRelease) {
  std::error_code EC;
  MemoryBlock M1 = Memory::allocateMappedMemory(sizeof(int), nullptr, Flags,EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(sizeof(int), M1.size());

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
}

TEST_P(MappedMemoryTest, MultipleAllocAndRelease) {
  std::error_code EC;
  MemoryBlock M1 = Memory::allocateMappedMemory(16, nullptr, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(64, nullptr, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(32, nullptr, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(16U, M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(64U, M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(32U, M3.size());

  EXPECT_FALSE(doesOverlap(M1, M2));
  EXPECT_FALSE(doesOverlap(M2, M3));
  EXPECT_FALSE(doesOverlap(M1, M3));

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));
  MemoryBlock M4 = Memory::allocateMappedMemory(16, nullptr, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  EXPECT_NE((void*)nullptr, M4.base());
  EXPECT_LE(16U, M4.size());
  EXPECT_FALSE(Memory::releaseMappedMemory(M4));
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, BasicWrite) {
  // This test applies only to readable and writeable combinations
  if (Flags &&
      !((Flags & Memory::MF_READ) && (Flags & Memory::MF_WRITE)))
    return;

  std::error_code EC;
  MemoryBlock M1 = Memory::allocateMappedMemory(sizeof(int), nullptr, Flags,EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(sizeof(int), M1.size());

  int *a = (int*)M1.base();
  *a = 1;
  EXPECT_EQ(1, *a);

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
}

TEST_P(MappedMemoryTest, MultipleWrite) {
  // This test applies only to readable and writeable combinations
  if (Flags &&
      !((Flags & Memory::MF_READ) && (Flags & Memory::MF_WRITE)))
    return;
  std::error_code EC;
  MemoryBlock M1 = Memory::allocateMappedMemory(sizeof(int), nullptr, Flags,
                                                EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(8 * sizeof(int), nullptr, Flags,
                                                EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(4 * sizeof(int), nullptr, Flags,
                                                EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_FALSE(doesOverlap(M1, M2));
  EXPECT_FALSE(doesOverlap(M2, M3));
  EXPECT_FALSE(doesOverlap(M1, M3));

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(1U * sizeof(int), M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(8U * sizeof(int), M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(4U * sizeof(int), M3.size());

  int *x = (int*)M1.base();
  *x = 1;

  int *y = (int*)M2.base();
  for (int i = 0; i < 8; i++) {
    y[i] = i;
  }

  int *z = (int*)M3.base();
  *z = 42;

  EXPECT_EQ(1, *x);
  EXPECT_EQ(7, y[7]);
  EXPECT_EQ(42, *z);

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));

  MemoryBlock M4 = Memory::allocateMappedMemory(64 * sizeof(int), nullptr,
                                                Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  EXPECT_NE((void*)nullptr, M4.base());
  EXPECT_LE(64U * sizeof(int), M4.size());
  x = (int*)M4.base();
  *x = 4;
  EXPECT_EQ(4, *x);
  EXPECT_FALSE(Memory::releaseMappedMemory(M4));

  // Verify that M2 remains unaffected by other activity
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(i, y[i]);
  }
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, EnabledWrite) {
  std::error_code EC;
  MemoryBlock M1 = Memory::allocateMappedMemory(2 * sizeof(int), nullptr, Flags,
                                                EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(8 * sizeof(int), nullptr, Flags,
                                                EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(4 * sizeof(int), nullptr, Flags,
                                                EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(2U * sizeof(int), M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(8U * sizeof(int), M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(4U * sizeof(int), M3.size());

  EXPECT_FALSE(Memory::protectMappedMemory(M1, getTestableEquivalent(Flags)));
  EXPECT_FALSE(Memory::protectMappedMemory(M2, getTestableEquivalent(Flags)));
  EXPECT_FALSE(Memory::protectMappedMemory(M3, getTestableEquivalent(Flags)));

  EXPECT_FALSE(doesOverlap(M1, M2));
  EXPECT_FALSE(doesOverlap(M2, M3));
  EXPECT_FALSE(doesOverlap(M1, M3));

  int *x = (int*)M1.base();
  *x = 1;
  int *y = (int*)M2.base();
  for (unsigned int i = 0; i < 8; i++) {
    y[i] = i;
  }
  int *z = (int*)M3.base();
  *z = 42;

  EXPECT_EQ(1, *x);
  EXPECT_EQ(7, y[7]);
  EXPECT_EQ(42, *z);

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));
  EXPECT_EQ(6, y[6]);

  MemoryBlock M4 = Memory::allocateMappedMemory(16, nullptr, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  EXPECT_NE((void*)nullptr, M4.base());
  EXPECT_LE(16U, M4.size());
  EXPECT_EQ(std::error_code(),
            Memory::protectMappedMemory(M4, getTestableEquivalent(Flags)));
  x = (int*)M4.base();
  *x = 4;
  EXPECT_EQ(4, *x);
  EXPECT_FALSE(Memory::releaseMappedMemory(M4));
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, SuccessiveNear) {
  std::error_code EC;
  MemoryBlock M1 = Memory::allocateMappedMemory(16, nullptr, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(64, &M1, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(32, &M2, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(16U, M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(64U, M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(32U, M3.size());

  EXPECT_FALSE(doesOverlap(M1, M2));
  EXPECT_FALSE(doesOverlap(M2, M3));
  EXPECT_FALSE(doesOverlap(M1, M3));

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, DuplicateNear) {
  std::error_code EC;
  MemoryBlock Near((void*)(3*PageSize), 16);
  MemoryBlock M1 = Memory::allocateMappedMemory(16, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(64, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(32, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(16U, M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(64U, M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(32U, M3.size());

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, ZeroNear) {
  std::error_code EC;
  MemoryBlock Near(nullptr, 0);
  MemoryBlock M1 = Memory::allocateMappedMemory(16, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(64, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(32, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(16U, M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(64U, M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(32U, M3.size());

  EXPECT_FALSE(doesOverlap(M1, M2));
  EXPECT_FALSE(doesOverlap(M2, M3));
  EXPECT_FALSE(doesOverlap(M1, M3));

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, ZeroSizeNear) {
  std::error_code EC;
  MemoryBlock Near((void*)(4*PageSize), 0);
  MemoryBlock M1 = Memory::allocateMappedMemory(16, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M2 = Memory::allocateMappedMemory(64, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);
  MemoryBlock M3 = Memory::allocateMappedMemory(32, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(16U, M1.size());
  EXPECT_NE((void*)nullptr, M2.base());
  EXPECT_LE(64U, M2.size());
  EXPECT_NE((void*)nullptr, M3.base());
  EXPECT_LE(32U, M3.size());

  EXPECT_FALSE(doesOverlap(M1, M2));
  EXPECT_FALSE(doesOverlap(M2, M3));
  EXPECT_FALSE(doesOverlap(M1, M3));

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
  EXPECT_FALSE(Memory::releaseMappedMemory(M3));
  EXPECT_FALSE(Memory::releaseMappedMemory(M2));
}

TEST_P(MappedMemoryTest, UnalignedNear) {
  std::error_code EC;
  MemoryBlock Near((void*)(2*PageSize+5), 0);
  MemoryBlock M1 = Memory::allocateMappedMemory(15, &Near, Flags, EC);
  EXPECT_EQ(std::error_code(), EC);

  EXPECT_NE((void*)nullptr, M1.base());
  EXPECT_LE(sizeof(int), M1.size());

  EXPECT_FALSE(Memory::releaseMappedMemory(M1));
}

// Note that Memory::MF_WRITE is not supported exclusively across
// operating systems and architectures and can imply MF_READ|MF_WRITE
unsigned MemoryFlags[] = {
                           Memory::MF_READ,
                           Memory::MF_WRITE,
                           Memory::MF_READ|Memory::MF_WRITE,
                           Memory::MF_EXEC,
                           Memory::MF_READ|Memory::MF_EXEC,
                           Memory::MF_READ|Memory::MF_WRITE|Memory::MF_EXEC
                         };

INSTANTIATE_TEST_CASE_P(AllocationTests,
                        MappedMemoryTest,
                        ::testing::ValuesIn(MemoryFlags),);

}  // anonymous namespace
