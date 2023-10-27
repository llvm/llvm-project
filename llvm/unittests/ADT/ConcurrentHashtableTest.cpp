//===- ConcurrentHashtableTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ConcurrentHashtable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"
#include "gtest/gtest.h"
#include <limits>
#include <random>
#include <vector>
using namespace llvm;
using namespace parallel;

namespace {
class String {
public:
  String() {}
  const std::string &getKey() const { return Data; }

  template <typename AllocatorTy>
  static String *create(const std::string &Num, AllocatorTy &Allocator) {
    String *Result = Allocator.template Allocate<String>();
    new (Result) String(Num);
    return Result;
  }

protected:
  String(const std::string &Num) { Data += Num; }

  std::string Data;
  std::array<char, 0x20> ExtraData;
};

TEST(ConcurrentHashTableTest, AddStringEntries) {
  PerThreadBumpPtrAllocator Allocator;
  ConcurrentHashTableByPtr<std::string, String, PerThreadBumpPtrAllocator,
                           ConcurrentHashTableInfoByPtr<
                               std::string, String, PerThreadBumpPtrAllocator>>
      HashTable(Allocator, 10);

  // PerThreadBumpPtrAllocator should be accessed from threads created by
  // ThreadPoolExecutor. Use TaskGroup to run on ThreadPoolExecutor threads.
  parallel::TaskGroup tg;

  tg.spawn([&]() {
    size_t AllocatedBytesAtStart = Allocator.getBytesAllocated();
    std::pair<String *, bool> res1 = HashTable.insert("1");
    // Check entry is inserted.
    EXPECT_TRUE(res1.first->getKey() == "1");
    EXPECT_TRUE(res1.second);

    std::pair<String *, bool> res2 = HashTable.insert("2");
    // Check old entry is still valid.
    EXPECT_TRUE(res1.first->getKey() == "1");
    // Check new entry is inserted.
    EXPECT_TRUE(res2.first->getKey() == "2");
    EXPECT_TRUE(res2.second);
    // Check new and old entries use different memory.
    EXPECT_TRUE(res1.first != res2.first);

    std::pair<String *, bool> res3 = HashTable.insert("3");
    // Check one more entry is inserted.
    EXPECT_TRUE(res3.first->getKey() == "3");
    EXPECT_TRUE(res3.second);

    std::pair<String *, bool> res4 = HashTable.insert("1");
    // Check duplicated entry is inserted.
    EXPECT_TRUE(res4.first->getKey() == "1");
    EXPECT_FALSE(res4.second);
    // Check duplicated entry uses the same memory.
    EXPECT_TRUE(res1.first == res4.first);

    // Check first entry is still valid.
    EXPECT_TRUE(res1.first->getKey() == "1");

    // Check data was allocated by allocator.
    EXPECT_TRUE(Allocator.getBytesAllocated() > AllocatedBytesAtStart);

    // Check statistic.
    std::string StatisticString;
    raw_string_ostream StatisticStream(StatisticString);
    HashTable.printStatistic(StatisticStream);

    EXPECT_TRUE(StatisticString.find("Overall number of entries = 3\n") !=
                std::string::npos);
  });
}

TEST(ConcurrentHashTableTest, AddStringMultiplueEntries) {
  PerThreadBumpPtrAllocator Allocator;
  const size_t NumElements = 10000;
  ConcurrentHashTableByPtr<std::string, String, PerThreadBumpPtrAllocator,
                           ConcurrentHashTableInfoByPtr<
                               std::string, String, PerThreadBumpPtrAllocator>>
      HashTable(Allocator);

  // PerThreadBumpPtrAllocator should be accessed from threads created by
  // ThreadPoolExecutor. Use TaskGroup to run on ThreadPoolExecutor threads.
  parallel::TaskGroup tg;

  tg.spawn([&]() {
    // Check insertion.
    for (size_t I = 0; I < NumElements; I++) {
      BumpPtrAllocator &ThreadLocalAllocator =
          Allocator.getThreadLocalAllocator();
      size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
      std::string StringForElement = formatv("{0}", I);
      std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
      EXPECT_TRUE(Entry.second);
      EXPECT_TRUE(Entry.first->getKey() == StringForElement);
      EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() >
                  AllocatedBytesAtStart);
    }

    std::string StatisticString;
    raw_string_ostream StatisticStream(StatisticString);
    HashTable.printStatistic(StatisticStream);

    // Verifying that the table contains exactly the number of elements we
    // inserted.
    EXPECT_TRUE(StatisticString.find("Overall number of entries = 10000\n") !=
                std::string::npos);

    // Check insertion of duplicates.
    for (size_t I = 0; I < NumElements; I++) {
      BumpPtrAllocator &ThreadLocalAllocator =
          Allocator.getThreadLocalAllocator();
      size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
      std::string StringForElement = formatv("{0}", I);
      std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
      EXPECT_FALSE(Entry.second);
      EXPECT_TRUE(Entry.first->getKey() == StringForElement);
      // Check no additional bytes were allocated for duplicate.
      EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() ==
                  AllocatedBytesAtStart);
    }

    // Check statistic.
    // Verifying that the table contains exactly the number of elements we
    // inserted.
    EXPECT_TRUE(StatisticString.find("Overall number of entries = 10000\n") !=
                std::string::npos);
  });
}

TEST(ConcurrentHashTableTest, AddStringMultiplueEntriesWithResize) {
  PerThreadBumpPtrAllocator Allocator;
  // Number of elements exceeds original size, thus hashtable should be resized.
  const size_t NumElements = 20000;
  ConcurrentHashTableByPtr<std::string, String, PerThreadBumpPtrAllocator,
                           ConcurrentHashTableInfoByPtr<
                               std::string, String, PerThreadBumpPtrAllocator>>
      HashTable(Allocator, 100);

  // PerThreadBumpPtrAllocator should be accessed from threads created by
  // ThreadPoolExecutor. Use TaskGroup to run on ThreadPoolExecutor threads.
  parallel::TaskGroup tg;

  tg.spawn([&]() {
    // Check insertion.
    for (size_t I = 0; I < NumElements; I++) {
      BumpPtrAllocator &ThreadLocalAllocator =
          Allocator.getThreadLocalAllocator();
      size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
      std::string StringForElement = formatv("{0} {1}", I, I + 100);
      std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
      EXPECT_TRUE(Entry.second);
      EXPECT_TRUE(Entry.first->getKey() == StringForElement);
      EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() >
                  AllocatedBytesAtStart);
    }

    std::string StatisticString;
    raw_string_ostream StatisticStream(StatisticString);
    HashTable.printStatistic(StatisticStream);

    // Verifying that the table contains exactly the number of elements we
    // inserted.
    EXPECT_TRUE(StatisticString.find("Overall number of entries = 20000\n") !=
                std::string::npos);

    // Check insertion of duplicates.
    for (size_t I = 0; I < NumElements; I++) {
      BumpPtrAllocator &ThreadLocalAllocator =
          Allocator.getThreadLocalAllocator();
      size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
      std::string StringForElement = formatv("{0} {1}", I, I + 100);
      std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
      EXPECT_FALSE(Entry.second);
      EXPECT_TRUE(Entry.first->getKey() == StringForElement);
      // Check no additional bytes were allocated for duplicate.
      EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() ==
                  AllocatedBytesAtStart);
    }

    // Check statistic.
    // Verifying that the table contains exactly the number of elements we
    // inserted.
    EXPECT_TRUE(StatisticString.find("Overall number of entries = 20000\n") !=
                std::string::npos);
  });
}

TEST(ConcurrentHashTableTest, AddStringEntriesParallel) {
  PerThreadBumpPtrAllocator Allocator;
  const size_t NumElements = 10000;
  ConcurrentHashTableByPtr<std::string, String, PerThreadBumpPtrAllocator,
                           ConcurrentHashTableInfoByPtr<
                               std::string, String, PerThreadBumpPtrAllocator>>
      HashTable(Allocator);

  // Check parallel insertion.
  parallelFor(0, NumElements, [&](size_t I) {
    BumpPtrAllocator &ThreadLocalAllocator =
        Allocator.getThreadLocalAllocator();
    size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
    std::string StringForElement = formatv("{0}", I);
    std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
    EXPECT_TRUE(Entry.second);
    EXPECT_TRUE(Entry.first->getKey() == StringForElement);
    EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() >
                AllocatedBytesAtStart);
  });

  std::string StatisticString;
  raw_string_ostream StatisticStream(StatisticString);
  HashTable.printStatistic(StatisticStream);

  // Verifying that the table contains exactly the number of elements we
  // inserted.
  EXPECT_TRUE(StatisticString.find("Overall number of entries = 10000\n") !=
              std::string::npos);

  // Check parallel insertion of duplicates.
  parallelFor(0, NumElements, [&](size_t I) {
    BumpPtrAllocator &ThreadLocalAllocator =
        Allocator.getThreadLocalAllocator();
    size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
    std::string StringForElement = formatv("{0}", I);
    std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
    EXPECT_FALSE(Entry.second);
    EXPECT_TRUE(Entry.first->getKey() == StringForElement);
    // Check no additional bytes were allocated for duplicate.
    EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() ==
                AllocatedBytesAtStart);
  });

  // Check statistic.
  // Verifying that the table contains exactly the number of elements we
  // inserted.
  EXPECT_TRUE(StatisticString.find("Overall number of entries = 10000\n") !=
              std::string::npos);
}

TEST(ConcurrentHashTableTest, AddStringEntriesParallelWithResize) {
  PerThreadBumpPtrAllocator Allocator;
  const size_t NumElements = 20000;
  ConcurrentHashTableByPtr<std::string, String, PerThreadBumpPtrAllocator,
                           ConcurrentHashTableInfoByPtr<
                               std::string, String, PerThreadBumpPtrAllocator>>
      HashTable(Allocator, 100);

  // Check parallel insertion.
  parallelFor(0, NumElements, [&](size_t I) {
    BumpPtrAllocator &ThreadLocalAllocator =
        Allocator.getThreadLocalAllocator();
    size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
    std::string StringForElement = formatv("{0}", I);
    std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
    EXPECT_TRUE(Entry.second);
    EXPECT_TRUE(Entry.first->getKey() == StringForElement);
    EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() >
                AllocatedBytesAtStart);
  });

  std::string StatisticString;
  raw_string_ostream StatisticStream(StatisticString);
  HashTable.printStatistic(StatisticStream);

  // Verifying that the table contains exactly the number of elements we
  // inserted.
  EXPECT_TRUE(StatisticString.find("Overall number of entries = 20000\n") !=
              std::string::npos);

  // Check parallel insertion of duplicates.
  parallelFor(0, NumElements, [&](size_t I) {
    BumpPtrAllocator &ThreadLocalAllocator =
        Allocator.getThreadLocalAllocator();
    size_t AllocatedBytesAtStart = ThreadLocalAllocator.getBytesAllocated();
    std::string StringForElement = formatv("{0}", I);
    std::pair<String *, bool> Entry = HashTable.insert(StringForElement);
    EXPECT_FALSE(Entry.second);
    EXPECT_TRUE(Entry.first->getKey() == StringForElement);
    // Check no additional bytes were allocated for duplicate.
    EXPECT_TRUE(ThreadLocalAllocator.getBytesAllocated() ==
                AllocatedBytesAtStart);
  });

  // Check statistic.
  // Verifying that the table contains exactly the number of elements we
  // inserted.
  EXPECT_TRUE(StatisticString.find("Overall number of entries = 20000\n") !=
              std::string::npos);
}

} // namespace
