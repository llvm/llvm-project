//===-- PerThreadTable.h -- PerThread Storage Structure ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Table indexed with one entry per thread.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PERTHREADTABLE_H
#define OFFLOAD_PERTHREADTABLE_H

#include <list>
#include <memory>
#include <mutex>

// Using an STL container (such as std::vector) indexed by thread ID has
// too many race conditions issues so we store each thread entry into a
// thread_local variable.
// T is the container type used to store the objects, e.g., std::vector,
// std::set, etc. by each thread. O is the type of the stored objects e.g.,
// omp_interop_val_t *, ...

template <typename ContainerType, typename ObjectType> struct PerThreadTable {
  using iterator = typename ContainerType::iterator;

  struct PerThreadData {
    size_t NElements = 0;
    std::unique_ptr<ContainerType> ThEntry;
  };

  std::mutex Mtx;
  std::list<std::shared_ptr<PerThreadData>> ThreadDataList;

  // define default constructors, disable copy and move constructors
  PerThreadTable() = default;
  PerThreadTable(const PerThreadTable &) = delete;
  PerThreadTable(PerThreadTable &&) = delete;
  PerThreadTable &operator=(const PerThreadTable &) = delete;
  PerThreadTable &operator=(PerThreadTable &&) = delete;
  ~PerThreadTable() {
    std::lock_guard<std::mutex> Lock(Mtx);
    ThreadDataList.clear();
  }

private:
  PerThreadData &getThreadData() {
    static thread_local std::shared_ptr<PerThreadData> ThData = nullptr;
    if (!ThData) {
      ThData = std::make_shared<PerThreadData>();
      std::lock_guard<std::mutex> Lock(Mtx);
      ThreadDataList.push_back(ThData);
    }
    return *ThData;
  }

protected:
  ContainerType &getThreadEntry() {
    auto &ThData = getThreadData();
    if (ThData.ThEntry)
      return *ThData.ThEntry;
    ThData.ThEntry = std::make_unique<ContainerType>();
    return *ThData.ThEntry;
  }

  size_t &getThreadNElements() {
    auto &ThData = getThreadData();
    return ThData.NElements;
  }

public:
  void add(ObjectType obj) {
    auto &Entry = getThreadEntry();
    auto &NElements = getThreadNElements();
    NElements++;
    Entry.add(obj);
  }

  iterator erase(iterator it) {
    auto &Entry = getThreadEntry();
    auto &NElements = getThreadNElements();
    NElements--;
    return Entry.erase(it);
  }

  size_t size() { return getThreadNElements(); }

  // Iterators to traverse objects owned by
  // the current thread
  iterator begin() {
    auto &Entry = getThreadEntry();
    return Entry.begin();
  }
  iterator end() {
    auto &Entry = getThreadEntry();
    return Entry.end();
  }

  template <class F> void clear(F f) {
    std::lock_guard<std::mutex> Lock(Mtx);
    for (auto ThData : ThreadDataList) {
      if (!ThData->ThEntry || ThData->NElements == 0)
        continue;
      ThData->ThEntry->clear(f);
      ThData->NElements = 0;
    }
    ThreadDataList.clear();
  }
};

#endif
