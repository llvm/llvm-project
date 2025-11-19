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
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <memory>
#include <mutex>
#include <type_traits>

template <typename ObjectType> class PerThread {
  struct PerThreadData {
    std::unique_ptr<ObjectType> ThreadEntry;
  };

  std::mutex Mutex;
  llvm::SmallVector<std::shared_ptr<PerThreadData>> ThreadDataList;

  PerThreadData &getThreadData() {
    static thread_local std::shared_ptr<PerThreadData> ThreadData = nullptr;
    if (!ThreadData) {
      ThreadData = std::make_shared<PerThreadData>();
      std::lock_guard<std::mutex> Lock(Mutex);
      ThreadDataList.push_back(ThreadData);
    }
    return *ThreadData;
  }

  ObjectType &getThreadEntry() {
    PerThreadData &ThreadData = getThreadData();
    if (ThreadData.ThreadEntry)
      return *ThreadData.ThreadEntry;
    ThreadData.ThreadEntry = std::make_unique<ObjectType>();
    return *ThreadData.ThreadEntry;
  }

public:
  // define default constructors, disable copy and move constructors
  PerThread() = default;
  PerThread(const PerThread &) = delete;
  PerThread(PerThread &&) = delete;
  PerThread &operator=(const PerThread &) = delete;
  PerThread &operator=(PerThread &&) = delete;
  ~PerThread() {
    assert(Mutex.try_lock() && (Mutex.unlock(), true) &&
           "Cannot be deleted while other threads are adding entries");
    ThreadDataList.clear();
  }

  ObjectType &get() { return getThreadEntry(); }

  template <class ClearFuncTy> void clear(ClearFuncTy ClearFunc) {
    assert(Mutex.try_lock() && (Mutex.unlock(), true) &&
           "Clear cannot be called while other threads are adding entries");
    for (std::shared_ptr<PerThreadData> ThreadData : ThreadDataList) {
      if (!ThreadData->ThreadEntry)
        continue;
      ClearFunc(*ThreadData->ThreadEntry);
    }
    ThreadDataList.clear();
  }
};

template <typename ContainerTy> struct ContainerConcepts {
  template <typename, template <typename> class, typename = std::void_t<>>
  struct has : std::false_type {};
  template <typename Ty, template <typename> class Op>
  struct has<Ty, Op, std::void_t<Op<Ty>>> : std::true_type {};

  template <typename Ty> using IteratorTypeCheck = typename Ty::iterator;
  template <typename Ty> using MappedTypeCheck = typename Ty::mapped_type;
  template <typename Ty> using ValueTypeCheck = typename Ty::value_type;
  template <typename Ty> using KeyTypeCheck = typename Ty::key_type;
  template <typename Ty> using SizeTyCheck = typename Ty::size_type;

  template <typename Ty>
  using ClearCheck = decltype(std::declval<Ty>().clear());
  template <typename Ty>
  using ReserveCheck = decltype(std::declval<Ty>().reserve(1));
  template <typename Ty>
  using ResizeCheck = decltype(std::declval<Ty>().resize(1));

  static constexpr bool hasIterator =
      has<ContainerTy, IteratorTypeCheck>::value;
  static constexpr bool hasClear = has<ContainerTy, ClearCheck>::value;
  static constexpr bool isAssociative =
      has<ContainerTy, MappedTypeCheck>::value;
  static constexpr bool hasReserve = has<ContainerTy, ReserveCheck>::value;
  static constexpr bool hasResize = has<ContainerTy, ResizeCheck>::value;

  template <typename, template <typename> class, typename = std::void_t<>>
  struct has_type {
    using type = void;
  };
  template <typename Ty, template <typename> class Op>
  struct has_type<Ty, Op, std::void_t<Op<Ty>>> {
    using type = Op<Ty>;
  };

  using iterator = typename has_type<ContainerTy, IteratorTypeCheck>::type;
  using value_type = typename std::conditional_t<
      isAssociative, typename has_type<ContainerTy, MappedTypeCheck>::type,
      typename has_type<ContainerTy, ValueTypeCheck>::type>;
  using key_type = typename std::conditional_t<
      isAssociative, typename has_type<ContainerTy, KeyTypeCheck>::type,
      typename has_type<ContainerTy, SizeTyCheck>::type>;
};

// Using an STL container (such as std::vector) indexed by thread ID has
// too many race conditions issues so we store each thread entry into a
// thread_local variable.
// ContainerType is the container type used to store the objects, e.g.,
// std::vector, std::set, etc. by each thread. ObjectType is the type of the
// stored objects e.g., omp_interop_val_t *, ...
template <typename ContainerType, typename ObjectType> class PerThreadTable {
  using iterator = typename ContainerConcepts<ContainerType>::iterator;

  struct PerThreadData {
    size_t NElements = 0;
    std::unique_ptr<ContainerType> ThreadEntry;
  };

  std::mutex Mutex;
  llvm::SmallVector<std::shared_ptr<PerThreadData>> ThreadDataList;

  PerThreadData &getThreadData() {
    static thread_local std::shared_ptr<PerThreadData> ThreadData = nullptr;
    if (!ThreadData) {
      ThreadData = std::make_shared<PerThreadData>();
      std::lock_guard<std::mutex> Lock(Mutex);
      ThreadDataList.push_back(ThreadData);
    }
    return *ThreadData;
  }

protected:
  ContainerType &getThreadEntry() {
    PerThreadData &ThreadData = getThreadData();
    if (ThreadData.ThreadEntry)
      return *ThreadData.ThreadEntry;
    ThreadData.ThreadEntry = std::make_unique<ContainerType>();
    return *ThreadData.ThreadEntry;
  }

  size_t &getThreadNElements() {
    PerThreadData &ThreadData = getThreadData();
    return ThreadData.NElements;
  }

  void setNElements(size_t Size) {
    size_t &NElements = getThreadNElements();
    NElements = Size;
  }

public:
  // define default constructors, disable copy and move constructors
  PerThreadTable() = default;
  PerThreadTable(const PerThreadTable &) = delete;
  PerThreadTable(PerThreadTable &&) = delete;
  PerThreadTable &operator=(const PerThreadTable &) = delete;
  PerThreadTable &operator=(PerThreadTable &&) = delete;
  ~PerThreadTable() {
    assert(Mutex.try_lock() && (Mutex.unlock(), true) &&
           "Cannot be deleted while other threads are adding entries");
    ThreadDataList.clear();
  }

  void add(ObjectType obj) {
    ContainerType &Entry = getThreadEntry();
    size_t &NElements = getThreadNElements();
    NElements++;
    Entry.add(obj);
  }

  iterator erase(iterator it) {
    ContainerType &Entry = getThreadEntry();
    size_t &NElements = getThreadNElements();
    NElements--;
    return Entry.erase(it);
  }

  size_t size() { return getThreadNElements(); }

  // Iterators to traverse objects owned by
  // the current thread
  iterator begin() {
    ContainerType &Entry = getThreadEntry();
    return Entry.begin();
  }
  iterator end() {
    ContainerType &Entry = getThreadEntry();
    return Entry.end();
  }

  template <class ClearFuncTy> void clear(ClearFuncTy ClearFunc) {
    assert(Mutex.try_lock() && (Mutex.unlock(), true) &&
           "Clear cannot be called while other threads are adding entries");
    for (std::shared_ptr<PerThreadData> ThreadData : ThreadDataList) {
      if (!ThreadData->ThreadEntry || ThreadData->NElements == 0)
        continue;
      if constexpr (ContainerConcepts<ContainerType>::hasIterator &&
                    ContainerConcepts<ContainerType>::hasClear) {
        for (auto &Obj : *ThreadData->ThreadEntry) {
          if constexpr (ContainerConcepts<ContainerType>::isAssociative) {
            ClearFunc(Obj.second);
          } else {
            ClearFunc(Obj);
          }
        }
        ThreadData->ThreadEntry->clear();
      } else {
        static_assert(true, "Container type not supported");
      }
      ThreadData->NElements = 0;
    }
    ThreadDataList.clear();
  }

  template <class DeinitFuncTy> llvm::Error deinit(DeinitFuncTy DeinitFunc) {
    assert(Mutex.try_lock() && (Mutex.unlock(), true) &&
           "Deinit cannot be called while other threads are adding entries");
    for (std::shared_ptr<PerThreadData> ThreadData : ThreadDataList) {
      if (!ThreadData->ThreadEntry || ThreadData->NElements == 0)
        continue;
      for (auto &Obj : *ThreadData->ThreadEntry) {
        if constexpr (ContainerConcepts<ContainerType>::isAssociative) {
          if (auto Err = DeinitFunc(Obj.second))
            return Err;
        } else {
          if (auto Err = DeinitFunc(Obj))
            return Err;
        }
      }
    }
    return llvm::Error::success();
  }
};

template <typename ContainerType, size_t ReserveSize = 0>
class PerThreadContainer
    : public PerThreadTable<ContainerType, typename ContainerConcepts<
                                               ContainerType>::value_type> {

  using IndexType = typename ContainerConcepts<ContainerType>::key_type;
  using ObjectType = typename ContainerConcepts<ContainerType>::value_type;

public:
  // Get the object for the given index in the current thread
  ObjectType &get(IndexType Index) {
    ContainerType &Entry = this->getThreadEntry();

    // specialized code for vector-like containers
    if constexpr (ContainerConcepts<ContainerType>::hasResize) {
      if (Index >= Entry.size()) {
        if constexpr (ContainerConcepts<ContainerType>::hasReserve &&
                      ReserveSize > 0)
          Entry.reserve(ReserveSize);

        // If the index is out of bounds, try resize the container
        Entry.resize(Index + 1);
      }
    }
    ObjectType &Ret = Entry[Index];
    this->setNElements(Entry.size());
    return Ret;
  }
};

#endif
