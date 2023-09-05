//===- ArrayList.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_ARRAYLIST_H
#define LLVM_LIB_DWARFLINKERPARALLEL_ARRAYLIST_H

#include "DWARFLinkerGlobalData.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"

namespace llvm {
namespace dwarflinker_parallel {

/// This class is a simple list of T structures. It keeps elements as
/// pre-allocated groups to save memory for each element's next pointer.
/// It allocates internal data using specified per-thread BumpPtrAllocator.
template <typename T, size_t ItemsGroupSize = 512> class ArrayList {
public:
  /// Copy specified \p Item into the list.
  T &noteItem(const T &Item) {
    assert(Allocator != nullptr);

    ItemsGroup *CurGroup = LastGroup;

    if (CurGroup == nullptr) {
      // Allocate first ItemsGroup.
      LastGroup = Allocator->Allocate<ItemsGroup>();
      LastGroup->ItemsCount = 0;
      LastGroup->Next = nullptr;
      GroupsHead = LastGroup;
      CurGroup = LastGroup;
    }

    if (CurGroup->ItemsCount == ItemsGroupSize) {
      // Allocate next ItemsGroup if current one is full.
      LastGroup = Allocator->Allocate<ItemsGroup>();
      LastGroup->ItemsCount = 0;
      LastGroup->Next = nullptr;
      CurGroup->Next = LastGroup;
      CurGroup = LastGroup;
    }

    // Copy item into the next position inside current ItemsGroup.
    CurGroup->Items[CurGroup->ItemsCount] = Item;
    return CurGroup->Items[CurGroup->ItemsCount++];
  }

  using ItemHandlerTy = function_ref<void(T &)>;

  /// Enumerate all items and apply specified \p Handler to each.
  void forEach(ItemHandlerTy Handler) {
    for (ItemsGroup *CurGroup = GroupsHead; CurGroup != nullptr;
         CurGroup = CurGroup->Next) {
      for (size_t Idx = 0; Idx < CurGroup->ItemsCount; Idx++) {
        Handler(CurGroup->Items[Idx]);
      }
    }
  }

  /// Check whether list is empty.
  bool empty() { return GroupsHead == nullptr; }

  /// Erase list.
  void erase() {
    GroupsHead = nullptr;
    LastGroup = nullptr;
  }

  void setAllocator(parallel::PerThreadBumpPtrAllocator *Allocator) {
    this->Allocator = Allocator;
  }

protected:
  struct ItemsGroup {
    std::array<T, ItemsGroupSize> Items;
    ItemsGroup *Next = nullptr;
    size_t ItemsCount = 0;
  };

  ItemsGroup *GroupsHead = nullptr;
  ItemsGroup *LastGroup = nullptr;
  parallel::PerThreadBumpPtrAllocator *Allocator = nullptr;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_ARRAYLIST_H
