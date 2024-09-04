//===-- CoreFileMemoryRanges.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/CoreFileMemoryRanges.h"

using namespace lldb;
using namespace lldb_private;

Status CoreFileMemoryRanges::FinalizeCoreFileSaveRanges() {
  Status error;
  std::vector<size_t> indexes_to_remove;
  this->Sort();
  for (size_t i = this->GetSize() - 1; i > 0; i--) {
    auto region = this->GetMutableEntryAtIndex(i);
    auto next_region = this->GetMutableEntryAtIndex(i - 1);
    if (next_region->GetRangeEnd() >= region->GetRangeBase() &&
        region->GetRangeBase() <= next_region->GetRangeEnd() &&
        region->data.lldb_permissions == next_region->data.lldb_permissions) {
      const addr_t base =
          std::min(region->GetRangeBase(), next_region->GetRangeBase());
      const addr_t byte_size =
          std::max(region->GetRangeEnd(), next_region->GetRangeEnd()) - base;

      next_region->SetRangeBase(base);
      next_region->SetByteSize(byte_size);

      // Because this is a range data vector, the entry has a base as well
      // as the data contained in the entry. So we have to update both.
      // And llvm::AddressRange isn't mutable so we have to create a new one.
      llvm::AddressRange range(base, base + byte_size);
      const CoreFileMemoryRange core_range = {
          range, next_region->data.lldb_permissions};
      next_region->data = core_range;
      if (!this->Erase(i, i + 1)) {
        error = Status::FromErrorString(
            "Core file memory ranges mutated outside of "
            "CalculateCoreFileSaveRanges");
        return error;
      }
    }
  }

  return error;
}
