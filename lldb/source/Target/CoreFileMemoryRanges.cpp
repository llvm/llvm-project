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

using Entry = CoreFileMemoryRanges::Entry;

static bool Overlaps(const Entry *region_one, const Entry *region_two) {
  return !(region_one->GetRangeEnd() < region_two->GetRangeBase() ||
           region_two->GetRangeEnd() < region_one->GetRangeBase());
}

static bool IntersectHelper(const Entry *region_one, const Entry *region_two) {
  return region_one->GetRangeBase() == region_two->GetRangeEnd() ||
         region_one->GetRangeEnd() == region_two->GetRangeBase();
}

static bool OnlyIntersects(const Entry *region_one, const Entry *region_two) {
  return IntersectHelper(region_one, region_two) ||
         IntersectHelper(region_two, region_one);
}

static bool PermissionsMatch(const Entry *region_one, const Entry *region_two) {
  return region_one->data.lldb_permissions == region_two->data.lldb_permissions;
}

// This assumes any overlapping ranges will share the same permissions
// and that adjacent ranges could have different permissions.
Status CoreFileMemoryRanges::FinalizeCoreFileSaveRanges() {
  Status error;
  this->Sort();
  for (size_t i = this->GetSize() - 1; i > 0; i--) {
    auto region_one = this->GetMutableEntryAtIndex(i);
    auto region_two = this->GetMutableEntryAtIndex(i - 1);
    if (Overlaps(region_one, region_two)) {
      // It's okay for interesecting regions to have different permissions but
      // if they overlap we fail because we don't know what to do with them.
      if (!PermissionsMatch(region_one, region_two)) {
        // Permissions mismatch and it's not a simple intersection.
        if (!OnlyIntersects(region_one, region_two)) {
          error = Status::FromErrorStringWithFormatv(
              "Memory region at {0}::{1} has different permssions than "
              "overlapping region at {2}::{3}",
              region_one->GetRangeBase(), region_one->GetRangeEnd(),
              region_two->GetRangeBase(), region_two->GetRangeEnd());
          return error;
        }
        // Simple intersection, we can just not merge these.
        else
          continue;
      }
      const addr_t base =
          std::min(region_one->GetRangeBase(), region_two->GetRangeBase());
      const addr_t byte_size =
          std::max(region_one->GetRangeEnd(), region_two->GetRangeEnd()) - base;

      region_two->SetRangeBase(base);
      region_two->SetByteSize(byte_size);

      // Because this is a range data vector, the entry has a base as well
      // as the data contained in the entry. So we have to update both.
      // And llvm::AddressRange isn't mutable so we have to create a new one.
      llvm::AddressRange range(base, base + byte_size);
      const CoreFileMemoryRange core_range = {
          range, region_two->data.lldb_permissions};
      region_two->data = core_range;
      // Erase is delete from [Inclusive, exclusive index).
      if (!this->Erase(i, i + 1)) {
        error = Status::FromErrorStringWithFormat(
            "Core file memory ranges mutated outside of "
            "CalculateCoreFileSaveRanges");
        return error;
      }
    }
  }

  return error;
}
