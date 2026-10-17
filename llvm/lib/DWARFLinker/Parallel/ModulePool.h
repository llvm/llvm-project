//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKER_PARALLEL_MODULEPOOL_H
#define LLVM_LIB_DWARFLINKER_PARALLEL_MODULEPOOL_H

#include "TypePool.h"
#include "llvm/ADT/StringMap.h"
#include <limits>
#include <mutex>

namespace llvm {
namespace dwarf_linker {
namespace parallel {

struct SectionDescriptor;

/// Where the DW_TAG_module DIE describing a clang module ended up in the
/// output. Clang imposes a one-definition rule (ODR) on module names regardless
/// of the source language, so a module is described once and every importer
/// resolves to that description.
///
/// A module DIE lands either in the artificial type unit, or in the plain DWARF
/// of the unit which emitted it.
struct ModuleAnchor {
  bool isSet() const { return TypeName != nullptr || Section != nullptr; }

  TypeEntry *TypeName = nullptr;
  SectionDescriptor *Section = nullptr;
  uint64_t LocalOffset = 0;

  /// Priority of the unit which recorded this anchor.
  uint64_t Priority = std::numeric_limits<uint64_t>::max();
};

class ModulePool {
public:
  ModuleAnchor *getOrCreate(StringRef Path) {
    std::lock_guard<std::mutex> Guard(Mutex);

    // The underlying StringMap guarantees the pointer remains stable. The
    // pointee is written while cloning, so it may only be read afterwards.
    return &Anchors[Path];
  }

  void set(StringRef Path, const ModuleAnchor &Location) {
    std::lock_guard<std::mutex> Guard(Mutex);

    // Anchors are keyed by module name while a unit is created per .pcm, so a
    // module built more than once has a unit for each copy. The lowest priority
    // wins, as in the type pool, guaranteeing determinism.
    ModuleAnchor &Anchor = Anchors[Path];
    if (!Anchor.isSet() || Location.Priority < Anchor.Priority)
      Anchor = Location;
  }

private:
  /// Unlike the type pool this is not a per-DIE structure. It is accessed once
  /// for each DW_TAG_module cloned and once for each DW_AT_import cloned.
  std::mutex Mutex;
  StringMap<ModuleAnchor> Anchors;
};

} // end of namespace parallel
} // end of namespace dwarf_linker
} // end of namespace llvm

#endif // LLVM_LIB_DWARFLINKER_PARALLEL_MODULEPOOL_H
