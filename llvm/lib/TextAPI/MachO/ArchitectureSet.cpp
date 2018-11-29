//===- llvm/TextAPI/ArchitectureSet.cpp - Architecture Set ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Implements the architecture set.
///
//===----------------------------------------------------------------------===//

#include "llvm/TextAPI/MachO/ArchitectureSet.h"

namespace llvm {
namespace MachO {

ArchitectureSet::ArchitectureSet(const std::vector<Architecture> &Archs)
    : ArchitectureSet() {
  for (auto Arch : Archs) {
    if (Arch == Architecture::unknown)
      continue;
    set(Arch);
  }
}

size_t ArchitectureSet::count() const {
  // popcnt
  size_t Cnt = 0;
  for (unsigned i = 0; i < sizeof(ArchSetType) * 8; ++i)
    if (ArchSet & (1U << i))
      ++Cnt;
  return Cnt;
}

ArchitectureSet::operator std::string() const {
  if (empty())
    return "[(empty)]";

  std::string result;
  auto size = count();
  for (auto arch : *this) {
    result.append(getArchitectureName(arch));
    size -= 1;
    if (size)
      result.append(" ");
  }
  return result;
}

ArchitectureSet::operator std::vector<Architecture>() const {
  std::vector<Architecture> archs;
  for (auto arch : *this) {
    if (arch == Architecture::unknown)
      continue;
    archs.emplace_back(arch);
  }
  return archs;
}

void ArchitectureSet::print(raw_ostream &os) const { os << std::string(*this); }

raw_ostream &operator<<(raw_ostream &os, ArchitectureSet set) {
  set.print(os);
  return os;
}

} // end namespace MachO.
} // end namespace llvm.
