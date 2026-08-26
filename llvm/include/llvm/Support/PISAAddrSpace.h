//===-- PISAAddrSpace.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PISAADDRSPACE_H
#define LLVM_SUPPORT_PISAADDRSPACE_H

namespace llvm {

namespace PISAAS {
// See the PISA address-space specification:
// https://intel.github.io/pisa/spaces_types.html#address-spaces
enum class AddressSpace : unsigned {
  GENERIC = 0,
  GLOBAL = 1,
  CONSTANT = 2,
  SHARED = 3,
  PRIVATE = 4,
};

// DWARFAddressSpace for PISA, this will be emitted as DW_AT_address_class
// attribute for variables and parameters.
enum class DWARF_AddressSpace : unsigned {
  DWARF_ADDR_global_shared = 0,
  DWARF_ADDR_shared_local = 1,
  DWARF_ADDR_private = 2,
};

constexpr int mapToDWARFAddrSpace(unsigned LLVMAddrSpace) {
  int dwarfAddrSpace = -1;

  switch (static_cast<AddressSpace>(LLVMAddrSpace)) {
  case AddressSpace::PRIVATE:
    dwarfAddrSpace = static_cast<int>(DWARF_AddressSpace::DWARF_ADDR_private);
    break;
  case AddressSpace::GLOBAL:
  case AddressSpace::CONSTANT:
    dwarfAddrSpace =
        static_cast<int>(DWARF_AddressSpace::DWARF_ADDR_global_shared);
    break;
  case AddressSpace::SHARED:
    dwarfAddrSpace =
        static_cast<int>(DWARF_AddressSpace::DWARF_ADDR_shared_local);
    break;
  default:
    // default is generic space, do not emit anything
    break;
  }

  return dwarfAddrSpace;
}

} // end namespace PISAAS

} // end namespace llvm

#endif // LLVM_SUPPORT_PISAADDRSPACE_H
