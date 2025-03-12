//===---------------- NVPTXAddrSpace.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// NVPTX address space definition
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NVPTXADDRSPACE_H
#define LLVM_SUPPORT_NVPTXADDRSPACE_H

namespace llvm {
namespace NVPTXAS {

enum AddressSpace : unsigned {
  ADDRESS_SPACE_GENERIC = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_SHARED = 3,
  ADDRESS_SPACE_CONST = 4,
  ADDRESS_SPACE_LOCAL = 5,
  ADDRESS_SPACE_TENSOR = 6,

  ADDRESS_SPACE_PARAM = 101,
};

// According to official PTX Writer's Guide, DWARF debug information should
// contain DW_AT_address_class attribute for all variables and parameters.
// It's required for cuda-gdb to be able to properly reflect the memory space
// of variable address. Acceptable address class codes are listed in this enum.
//
// More detailed information:
// https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html#cuda-specific-dwarf-definitions
enum DWARF_AddressSpace : unsigned {
  DWARF_ADDR_code_space = 1,
  DWARF_ADDR_reg_space = 2,
  DWARF_ADDR_sreg_space = 3,
  DWARF_ADDR_const_space = 4,
  DWARF_ADDR_global_space = 5,
  DWARF_ADDR_local_space = 6,
  DWARF_ADDR_param_space = 7,
  DWARF_ADDR_shared_space = 8,
  DWARF_ADDR_surf_space = 9,
  DWARF_ADDR_tex_space = 10,
  DWARF_ADDR_tex_sampler_space = 11,
  DWARF_ADDR_generic_space = 12
};

} // end namespace NVPTXAS
} // end namespace llvm

#endif // LLVM_SUPPORT_NVPTXADDRSPACE_H
