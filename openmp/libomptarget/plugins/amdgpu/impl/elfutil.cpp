/*===--------------------------------------------------------------------------
 *              ATMI (Asynchronous Task and Memory Interface)
 *
 * This file is distributed under the MIT License. See LICENSE.txt for details.
 *===------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "llvm/BinaryFormat/ELF.h"
#include "omptarget.h"
#include "Debug.h"
#include "elf_common.h"

#ifndef TARGET_NAME
#define TARGET_NAME AMDHSA
#endif
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"

using namespace llvm;
using namespace ELF;

int get_elf_mach_gfx(__tgt_device_image *image) {
  uint32_t EFlags = elf_e_flags(image);
  uint32_t Gfx = (EFlags & EF_AMDGPU_MACH);
  return Gfx;
}

char* get_elf_mach_gfx_name(__tgt_device_image *image) {
  uint32_t Gfx = get_elf_mach_gfx(image);
  switch  (Gfx) {
  case EF_AMDGPU_MACH_AMDGCN_GFX801 :  return "gfx801" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX802 :  return "gfx802" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX803 :  return "gfx803" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX805 :  return "gfx805" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX810 :  return "gfx810" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX900 :  return "gfx900" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX902 :  return "gfx902" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX904 :  return "gfx904" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX906 :  return "gfx906" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX908 :  return "gfx908" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX909 :  return "gfx909" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX90C :  return "gfx90c" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1010 :  return "gfx1010" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1011 :  return "gfx1011" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1012 :  return "gfx1012" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1030 :  return "gfx1030" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1031 :  return "gfx1031" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1032 :  return "gfx1032" ;
  case EF_AMDGPU_MACH_AMDGCN_GFX1033 :  return "gfx1033" ;
  default: return "--unknown gfx";
  }
}

bool elf_machine_id_is_amdgcn(__tgt_device_image *image) {
  const uint16_t amdgcnMachineID = EM_AMDGPU;
  int32_t r = elf_check_machine(image, amdgcnMachineID);
  if (!r) {
    DP("Supported machine ID not found\n");
  }
  return r;
}

