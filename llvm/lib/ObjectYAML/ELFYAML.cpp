//===- ELFYAML.cpp - ELF YAMLIO implementation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of ELF.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MipsABIFlags.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/WithColor.h"
#include <cassert>
#include <cstdint>

namespace llvm {

ELFYAML::Chunk::~Chunk() = default;

namespace yaml {

void ScalarEnumerationTraits<ELFYAML::ELF_ET>::enumeration(
    IO &IO, ELFYAML::ELF_ET &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(ET_NONE);
  ECase(ET_REL);
  ECase(ET_EXEC);
  ECase(ET_DYN);
  ECase(ET_CORE);
#undef ECase
  IO.enumFallback<Hex16>(Value);
}

void ScalarEnumerationTraits<ELFYAML::ELF_PT>::enumeration(
    IO &IO, ELFYAML::ELF_PT &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(PT_NULL);
  ECase(PT_LOAD);
  ECase(PT_DYNAMIC);
  ECase(PT_INTERP);
  ECase(PT_NOTE);
  ECase(PT_SHLIB);
  ECase(PT_PHDR);
  ECase(PT_TLS);
  ECase(PT_GNU_EH_FRAME);
  ECase(PT_GNU_STACK);
  ECase(PT_GNU_RELRO);
  ECase(PT_GNU_PROPERTY);
#undef ECase
  IO.enumFallback<Hex32>(Value);
}

void ScalarEnumerationTraits<ELFYAML::ELF_EM>::enumeration(
    IO &IO, ELFYAML::ELF_EM &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(EM_NONE);
  ECase(EM_M32);
  ECase(EM_SPARC);
  ECase(EM_386);
  ECase(EM_68K);
  ECase(EM_88K);
  ECase(EM_IAMCU);
  ECase(EM_860);
  ECase(EM_MIPS);
  ECase(EM_S370);
  ECase(EM_MIPS_RS3_LE);
  ECase(EM_PARISC);
  ECase(EM_VPP500);
  ECase(EM_SPARC32PLUS);
  ECase(EM_960);
  ECase(EM_PPC);
  ECase(EM_PPC64);
  ECase(EM_S390);
  ECase(EM_SPU);
  ECase(EM_V800);
  ECase(EM_FR20);
  ECase(EM_RH32);
  ECase(EM_RCE);
  ECase(EM_ARM);
  ECase(EM_ALPHA);
  ECase(EM_SH);
  ECase(EM_SPARCV9);
  ECase(EM_TRICORE);
  ECase(EM_ARC);
  ECase(EM_H8_300);
  ECase(EM_H8_300H);
  ECase(EM_H8S);
  ECase(EM_H8_500);
  ECase(EM_IA_64);
  ECase(EM_MIPS_X);
  ECase(EM_COLDFIRE);
  ECase(EM_68HC12);
  ECase(EM_MMA);
  ECase(EM_PCP);
  ECase(EM_NCPU);
  ECase(EM_NDR1);
  ECase(EM_STARCORE);
  ECase(EM_ME16);
  ECase(EM_ST100);
  ECase(EM_TINYJ);
  ECase(EM_X86_64);
  ECase(EM_PDSP);
  ECase(EM_PDP10);
  ECase(EM_PDP11);
  ECase(EM_FX66);
  ECase(EM_ST9PLUS);
  ECase(EM_ST7);
  ECase(EM_68HC16);
  ECase(EM_68HC11);
  ECase(EM_68HC08);
  ECase(EM_68HC05);
  ECase(EM_SVX);
  ECase(EM_ST19);
  ECase(EM_VAX);
  ECase(EM_CRIS);
  ECase(EM_JAVELIN);
  ECase(EM_FIREPATH);
  ECase(EM_ZSP);
  ECase(EM_MMIX);
  ECase(EM_HUANY);
  ECase(EM_PRISM);
  ECase(EM_AVR);
  ECase(EM_FR30);
  ECase(EM_D10V);
  ECase(EM_D30V);
  ECase(EM_V850);
  ECase(EM_M32R);
  ECase(EM_MN10300);
  ECase(EM_MN10200);
  ECase(EM_PJ);
  ECase(EM_OPENRISC);
  ECase(EM_ARC_COMPACT);
  ECase(EM_XTENSA);
  ECase(EM_VIDEOCORE);
  ECase(EM_TMM_GPP);
  ECase(EM_NS32K);
  ECase(EM_TPC);
  ECase(EM_SNP1K);
  ECase(EM_ST200);
  ECase(EM_IP2K);
  ECase(EM_MAX);
  ECase(EM_CR);
  ECase(EM_F2MC16);
  ECase(EM_MSP430);
  ECase(EM_BLACKFIN);
  ECase(EM_SE_C33);
  ECase(EM_SEP);
  ECase(EM_ARCA);
  ECase(EM_UNICORE);
  ECase(EM_EXCESS);
  ECase(EM_DXP);
  ECase(EM_ALTERA_NIOS2);
  ECase(EM_CRX);
  ECase(EM_XGATE);
  ECase(EM_C166);
  ECase(EM_M16C);
  ECase(EM_DSPIC30F);
  ECase(EM_CE);
  ECase(EM_M32C);
  ECase(EM_TSK3000);
  ECase(EM_RS08);
  ECase(EM_SHARC);
  ECase(EM_ECOG2);
  ECase(EM_SCORE7);
  ECase(EM_DSP24);
  ECase(EM_VIDEOCORE3);
  ECase(EM_LATTICEMICO32);
  ECase(EM_SE_C17);
  ECase(EM_TI_C6000);
  ECase(EM_TI_C2000);
  ECase(EM_TI_C5500);
  ECase(EM_MMDSP_PLUS);
  ECase(EM_CYPRESS_M8C);
  ECase(EM_R32C);
  ECase(EM_TRIMEDIA);
  ECase(EM_HEXAGON);
  ECase(EM_8051);
  ECase(EM_STXP7X);
  ECase(EM_NDS32);
  ECase(EM_ECOG1);
  ECase(EM_ECOG1X);
  ECase(EM_MAXQ30);
  ECase(EM_XIMO16);
  ECase(EM_MANIK);
  ECase(EM_CRAYNV2);
  ECase(EM_RX);
  ECase(EM_METAG);
  ECase(EM_MCST_ELBRUS);
  ECase(EM_ECOG16);
  ECase(EM_CR16);
  ECase(EM_ETPU);
  ECase(EM_SLE9X);
  ECase(EM_L10M);
  ECase(EM_K10M);
  ECase(EM_AARCH64);
  ECase(EM_AVR32);
  ECase(EM_STM8);
  ECase(EM_TILE64);
  ECase(EM_TILEPRO);
  ECase(EM_CUDA);
  ECase(EM_TILEGX);
  ECase(EM_CLOUDSHIELD);
  ECase(EM_COREA_1ST);
  ECase(EM_COREA_2ND);
  ECase(EM_ARC_COMPACT2);
  ECase(EM_OPEN8);
  ECase(EM_RL78);
  ECase(EM_VIDEOCORE5);
  ECase(EM_78KOR);
  ECase(EM_56800EX);
  ECase(EM_AMDGPU);
  ECase(EM_RISCV);
  ECase(EM_LANAI);
  ECase(EM_BPF);
#undef ECase
  IO.enumFallback<Hex16>(Value);
}

void ScalarEnumerationTraits<ELFYAML::ELF_ELFCLASS>::enumeration(
    IO &IO, ELFYAML::ELF_ELFCLASS &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  // Since the semantics of ELFCLASSNONE is "invalid", just don't accept it
  // here.
  ECase(ELFCLASS32);
  ECase(ELFCLASS64);
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_ELFDATA>::enumeration(
    IO &IO, ELFYAML::ELF_ELFDATA &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  // ELFDATANONE is an invalid data encoding, but we accept it because
  // we want to be able to produce invalid binaries for the tests.
  ECase(ELFDATANONE);
  ECase(ELFDATA2LSB);
  ECase(ELFDATA2MSB);
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_ELFOSABI>::enumeration(
    IO &IO, ELFYAML::ELF_ELFOSABI &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(ELFOSABI_NONE);
  ECase(ELFOSABI_HPUX);
  ECase(ELFOSABI_NETBSD);
  ECase(ELFOSABI_GNU);
  ECase(ELFOSABI_LINUX);
  ECase(ELFOSABI_HURD);
  ECase(ELFOSABI_SOLARIS);
  ECase(ELFOSABI_AIX);
  ECase(ELFOSABI_IRIX);
  ECase(ELFOSABI_FREEBSD);
  ECase(ELFOSABI_TRU64);
  ECase(ELFOSABI_MODESTO);
  ECase(ELFOSABI_OPENBSD);
  ECase(ELFOSABI_OPENVMS);
  ECase(ELFOSABI_NSK);
  ECase(ELFOSABI_AROS);
  ECase(ELFOSABI_FENIXOS);
  ECase(ELFOSABI_CLOUDABI);
  ECase(ELFOSABI_AMDGPU_HSA);
  ECase(ELFOSABI_AMDGPU_PAL);
  ECase(ELFOSABI_AMDGPU_MESA3D);
  ECase(ELFOSABI_ARM);
  ECase(ELFOSABI_C6000_ELFABI);
  ECase(ELFOSABI_C6000_LINUX);
  ECase(ELFOSABI_STANDALONE);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarBitSetTraits<ELFYAML::ELF_EF>::bitset(IO &IO,
                                                 ELFYAML::ELF_EF &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");
#define BCase(X) IO.bitSetCase(Value, #X, ELF::X)
#define BCaseMask(X, M) IO.maskedBitSetCase(Value, #X, ELF::X, ELF::M)
  switch (Object->Header.Machine) {
  case ELF::EM_ARM:
    BCase(EF_ARM_SOFT_FLOAT);
    BCase(EF_ARM_VFP_FLOAT);
    BCaseMask(EF_ARM_EABI_UNKNOWN, EF_ARM_EABIMASK);
    BCaseMask(EF_ARM_EABI_VER1, EF_ARM_EABIMASK);
    BCaseMask(EF_ARM_EABI_VER2, EF_ARM_EABIMASK);
    BCaseMask(EF_ARM_EABI_VER3, EF_ARM_EABIMASK);
    BCaseMask(EF_ARM_EABI_VER4, EF_ARM_EABIMASK);
    BCaseMask(EF_ARM_EABI_VER5, EF_ARM_EABIMASK);
    break;
  case ELF::EM_MIPS:
    BCase(EF_MIPS_NOREORDER);
    BCase(EF_MIPS_PIC);
    BCase(EF_MIPS_CPIC);
    BCase(EF_MIPS_ABI2);
    BCase(EF_MIPS_32BITMODE);
    BCase(EF_MIPS_FP64);
    BCase(EF_MIPS_NAN2008);
    BCase(EF_MIPS_MICROMIPS);
    BCase(EF_MIPS_ARCH_ASE_M16);
    BCase(EF_MIPS_ARCH_ASE_MDMX);
    BCaseMask(EF_MIPS_ABI_O32, EF_MIPS_ABI);
    BCaseMask(EF_MIPS_ABI_O64, EF_MIPS_ABI);
    BCaseMask(EF_MIPS_ABI_EABI32, EF_MIPS_ABI);
    BCaseMask(EF_MIPS_ABI_EABI64, EF_MIPS_ABI);
    BCaseMask(EF_MIPS_MACH_3900, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_4010, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_4100, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_4650, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_4120, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_4111, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_SB1, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_OCTEON, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_XLR, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_OCTEON2, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_OCTEON3, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_5400, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_5900, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_5500, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_9000, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_LS2E, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_LS2F, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_MACH_LS3A, EF_MIPS_MACH);
    BCaseMask(EF_MIPS_ARCH_1, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_2, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_3, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_4, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_5, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_32, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_64, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_32R2, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_64R2, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_32R6, EF_MIPS_ARCH);
    BCaseMask(EF_MIPS_ARCH_64R6, EF_MIPS_ARCH);
    break;
  case ELF::EM_HEXAGON:
    BCase(EF_HEXAGON_MACH_V2);
    BCase(EF_HEXAGON_MACH_V3);
    BCase(EF_HEXAGON_MACH_V4);
    BCase(EF_HEXAGON_MACH_V5);
    BCase(EF_HEXAGON_MACH_V55);
    BCase(EF_HEXAGON_MACH_V60);
    BCase(EF_HEXAGON_MACH_V62);
    BCase(EF_HEXAGON_MACH_V65);
    BCase(EF_HEXAGON_ISA_V2);
    BCase(EF_HEXAGON_ISA_V3);
    BCase(EF_HEXAGON_ISA_V4);
    BCase(EF_HEXAGON_ISA_V5);
    BCase(EF_HEXAGON_ISA_V55);
    BCase(EF_HEXAGON_ISA_V60);
    BCase(EF_HEXAGON_ISA_V62);
    BCase(EF_HEXAGON_ISA_V65);
    break;
  case ELF::EM_AVR:
    BCase(EF_AVR_ARCH_AVR1);
    BCase(EF_AVR_ARCH_AVR2);
    BCase(EF_AVR_ARCH_AVR25);
    BCase(EF_AVR_ARCH_AVR3);
    BCase(EF_AVR_ARCH_AVR31);
    BCase(EF_AVR_ARCH_AVR35);
    BCase(EF_AVR_ARCH_AVR4);
    BCase(EF_AVR_ARCH_AVR51);
    BCase(EF_AVR_ARCH_AVR6);
    BCase(EF_AVR_ARCH_AVRTINY);
    BCase(EF_AVR_ARCH_XMEGA1);
    BCase(EF_AVR_ARCH_XMEGA2);
    BCase(EF_AVR_ARCH_XMEGA3);
    BCase(EF_AVR_ARCH_XMEGA4);
    BCase(EF_AVR_ARCH_XMEGA5);
    BCase(EF_AVR_ARCH_XMEGA6);
    BCase(EF_AVR_ARCH_XMEGA7);
    break;
  case ELF::EM_RISCV:
    BCase(EF_RISCV_RVC);
    BCaseMask(EF_RISCV_FLOAT_ABI_SOFT, EF_RISCV_FLOAT_ABI);
    BCaseMask(EF_RISCV_FLOAT_ABI_SINGLE, EF_RISCV_FLOAT_ABI);
    BCaseMask(EF_RISCV_FLOAT_ABI_DOUBLE, EF_RISCV_FLOAT_ABI);
    BCaseMask(EF_RISCV_FLOAT_ABI_QUAD, EF_RISCV_FLOAT_ABI);
    BCase(EF_RISCV_RVE);
    break;
  case ELF::EM_AMDGPU:
    BCaseMask(EF_AMDGPU_MACH_NONE, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_R600, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_R630, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_RS880, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_RV670, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_RV710, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_RV730, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_RV770, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_CEDAR, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_CYPRESS, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_JUNIPER, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_REDWOOD, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_SUMO, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_BARTS, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_CAICOS, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_CAYMAN, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_R600_TURKS, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX600, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX601, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX700, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX701, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX702, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX703, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX704, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX801, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX802, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX803, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX810, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX900, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX902, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX904, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX906, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX908, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX909, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX1010, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX1011, EF_AMDGPU_MACH);
    BCaseMask(EF_AMDGPU_MACH_AMDGCN_GFX1012, EF_AMDGPU_MACH);
    BCase(EF_AMDGPU_XNACK);
    BCase(EF_AMDGPU_SRAM_ECC);
    break;
  case ELF::EM_X86_64:
    break;
  default:
    llvm_unreachable("Unsupported architecture");
  }
#undef BCase
#undef BCaseMask
}

void ScalarEnumerationTraits<ELFYAML::ELF_SHT>::enumeration(
    IO &IO, ELFYAML::ELF_SHT &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(SHT_NULL);
  ECase(SHT_PROGBITS);
  ECase(SHT_SYMTAB);
  // FIXME: Issue a diagnostic with this information.
  ECase(SHT_STRTAB);
  ECase(SHT_RELA);
  ECase(SHT_HASH);
  ECase(SHT_DYNAMIC);
  ECase(SHT_NOTE);
  ECase(SHT_NOBITS);
  ECase(SHT_REL);
  ECase(SHT_SHLIB);
  ECase(SHT_DYNSYM);
  ECase(SHT_INIT_ARRAY);
  ECase(SHT_FINI_ARRAY);
  ECase(SHT_PREINIT_ARRAY);
  ECase(SHT_GROUP);
  ECase(SHT_SYMTAB_SHNDX);
  ECase(SHT_RELR);
  ECase(SHT_ANDROID_REL);
  ECase(SHT_ANDROID_RELA);
  ECase(SHT_ANDROID_RELR);
  ECase(SHT_LLVM_ODRTAB);
  ECase(SHT_LLVM_LINKER_OPTIONS);
  ECase(SHT_LLVM_CALL_GRAPH_PROFILE);
  ECase(SHT_LLVM_ADDRSIG);
  ECase(SHT_LLVM_DEPENDENT_LIBRARIES);
  ECase(SHT_LLVM_SYMPART);
  ECase(SHT_LLVM_PART_EHDR);
  ECase(SHT_LLVM_PART_PHDR);
  ECase(SHT_GNU_ATTRIBUTES);
  ECase(SHT_GNU_HASH);
  ECase(SHT_GNU_verdef);
  ECase(SHT_GNU_verneed);
  ECase(SHT_GNU_versym);
  switch (Object->Header.Machine) {
  case ELF::EM_ARM:
    ECase(SHT_ARM_EXIDX);
    ECase(SHT_ARM_PREEMPTMAP);
    ECase(SHT_ARM_ATTRIBUTES);
    ECase(SHT_ARM_DEBUGOVERLAY);
    ECase(SHT_ARM_OVERLAYSECTION);
    break;
  case ELF::EM_HEXAGON:
    ECase(SHT_HEX_ORDERED);
    break;
  case ELF::EM_X86_64:
    ECase(SHT_X86_64_UNWIND);
    break;
  case ELF::EM_MIPS:
    ECase(SHT_MIPS_REGINFO);
    ECase(SHT_MIPS_OPTIONS);
    ECase(SHT_MIPS_DWARF);
    ECase(SHT_MIPS_ABIFLAGS);
    break;
  default:
    // Nothing to do.
    break;
  }
#undef ECase
  IO.enumFallback<Hex32>(Value);
}

void ScalarBitSetTraits<ELFYAML::ELF_PF>::bitset(IO &IO,
                                                 ELFYAML::ELF_PF &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, ELF::X)
  BCase(PF_X);
  BCase(PF_W);
  BCase(PF_R);
}

void ScalarBitSetTraits<ELFYAML::ELF_SHF>::bitset(IO &IO,
                                                  ELFYAML::ELF_SHF &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
#define BCase(X) IO.bitSetCase(Value, #X, ELF::X)
  BCase(SHF_WRITE);
  BCase(SHF_ALLOC);
  BCase(SHF_EXCLUDE);
  BCase(SHF_EXECINSTR);
  BCase(SHF_MERGE);
  BCase(SHF_STRINGS);
  BCase(SHF_INFO_LINK);
  BCase(SHF_LINK_ORDER);
  BCase(SHF_OS_NONCONFORMING);
  BCase(SHF_GROUP);
  BCase(SHF_TLS);
  BCase(SHF_COMPRESSED);
  switch (Object->Header.Machine) {
  case ELF::EM_ARM:
    BCase(SHF_ARM_PURECODE);
    break;
  case ELF::EM_HEXAGON:
    BCase(SHF_HEX_GPREL);
    break;
  case ELF::EM_MIPS:
    BCase(SHF_MIPS_NODUPES);
    BCase(SHF_MIPS_NAMES);
    BCase(SHF_MIPS_LOCAL);
    BCase(SHF_MIPS_NOSTRIP);
    BCase(SHF_MIPS_GPREL);
    BCase(SHF_MIPS_MERGE);
    BCase(SHF_MIPS_ADDR);
    BCase(SHF_MIPS_STRING);
    break;
  case ELF::EM_X86_64:
    BCase(SHF_X86_64_LARGE);
    break;
  default:
    // Nothing to do.
    break;
  }
#undef BCase
}

void ScalarEnumerationTraits<ELFYAML::ELF_SHN>::enumeration(
    IO &IO, ELFYAML::ELF_SHN &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(SHN_UNDEF);
  ECase(SHN_LORESERVE);
  ECase(SHN_LOPROC);
  ECase(SHN_HIPROC);
  ECase(SHN_LOOS);
  ECase(SHN_HIOS);
  ECase(SHN_ABS);
  ECase(SHN_COMMON);
  ECase(SHN_XINDEX);
  ECase(SHN_HIRESERVE);
  ECase(SHN_AMDGPU_LDS);
  ECase(SHN_HEXAGON_SCOMMON);
  ECase(SHN_HEXAGON_SCOMMON_1);
  ECase(SHN_HEXAGON_SCOMMON_2);
  ECase(SHN_HEXAGON_SCOMMON_4);
  ECase(SHN_HEXAGON_SCOMMON_8);
#undef ECase
  IO.enumFallback<Hex16>(Value);
}

void ScalarEnumerationTraits<ELFYAML::ELF_STB>::enumeration(
    IO &IO, ELFYAML::ELF_STB &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(STB_LOCAL);
  ECase(STB_GLOBAL);
  ECase(STB_WEAK);
  ECase(STB_GNU_UNIQUE);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}

void ScalarEnumerationTraits<ELFYAML::ELF_STT>::enumeration(
    IO &IO, ELFYAML::ELF_STT &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(STT_NOTYPE);
  ECase(STT_OBJECT);
  ECase(STT_FUNC);
  ECase(STT_SECTION);
  ECase(STT_FILE);
  ECase(STT_COMMON);
  ECase(STT_TLS);
  ECase(STT_GNU_IFUNC);
#undef ECase
  IO.enumFallback<Hex8>(Value);
}


void ScalarEnumerationTraits<ELFYAML::ELF_RSS>::enumeration(
    IO &IO, ELFYAML::ELF_RSS &Value) {
#define ECase(X) IO.enumCase(Value, #X, ELF::X)
  ECase(RSS_UNDEF);
  ECase(RSS_GP);
  ECase(RSS_GP0);
  ECase(RSS_LOC);
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::ELF_REL>::enumeration(
    IO &IO, ELFYAML::ELF_REL &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");
#define ELF_RELOC(X, Y) IO.enumCase(Value, #X, ELF::X);
  switch (Object->Header.Machine) {
  case ELF::EM_X86_64:
#include "llvm/BinaryFormat/ELFRelocs/x86_64.def"
    break;
  case ELF::EM_MIPS:
#include "llvm/BinaryFormat/ELFRelocs/Mips.def"
    break;
  case ELF::EM_HEXAGON:
#include "llvm/BinaryFormat/ELFRelocs/Hexagon.def"
    break;
  case ELF::EM_386:
  case ELF::EM_IAMCU:
#include "llvm/BinaryFormat/ELFRelocs/i386.def"
    break;
  case ELF::EM_AARCH64:
#include "llvm/BinaryFormat/ELFRelocs/AArch64.def"
    break;
  case ELF::EM_ARM:
#include "llvm/BinaryFormat/ELFRelocs/ARM.def"
    break;
  case ELF::EM_ARC:
#include "llvm/BinaryFormat/ELFRelocs/ARC.def"
    break;
  case ELF::EM_RISCV:
#include "llvm/BinaryFormat/ELFRelocs/RISCV.def"
    break;
  case ELF::EM_LANAI:
#include "llvm/BinaryFormat/ELFRelocs/Lanai.def"
    break;
  case ELF::EM_AMDGPU:
#include "llvm/BinaryFormat/ELFRelocs/AMDGPU.def"
    break;
  case ELF::EM_BPF:
#include "llvm/BinaryFormat/ELFRelocs/BPF.def"
    break;
  case ELF::EM_PPC64:
#include "llvm/BinaryFormat/ELFRelocs/PowerPC64.def"
    break;
  default:
    // Nothing to do.
    break;
  }
#undef ELF_RELOC
  IO.enumFallback<Hex32>(Value);
}

void ScalarEnumerationTraits<ELFYAML::ELF_DYNTAG>::enumeration(
    IO &IO, ELFYAML::ELF_DYNTAG &Value) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");

// Disable architecture specific tags by default. We might enable them below.
#define AARCH64_DYNAMIC_TAG(name, value)
#define MIPS_DYNAMIC_TAG(name, value)
#define HEXAGON_DYNAMIC_TAG(name, value)
#define PPC_DYNAMIC_TAG(name, value)
#define PPC64_DYNAMIC_TAG(name, value)
// Ignore marker tags such as DT_HIOS (maps to DT_VERNEEDNUM), etc.
#define DYNAMIC_TAG_MARKER(name, value)

#define STRINGIFY(X) (#X)
#define DYNAMIC_TAG(X, Y) IO.enumCase(Value, STRINGIFY(DT_##X), ELF::DT_##X);
  switch (Object->Header.Machine) {
  case ELF::EM_AARCH64:
#undef AARCH64_DYNAMIC_TAG
#define AARCH64_DYNAMIC_TAG(name, value) DYNAMIC_TAG(name, value)
#include "llvm/BinaryFormat/DynamicTags.def"
#undef AARCH64_DYNAMIC_TAG
#define AARCH64_DYNAMIC_TAG(name, value)
    break;
  case ELF::EM_MIPS:
#undef MIPS_DYNAMIC_TAG
#define MIPS_DYNAMIC_TAG(name, value) DYNAMIC_TAG(name, value)
#include "llvm/BinaryFormat/DynamicTags.def"
#undef MIPS_DYNAMIC_TAG
#define MIPS_DYNAMIC_TAG(name, value)
    break;
  case ELF::EM_HEXAGON:
#undef HEXAGON_DYNAMIC_TAG
#define HEXAGON_DYNAMIC_TAG(name, value) DYNAMIC_TAG(name, value)
#include "llvm/BinaryFormat/DynamicTags.def"
#undef HEXAGON_DYNAMIC_TAG
#define HEXAGON_DYNAMIC_TAG(name, value)
    break;
  case ELF::EM_PPC:
#undef PPC_DYNAMIC_TAG
#define PPC_DYNAMIC_TAG(name, value) DYNAMIC_TAG(name, value)
#include "llvm/BinaryFormat/DynamicTags.def"
#undef PPC_DYNAMIC_TAG
#define PPC_DYNAMIC_TAG(name, value)
    break;
  case ELF::EM_PPC64:
#undef PPC64_DYNAMIC_TAG
#define PPC64_DYNAMIC_TAG(name, value) DYNAMIC_TAG(name, value)
#include "llvm/BinaryFormat/DynamicTags.def"
#undef PPC64_DYNAMIC_TAG
#define PPC64_DYNAMIC_TAG(name, value)
    break;
  default:
#include "llvm/BinaryFormat/DynamicTags.def"
    break;
  }
#undef AARCH64_DYNAMIC_TAG
#undef MIPS_DYNAMIC_TAG
#undef HEXAGON_DYNAMIC_TAG
#undef PPC_DYNAMIC_TAG
#undef PPC64_DYNAMIC_TAG
#undef DYNAMIC_TAG_MARKER
#undef STRINGIFY
#undef DYNAMIC_TAG

  IO.enumFallback<Hex64>(Value);
}

void ScalarEnumerationTraits<ELFYAML::MIPS_AFL_REG>::enumeration(
    IO &IO, ELFYAML::MIPS_AFL_REG &Value) {
#define ECase(X) IO.enumCase(Value, #X, Mips::AFL_##X)
  ECase(REG_NONE);
  ECase(REG_32);
  ECase(REG_64);
  ECase(REG_128);
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::MIPS_ABI_FP>::enumeration(
    IO &IO, ELFYAML::MIPS_ABI_FP &Value) {
#define ECase(X) IO.enumCase(Value, #X, Mips::Val_GNU_MIPS_ABI_##X)
  ECase(FP_ANY);
  ECase(FP_DOUBLE);
  ECase(FP_SINGLE);
  ECase(FP_SOFT);
  ECase(FP_OLD_64);
  ECase(FP_XX);
  ECase(FP_64);
  ECase(FP_64A);
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::MIPS_AFL_EXT>::enumeration(
    IO &IO, ELFYAML::MIPS_AFL_EXT &Value) {
#define ECase(X) IO.enumCase(Value, #X, Mips::AFL_##X)
  ECase(EXT_NONE);
  ECase(EXT_XLR);
  ECase(EXT_OCTEON2);
  ECase(EXT_OCTEONP);
  ECase(EXT_LOONGSON_3A);
  ECase(EXT_OCTEON);
  ECase(EXT_5900);
  ECase(EXT_4650);
  ECase(EXT_4010);
  ECase(EXT_4100);
  ECase(EXT_3900);
  ECase(EXT_10000);
  ECase(EXT_SB1);
  ECase(EXT_4111);
  ECase(EXT_4120);
  ECase(EXT_5400);
  ECase(EXT_5500);
  ECase(EXT_LOONGSON_2E);
  ECase(EXT_LOONGSON_2F);
  ECase(EXT_OCTEON3);
#undef ECase
}

void ScalarEnumerationTraits<ELFYAML::MIPS_ISA>::enumeration(
    IO &IO, ELFYAML::MIPS_ISA &Value) {
  IO.enumCase(Value, "MIPS1", 1);
  IO.enumCase(Value, "MIPS2", 2);
  IO.enumCase(Value, "MIPS3", 3);
  IO.enumCase(Value, "MIPS4", 4);
  IO.enumCase(Value, "MIPS5", 5);
  IO.enumCase(Value, "MIPS32", 32);
  IO.enumCase(Value, "MIPS64", 64);
}

void ScalarBitSetTraits<ELFYAML::MIPS_AFL_ASE>::bitset(
    IO &IO, ELFYAML::MIPS_AFL_ASE &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, Mips::AFL_ASE_##X)
  BCase(DSP);
  BCase(DSPR2);
  BCase(EVA);
  BCase(MCU);
  BCase(MDMX);
  BCase(MIPS3D);
  BCase(MT);
  BCase(SMARTMIPS);
  BCase(VIRT);
  BCase(MSA);
  BCase(MIPS16);
  BCase(MICROMIPS);
  BCase(XPA);
#undef BCase
}

void ScalarBitSetTraits<ELFYAML::MIPS_AFL_FLAGS1>::bitset(
    IO &IO, ELFYAML::MIPS_AFL_FLAGS1 &Value) {
#define BCase(X) IO.bitSetCase(Value, #X, Mips::AFL_FLAGS1_##X)
  BCase(ODDSPREG);
#undef BCase
}

void MappingTraits<ELFYAML::FileHeader>::mapping(IO &IO,
                                                 ELFYAML::FileHeader &FileHdr) {
  IO.mapRequired("Class", FileHdr.Class);
  IO.mapRequired("Data", FileHdr.Data);
  IO.mapOptional("OSABI", FileHdr.OSABI, ELFYAML::ELF_ELFOSABI(0));
  IO.mapOptional("ABIVersion", FileHdr.ABIVersion, Hex8(0));
  IO.mapRequired("Type", FileHdr.Type);
  IO.mapRequired("Machine", FileHdr.Machine);
  IO.mapOptional("Flags", FileHdr.Flags, ELFYAML::ELF_EF(0));
  IO.mapOptional("Entry", FileHdr.Entry, Hex64(0));

  IO.mapOptional("SHEntSize", FileHdr.SHEntSize);
  IO.mapOptional("SHOff", FileHdr.SHOff);
  IO.mapOptional("SHNum", FileHdr.SHNum);
  IO.mapOptional("SHStrNdx", FileHdr.SHStrNdx);
}

void MappingTraits<ELFYAML::ProgramHeader>::mapping(
    IO &IO, ELFYAML::ProgramHeader &Phdr) {
  IO.mapRequired("Type", Phdr.Type);
  IO.mapOptional("Flags", Phdr.Flags, ELFYAML::ELF_PF(0));
  IO.mapOptional("Sections", Phdr.Sections);
  IO.mapOptional("VAddr", Phdr.VAddr, Hex64(0));
  IO.mapOptional("PAddr", Phdr.PAddr, Hex64(0));
  IO.mapOptional("Align", Phdr.Align);
  IO.mapOptional("FileSize", Phdr.FileSize);
  IO.mapOptional("MemSize", Phdr.MemSize);
  IO.mapOptional("Offset", Phdr.Offset);
}

LLVM_YAML_STRONG_TYPEDEF(StringRef, StOtherPiece)

template <> struct ScalarTraits<StOtherPiece> {
  static void output(const StOtherPiece &Val, void *, raw_ostream &Out) {
    Out << Val;
  }
  static StringRef input(StringRef Scalar, void *, StOtherPiece &Val) {
    Val = Scalar;
    return {};
  }
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};
template <> struct SequenceElementTraits<StOtherPiece> {
  static const bool flow = true;
};

template <> struct ScalarTraits<ELFYAML::YAMLFlowString> {
  static void output(const ELFYAML::YAMLFlowString &Val, void *,
                     raw_ostream &Out) {
    Out << Val;
  }
  static StringRef input(StringRef Scalar, void *,
                         ELFYAML::YAMLFlowString &Val) {
    Val = Scalar;
    return {};
  }
  static QuotingType mustQuote(StringRef S) {
    return ScalarTraits<StringRef>::mustQuote(S);
  }
};
template <> struct SequenceElementTraits<ELFYAML::YAMLFlowString> {
  static const bool flow = true;
};

namespace {

struct NormalizedOther {
  NormalizedOther(IO &IO) : YamlIO(IO) {}
  NormalizedOther(IO &IO, Optional<uint8_t> Original) : YamlIO(IO) {
    assert(Original && "This constructor is only used for outputting YAML and "
                       "assumes a non-empty Original");
    std::vector<StOtherPiece> Ret;
    const auto *Object = static_cast<ELFYAML::Object *>(YamlIO.getContext());
    for (std::pair<StringRef, uint8_t> &P :
         getFlags(Object->Header.Machine).takeVector()) {
      uint8_t FlagValue = P.second;
      if ((*Original & FlagValue) != FlagValue)
        continue;
      *Original &= ~FlagValue;
      Ret.push_back({P.first});
    }

    if (*Original != 0) {
      UnknownFlagsHolder = std::to_string(*Original);
      Ret.push_back({UnknownFlagsHolder});
    }

    if (!Ret.empty())
      Other = std::move(Ret);
  }

  uint8_t toValue(StringRef Name) {
    const auto *Object = static_cast<ELFYAML::Object *>(YamlIO.getContext());
    MapVector<StringRef, uint8_t> Flags = getFlags(Object->Header.Machine);

    auto It = Flags.find(Name);
    if (It != Flags.end())
      return It->second;

    uint8_t Val;
    if (to_integer(Name, Val))
      return Val;

    YamlIO.setError("an unknown value is used for symbol's 'Other' field: " +
                    Name);
    return 0;
  }

  Optional<uint8_t> denormalize(IO &) {
    if (!Other)
      return None;
    uint8_t Ret = 0;
    for (StOtherPiece &Val : *Other)
      Ret |= toValue(Val);
    return Ret;
  }

  // st_other field is used to encode symbol visibility and platform-dependent
  // flags and values. This method returns a name to value map that is used for
  // parsing and encoding this field.
  MapVector<StringRef, uint8_t> getFlags(unsigned EMachine) {
    MapVector<StringRef, uint8_t> Map;
    // STV_* values are just enumeration values. We add them in a reversed order
    // because when we convert the st_other to named constants when printing
    // YAML we want to use a maximum number of bits on each step:
    // when we have st_other == 3, we want to print it as STV_PROTECTED (3), but
    // not as STV_HIDDEN (2) + STV_INTERNAL (1).
    Map["STV_PROTECTED"] = ELF::STV_PROTECTED;
    Map["STV_HIDDEN"] = ELF::STV_HIDDEN;
    Map["STV_INTERNAL"] = ELF::STV_INTERNAL;
    // STV_DEFAULT is used to represent the default visibility and has a value
    // 0. We want to be able to read it from YAML documents, but there is no
    // reason to print it.
    if (!YamlIO.outputting())
      Map["STV_DEFAULT"] = ELF::STV_DEFAULT;

    // MIPS is not consistent. All of the STO_MIPS_* values are bit flags,
    // except STO_MIPS_MIPS16 which overlaps them. It should be checked and
    // consumed first when we print the output, because we do not want to print
    // any other flags that have the same bits instead.
    if (EMachine == ELF::EM_MIPS) {
      Map["STO_MIPS_MIPS16"] = ELF::STO_MIPS_MIPS16;
      Map["STO_MIPS_MICROMIPS"] = ELF::STO_MIPS_MICROMIPS;
      Map["STO_MIPS_PIC"] = ELF::STO_MIPS_PIC;
      Map["STO_MIPS_PLT"] = ELF::STO_MIPS_PLT;
      Map["STO_MIPS_OPTIONAL"] = ELF::STO_MIPS_OPTIONAL;
    }
    return Map;
  }

  IO &YamlIO;
  Optional<std::vector<StOtherPiece>> Other;
  std::string UnknownFlagsHolder;
};

} // end anonymous namespace

void MappingTraits<ELFYAML::Symbol>::mapping(IO &IO, ELFYAML::Symbol &Symbol) {
  IO.mapOptional("Name", Symbol.Name, StringRef());
  IO.mapOptional("NameIndex", Symbol.NameIndex);
  IO.mapOptional("Type", Symbol.Type, ELFYAML::ELF_STT(0));
  IO.mapOptional("Section", Symbol.Section, StringRef());
  IO.mapOptional("Index", Symbol.Index);
  IO.mapOptional("Binding", Symbol.Binding, ELFYAML::ELF_STB(0));
  IO.mapOptional("Value", Symbol.Value, Hex64(0));
  IO.mapOptional("Size", Symbol.Size, Hex64(0));

  // Symbol's Other field is a bit special. It is usually a field that
  // represents st_other and holds the symbol visibility. However, on some
  // platforms, it can contain bit fields and regular values, or even sometimes a
  // crazy mix of them (see comments for NormalizedOther). Because of this, we
  // need special handling.
  MappingNormalization<NormalizedOther, Optional<uint8_t>> Keys(IO,
                                                                Symbol.Other);
  IO.mapOptional("Other", Keys->Other);
}

StringRef MappingTraits<ELFYAML::Symbol>::validate(IO &IO,
                                                   ELFYAML::Symbol &Symbol) {
  if (Symbol.Index && Symbol.Section.data())
    return "Index and Section cannot both be specified for Symbol";
  if (Symbol.NameIndex && !Symbol.Name.empty())
    return "Name and NameIndex cannot both be specified for Symbol";
  return StringRef();
}

static void commonSectionMapping(IO &IO, ELFYAML::Section &Section) {
  IO.mapOptional("Name", Section.Name, StringRef());
  IO.mapRequired("Type", Section.Type);
  IO.mapOptional("Flags", Section.Flags);
  IO.mapOptional("Address", Section.Address, Hex64(0));
  IO.mapOptional("Link", Section.Link, StringRef());
  IO.mapOptional("AddressAlign", Section.AddressAlign, Hex64(0));
  IO.mapOptional("EntSize", Section.EntSize);

  // obj2yaml does not dump these fields. They are expected to be empty when we
  // are producing YAML, because yaml2obj sets appropriate values for them
  // automatically when they are not explicitly defined.
  assert(!IO.outputting() ||
         (!Section.ShOffset.hasValue() && !Section.ShSize.hasValue() &&
          !Section.ShName.hasValue() && !Section.ShFlags.hasValue()));
  IO.mapOptional("ShName", Section.ShName);
  IO.mapOptional("ShOffset", Section.ShOffset);
  IO.mapOptional("ShSize", Section.ShSize);
  IO.mapOptional("ShFlags", Section.ShFlags);
}

static void sectionMapping(IO &IO, ELFYAML::DynamicSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Entries", Section.Entries);
  IO.mapOptional("Content", Section.Content);
}

static void sectionMapping(IO &IO, ELFYAML::RawContentSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Size", Section.Size);
  IO.mapOptional("Info", Section.Info);
}

static void sectionMapping(IO &IO, ELFYAML::StackSizesSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Size", Section.Size);
  IO.mapOptional("Entries", Section.Entries);
}

static void sectionMapping(IO &IO, ELFYAML::HashSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Bucket", Section.Bucket);
  IO.mapOptional("Chain", Section.Chain);
  IO.mapOptional("Size", Section.Size);
}

static void sectionMapping(IO &IO, ELFYAML::NoteSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Size", Section.Size);
  IO.mapOptional("Notes", Section.Notes);
}


static void sectionMapping(IO &IO, ELFYAML::GnuHashSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Header", Section.Header);
  IO.mapOptional("BloomFilter", Section.BloomFilter);
  IO.mapOptional("HashBuckets", Section.HashBuckets);
  IO.mapOptional("HashValues", Section.HashValues);
}
static void sectionMapping(IO &IO, ELFYAML::NoBitsSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Size", Section.Size, Hex64(0));
}

static void sectionMapping(IO &IO, ELFYAML::VerdefSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapRequired("Info", Section.Info);
  IO.mapOptional("Entries", Section.Entries);
  IO.mapOptional("Content", Section.Content);
}

static void sectionMapping(IO &IO, ELFYAML::SymverSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapRequired("Entries", Section.Entries);
}

static void sectionMapping(IO &IO, ELFYAML::VerneedSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapRequired("Info", Section.Info);
  IO.mapOptional("Dependencies", Section.VerneedV);
  IO.mapOptional("Content", Section.Content);
}

static void sectionMapping(IO &IO, ELFYAML::RelocationSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Info", Section.RelocatableSec, StringRef());
  IO.mapOptional("Relocations", Section.Relocations);
}

static void groupSectionMapping(IO &IO, ELFYAML::Group &Group) {
  commonSectionMapping(IO, Group);
  IO.mapOptional("Info", Group.Signature);
  IO.mapRequired("Members", Group.Members);
}

static void sectionMapping(IO &IO, ELFYAML::SymtabShndxSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapRequired("Entries", Section.Entries);
}

static void sectionMapping(IO &IO, ELFYAML::AddrsigSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Content", Section.Content);
  IO.mapOptional("Size", Section.Size);
  IO.mapOptional("Symbols", Section.Symbols);
}

static void fillMapping(IO &IO, ELFYAML::Fill &Fill) {
  IO.mapOptional("Name", Fill.Name, StringRef());
  IO.mapOptional("Pattern", Fill.Pattern);
  IO.mapRequired("Size", Fill.Size);
}

static void sectionMapping(IO &IO, ELFYAML::LinkerOptionsSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Options", Section.Options);
  IO.mapOptional("Content", Section.Content);
}

static void sectionMapping(IO &IO,
                           ELFYAML::DependentLibrariesSection &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Libraries", Section.Libs);
  IO.mapOptional("Content", Section.Content);
}

void MappingTraits<ELFYAML::SectionOrType>::mapping(
    IO &IO, ELFYAML::SectionOrType &sectionOrType) {
  IO.mapRequired("SectionOrType", sectionOrType.sectionNameOrType);
}

void MappingTraits<ELFYAML::SectionName>::mapping(
    IO &IO, ELFYAML::SectionName &sectionName) {
  IO.mapRequired("Section", sectionName.Section);
}

static void sectionMapping(IO &IO, ELFYAML::MipsABIFlags &Section) {
  commonSectionMapping(IO, Section);
  IO.mapOptional("Version", Section.Version, Hex16(0));
  IO.mapRequired("ISA", Section.ISALevel);
  IO.mapOptional("ISARevision", Section.ISARevision, Hex8(0));
  IO.mapOptional("ISAExtension", Section.ISAExtension,
                 ELFYAML::MIPS_AFL_EXT(Mips::AFL_EXT_NONE));
  IO.mapOptional("ASEs", Section.ASEs, ELFYAML::MIPS_AFL_ASE(0));
  IO.mapOptional("FpABI", Section.FpABI,
                 ELFYAML::MIPS_ABI_FP(Mips::Val_GNU_MIPS_ABI_FP_ANY));
  IO.mapOptional("GPRSize", Section.GPRSize,
                 ELFYAML::MIPS_AFL_REG(Mips::AFL_REG_NONE));
  IO.mapOptional("CPR1Size", Section.CPR1Size,
                 ELFYAML::MIPS_AFL_REG(Mips::AFL_REG_NONE));
  IO.mapOptional("CPR2Size", Section.CPR2Size,
                 ELFYAML::MIPS_AFL_REG(Mips::AFL_REG_NONE));
  IO.mapOptional("Flags1", Section.Flags1, ELFYAML::MIPS_AFL_FLAGS1(0));
  IO.mapOptional("Flags2", Section.Flags2, Hex32(0));
}

void MappingTraits<std::unique_ptr<ELFYAML::Chunk>>::mapping(
    IO &IO, std::unique_ptr<ELFYAML::Chunk> &Section) {
  ELFYAML::ELF_SHT Type;
  if (IO.outputting()) {
    Type = cast<ELFYAML::Section>(Section.get())->Type;
  } else {
    // When the Type string does not have a "SHT_" prefix, we know it is not a
    // description of a regular ELF output section. Currently, we have one
    // special type named "Fill". See comments for Fill.
    StringRef StrType;
    IO.mapRequired("Type", StrType);
    if (StrType == "Fill") {
      Section.reset(new ELFYAML::Fill());
      fillMapping(IO, *cast<ELFYAML::Fill>(Section.get()));
      return;
    }

    IO.mapRequired("Type", Type);
  }

  switch (Type) {
  case ELF::SHT_DYNAMIC:
    if (!IO.outputting())
      Section.reset(new ELFYAML::DynamicSection());
    sectionMapping(IO, *cast<ELFYAML::DynamicSection>(Section.get()));
    break;
  case ELF::SHT_REL:
  case ELF::SHT_RELA:
    if (!IO.outputting())
      Section.reset(new ELFYAML::RelocationSection());
    sectionMapping(IO, *cast<ELFYAML::RelocationSection>(Section.get()));
    break;
  case ELF::SHT_GROUP:
    if (!IO.outputting())
      Section.reset(new ELFYAML::Group());
    groupSectionMapping(IO, *cast<ELFYAML::Group>(Section.get()));
    break;
  case ELF::SHT_NOBITS:
    if (!IO.outputting())
      Section.reset(new ELFYAML::NoBitsSection());
    sectionMapping(IO, *cast<ELFYAML::NoBitsSection>(Section.get()));
    break;
  case ELF::SHT_HASH:
    if (!IO.outputting())
      Section.reset(new ELFYAML::HashSection());
    sectionMapping(IO, *cast<ELFYAML::HashSection>(Section.get()));
    break;
  case ELF::SHT_NOTE:
    if (!IO.outputting())
      Section.reset(new ELFYAML::NoteSection());
    sectionMapping(IO, *cast<ELFYAML::NoteSection>(Section.get()));
    break;
 case ELF::SHT_GNU_HASH:
    if (!IO.outputting())
      Section.reset(new ELFYAML::GnuHashSection());
    sectionMapping(IO, *cast<ELFYAML::GnuHashSection>(Section.get()));
    break;
  case ELF::SHT_MIPS_ABIFLAGS:
    if (!IO.outputting())
      Section.reset(new ELFYAML::MipsABIFlags());
    sectionMapping(IO, *cast<ELFYAML::MipsABIFlags>(Section.get()));
    break;
  case ELF::SHT_GNU_verdef:
    if (!IO.outputting())
      Section.reset(new ELFYAML::VerdefSection());
    sectionMapping(IO, *cast<ELFYAML::VerdefSection>(Section.get()));
    break;
  case ELF::SHT_GNU_versym:
    if (!IO.outputting())
      Section.reset(new ELFYAML::SymverSection());
    sectionMapping(IO, *cast<ELFYAML::SymverSection>(Section.get()));
    break;
  case ELF::SHT_GNU_verneed:
    if (!IO.outputting())
      Section.reset(new ELFYAML::VerneedSection());
    sectionMapping(IO, *cast<ELFYAML::VerneedSection>(Section.get()));
    break;
  case ELF::SHT_SYMTAB_SHNDX:
    if (!IO.outputting())
      Section.reset(new ELFYAML::SymtabShndxSection());
    sectionMapping(IO, *cast<ELFYAML::SymtabShndxSection>(Section.get()));
    break;
  case ELF::SHT_LLVM_ADDRSIG:
    if (!IO.outputting())
      Section.reset(new ELFYAML::AddrsigSection());
    sectionMapping(IO, *cast<ELFYAML::AddrsigSection>(Section.get()));
    break;
  case ELF::SHT_LLVM_LINKER_OPTIONS:
    if (!IO.outputting())
      Section.reset(new ELFYAML::LinkerOptionsSection());
    sectionMapping(IO, *cast<ELFYAML::LinkerOptionsSection>(Section.get()));
    break;
  case ELF::SHT_LLVM_DEPENDENT_LIBRARIES:
    if (!IO.outputting())
      Section.reset(new ELFYAML::DependentLibrariesSection());
    sectionMapping(IO,
                   *cast<ELFYAML::DependentLibrariesSection>(Section.get()));
    break;
  default:
    if (!IO.outputting()) {
      StringRef Name;
      IO.mapOptional("Name", Name, StringRef());
      Name = ELFYAML::dropUniqueSuffix(Name);

      if (ELFYAML::StackSizesSection::nameMatches(Name))
        Section = std::make_unique<ELFYAML::StackSizesSection>();
      else
        Section = std::make_unique<ELFYAML::RawContentSection>();
    }

    if (auto S = dyn_cast<ELFYAML::RawContentSection>(Section.get()))
      sectionMapping(IO, *S);
    else
      sectionMapping(IO, *cast<ELFYAML::StackSizesSection>(Section.get()));
  }
}

StringRef MappingTraits<std::unique_ptr<ELFYAML::Chunk>>::validate(
    IO &io, std::unique_ptr<ELFYAML::Chunk> &C) {
  if (const auto *RawSection = dyn_cast<ELFYAML::RawContentSection>(C.get())) {
    if (RawSection->Size && RawSection->Content &&
        (uint64_t)(*RawSection->Size) < RawSection->Content->binary_size())
      return "Section size must be greater than or equal to the content size";
    if (RawSection->Flags && RawSection->ShFlags)
      return "ShFlags and Flags cannot be used together";
    return {};
  }

  if (const auto *SS = dyn_cast<ELFYAML::StackSizesSection>(C.get())) {
    if (!SS->Entries && !SS->Content && !SS->Size)
      return ".stack_sizes: one of Content, Entries and Size must be specified";

    if (SS->Size && SS->Content &&
        (uint64_t)(*SS->Size) < SS->Content->binary_size())
      return ".stack_sizes: Size must be greater than or equal to the content "
             "size";

    // We accept Content, Size or both together when there are no Entries.
    if (!SS->Entries)
      return {};

    if (SS->Size)
      return ".stack_sizes: Size and Entries cannot be used together";
    if (SS->Content)
      return ".stack_sizes: Content and Entries cannot be used together";
    return {};
  }

  if (const auto *HS = dyn_cast<ELFYAML::HashSection>(C.get())) {
    if (!HS->Content && !HS->Bucket && !HS->Chain && !HS->Size)
      return "one of \"Content\", \"Size\", \"Bucket\" or \"Chain\" must be "
             "specified";

    if (HS->Content || HS->Size) {
      if (HS->Size && HS->Content &&
          (uint64_t)*HS->Size < HS->Content->binary_size())
        return "\"Size\" must be greater than or equal to the content "
               "size";

      if (HS->Bucket)
        return "\"Bucket\" cannot be used with \"Content\" or \"Size\"";
      if (HS->Chain)
        return "\"Chain\" cannot be used with \"Content\" or \"Size\"";
      return {};
    }

    if ((HS->Bucket && !HS->Chain) || (!HS->Bucket && HS->Chain))
      return "\"Bucket\" and \"Chain\" must be used together";
    return {};
  }

  if (const auto *Sec = dyn_cast<ELFYAML::AddrsigSection>(C.get())) {
    if (!Sec->Symbols && !Sec->Content && !Sec->Size)
      return "one of \"Content\", \"Size\" or \"Symbols\" must be specified";

    if (Sec->Content || Sec->Size) {
      if (Sec->Size && Sec->Content &&
          (uint64_t)*Sec->Size < Sec->Content->binary_size())
        return "\"Size\" must be greater than or equal to the content "
               "size";

      if (Sec->Symbols)
        return "\"Symbols\" cannot be used with \"Content\" or \"Size\"";
      return {};
    }

    if (!Sec->Symbols)
      return {};

    for (const ELFYAML::AddrsigSymbol &AS : *Sec->Symbols)
      if (AS.Index && AS.Name)
        return "\"Index\" and \"Name\" cannot be used together when defining a "
               "symbol";
    return {};
  }

  if (const auto *NS = dyn_cast<ELFYAML::NoteSection>(C.get())) {
    if (!NS->Content && !NS->Size && !NS->Notes)
      return "one of \"Content\", \"Size\" or \"Notes\" must be "
             "specified";

    if (!NS->Content && !NS->Size)
      return {};

    if (NS->Size && NS->Content &&
        (uint64_t)*NS->Size < NS->Content->binary_size())
      return "\"Size\" must be greater than or equal to the content "
             "size";

    if (NS->Notes)
      return "\"Notes\" cannot be used with \"Content\" or \"Size\"";
    return {};
  }

  if (const auto *Sec = dyn_cast<ELFYAML::GnuHashSection>(C.get())) {
    if (!Sec->Content && !Sec->Header && !Sec->BloomFilter &&
        !Sec->HashBuckets && !Sec->HashValues)
      return "either \"Content\" or \"Header\", \"BloomFilter\", "
             "\"HashBuckets\" and \"HashBuckets\" must be specified";

    if (Sec->Header || Sec->BloomFilter || Sec->HashBuckets ||
        Sec->HashValues) {
      if (!Sec->Header || !Sec->BloomFilter || !Sec->HashBuckets ||
          !Sec->HashValues)
        return "\"Header\", \"BloomFilter\", "
               "\"HashBuckets\" and \"HashValues\" must be used together";
      if (Sec->Content)
        return "\"Header\", \"BloomFilter\", "
               "\"HashBuckets\" and \"HashValues\" can't be used together with "
               "\"Content\"";
      return {};
    }

    // Only Content is specified.
    return {};
  }

  if (const auto *Sec = dyn_cast<ELFYAML::LinkerOptionsSection>(C.get())) {
    if (Sec->Options && Sec->Content)
      return "\"Options\" and \"Content\" can't be used together";
    return {};
  }

  if (const auto *Sec = dyn_cast<ELFYAML::DependentLibrariesSection>(C.get())) {
    if (Sec->Libs && Sec->Content)
      return "SHT_LLVM_DEPENDENT_LIBRARIES: \"Libraries\" and \"Content\" "
             "can't "
             "be used together";
    return {};
  }

  if (const auto *F = dyn_cast<ELFYAML::Fill>(C.get())) {
    if (!F->Pattern)
      return {};
    if (F->Pattern->binary_size() != 0 && !F->Size)
      return "\"Size\" can't be 0 when \"Pattern\" is not empty";
    return {};
  }

  if (const auto *VD = dyn_cast<ELFYAML::VerdefSection>(C.get())) {
    if (VD->Entries && VD->Content)
      return "SHT_GNU_verdef: \"Entries\" and \"Content\" can't be used "
             "together";
    return {};
  }

  if (const auto *VD = dyn_cast<ELFYAML::VerneedSection>(C.get())) {
    if (VD->VerneedV && VD->Content)
      return "SHT_GNU_verneed: \"Dependencies\" and \"Content\" can't be used "
             "together";
    return {};
  }

  return {};
}

namespace {

struct NormalizedMips64RelType {
  NormalizedMips64RelType(IO &)
      : Type(ELFYAML::ELF_REL(ELF::R_MIPS_NONE)),
        Type2(ELFYAML::ELF_REL(ELF::R_MIPS_NONE)),
        Type3(ELFYAML::ELF_REL(ELF::R_MIPS_NONE)),
        SpecSym(ELFYAML::ELF_REL(ELF::RSS_UNDEF)) {}
  NormalizedMips64RelType(IO &, ELFYAML::ELF_REL Original)
      : Type(Original & 0xFF), Type2(Original >> 8 & 0xFF),
        Type3(Original >> 16 & 0xFF), SpecSym(Original >> 24 & 0xFF) {}

  ELFYAML::ELF_REL denormalize(IO &) {
    ELFYAML::ELF_REL Res = Type | Type2 << 8 | Type3 << 16 | SpecSym << 24;
    return Res;
  }

  ELFYAML::ELF_REL Type;
  ELFYAML::ELF_REL Type2;
  ELFYAML::ELF_REL Type3;
  ELFYAML::ELF_RSS SpecSym;
};

} // end anonymous namespace

void MappingTraits<ELFYAML::StackSizeEntry>::mapping(
    IO &IO, ELFYAML::StackSizeEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapOptional("Address", E.Address, Hex64(0));
  IO.mapRequired("Size", E.Size);
}

void MappingTraits<ELFYAML::GnuHashHeader>::mapping(IO &IO,
                                                    ELFYAML::GnuHashHeader &E) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapOptional("NBuckets", E.NBuckets);
  IO.mapRequired("SymNdx", E.SymNdx);
  IO.mapOptional("MaskWords", E.MaskWords);
  IO.mapRequired("Shift2", E.Shift2);
}

void MappingTraits<ELFYAML::DynamicEntry>::mapping(IO &IO,
                                                   ELFYAML::DynamicEntry &Rel) {
  assert(IO.getContext() && "The IO context is not initialized");

  IO.mapRequired("Tag", Rel.Tag);
  IO.mapRequired("Value", Rel.Val);
}

void MappingTraits<ELFYAML::NoteEntry>::mapping(IO &IO, ELFYAML::NoteEntry &N) {
  assert(IO.getContext() && "The IO context is not initialized");

  IO.mapOptional("Name", N.Name);
  IO.mapOptional("Desc", N.Desc);
  IO.mapRequired("Type", N.Type);
}

void MappingTraits<ELFYAML::VerdefEntry>::mapping(IO &IO,
                                                  ELFYAML::VerdefEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");

  IO.mapRequired("Version", E.Version);
  IO.mapRequired("Flags", E.Flags);
  IO.mapRequired("VersionNdx", E.VersionNdx);
  IO.mapRequired("Hash", E.Hash);
  IO.mapRequired("Names", E.VerNames);
}

void MappingTraits<ELFYAML::VerneedEntry>::mapping(IO &IO,
                                                   ELFYAML::VerneedEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");

  IO.mapRequired("Version", E.Version);
  IO.mapRequired("File", E.File);
  IO.mapRequired("Entries", E.AuxV);
}

void MappingTraits<ELFYAML::VernauxEntry>::mapping(IO &IO,
                                                   ELFYAML::VernauxEntry &E) {
  assert(IO.getContext() && "The IO context is not initialized");

  IO.mapRequired("Name", E.Name);
  IO.mapRequired("Hash", E.Hash);
  IO.mapRequired("Flags", E.Flags);
  IO.mapRequired("Other", E.Other);
}

void MappingTraits<ELFYAML::Relocation>::mapping(IO &IO,
                                                 ELFYAML::Relocation &Rel) {
  const auto *Object = static_cast<ELFYAML::Object *>(IO.getContext());
  assert(Object && "The IO context is not initialized");

  IO.mapRequired("Offset", Rel.Offset);
  IO.mapOptional("Symbol", Rel.Symbol);

  if (Object->Header.Machine == ELFYAML::ELF_EM(ELF::EM_MIPS) &&
      Object->Header.Class == ELFYAML::ELF_ELFCLASS(ELF::ELFCLASS64)) {
    MappingNormalization<NormalizedMips64RelType, ELFYAML::ELF_REL> Key(
        IO, Rel.Type);
    IO.mapRequired("Type", Key->Type);
    IO.mapOptional("Type2", Key->Type2, ELFYAML::ELF_REL(ELF::R_MIPS_NONE));
    IO.mapOptional("Type3", Key->Type3, ELFYAML::ELF_REL(ELF::R_MIPS_NONE));
    IO.mapOptional("SpecSym", Key->SpecSym, ELFYAML::ELF_RSS(ELF::RSS_UNDEF));
  } else
    IO.mapRequired("Type", Rel.Type);

  IO.mapOptional("Addend", Rel.Addend, (int64_t)0);
}

void MappingTraits<ELFYAML::Object>::mapping(IO &IO, ELFYAML::Object &Object) {
  assert(!IO.getContext() && "The IO context is initialized already");
  IO.setContext(&Object);
  IO.mapTag("!ELF", true);
  IO.mapRequired("FileHeader", Object.Header);
  IO.mapOptional("ProgramHeaders", Object.ProgramHeaders);
  IO.mapOptional("Sections", Object.Chunks);
  IO.mapOptional("Symbols", Object.Symbols);
  IO.mapOptional("DynamicSymbols", Object.DynamicSymbols);
  IO.setContext(nullptr);
}

void MappingTraits<ELFYAML::AddrsigSymbol>::mapping(IO &IO, ELFYAML::AddrsigSymbol &Sym) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapOptional("Name", Sym.Name);
  IO.mapOptional("Index", Sym.Index);
}

void MappingTraits<ELFYAML::LinkerOption>::mapping(IO &IO,
                                                   ELFYAML::LinkerOption &Opt) {
  assert(IO.getContext() && "The IO context is not initialized");
  IO.mapRequired("Name", Opt.Key);
  IO.mapRequired("Value", Opt.Value);
}

LLVM_YAML_STRONG_TYPEDEF(uint8_t, MIPS_AFL_REG)
LLVM_YAML_STRONG_TYPEDEF(uint8_t, MIPS_ABI_FP)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_AFL_EXT)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_AFL_ASE)
LLVM_YAML_STRONG_TYPEDEF(uint32_t, MIPS_AFL_FLAGS1)

} // end namespace yaml

} // end namespace llvm
