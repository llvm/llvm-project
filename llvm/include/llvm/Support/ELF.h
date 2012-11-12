//===-- llvm/Support/ELF.h - ELF constants and data structures --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header contains common, non-processor-specific data structures and
// constants for the ELF file format.
//
// The details of the ELF32 bits in this file are largely based on the Tool
// Interface Standard (TIS) Executable and Linking Format (ELF) Specification
// Version 1.2, May 1995. The ELF64 stuff is based on ELF-64 Object File Format
// Version 1.5, Draft 2, May 1998 as well as OpenBSD header files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ELF_H
#define LLVM_SUPPORT_ELF_H

#include "llvm/Support/DataTypes.h"
#include <cstring>

namespace llvm {

namespace ELF {

typedef uint32_t Elf32_Addr; // Program address
typedef uint32_t Elf32_Off;  // File offset
typedef uint16_t Elf32_Half;
typedef uint32_t Elf32_Word;
typedef int32_t  Elf32_Sword;

typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;
typedef uint16_t Elf64_Half;
typedef uint32_t Elf64_Word;
typedef int32_t  Elf64_Sword;
typedef uint64_t Elf64_Xword;
typedef int64_t  Elf64_Sxword;

// Object file magic string.
static const char ElfMagic[] = { 0x7f, 'E', 'L', 'F', '\0' };

// e_ident size and indices.
enum {
  EI_MAG0       = 0,          // File identification index.
  EI_MAG1       = 1,          // File identification index.
  EI_MAG2       = 2,          // File identification index.
  EI_MAG3       = 3,          // File identification index.
  EI_CLASS      = 4,          // File class.
  EI_DATA       = 5,          // Data encoding.
  EI_VERSION    = 6,          // File version.
  EI_OSABI      = 7,          // OS/ABI identification.
  EI_ABIVERSION = 8,          // ABI version.
  EI_PAD        = 9,          // Start of padding bytes.
  EI_NIDENT     = 16          // Number of bytes in e_ident.
};

struct Elf32_Ehdr {
  unsigned char e_ident[EI_NIDENT]; // ELF Identification bytes
  Elf32_Half    e_type;      // Type of file (see ET_* below)
  Elf32_Half    e_machine;   // Required architecture for this file (see EM_*)
  Elf32_Word    e_version;   // Must be equal to 1
  Elf32_Addr    e_entry;     // Address to jump to in order to start program
  Elf32_Off     e_phoff;     // Program header table's file offset, in bytes
  Elf32_Off     e_shoff;     // Section header table's file offset, in bytes
  Elf32_Word    e_flags;     // Processor-specific flags
  Elf32_Half    e_ehsize;    // Size of ELF header, in bytes
  Elf32_Half    e_phentsize; // Size of an entry in the program header table
  Elf32_Half    e_phnum;     // Number of entries in the program header table
  Elf32_Half    e_shentsize; // Size of an entry in the section header table
  Elf32_Half    e_shnum;     // Number of entries in the section header table
  Elf32_Half    e_shstrndx;  // Sect hdr table index of sect name string table
  bool checkMagic() const {
    return (memcmp(e_ident, ElfMagic, strlen(ElfMagic))) == 0;
  }
  unsigned char getFileClass() const { return e_ident[EI_CLASS]; }
  unsigned char getDataEncoding() const { return e_ident[EI_DATA]; }
};

// 64-bit ELF header. Fields are the same as for ELF32, but with different
// types (see above).
struct Elf64_Ehdr {
  unsigned char e_ident[EI_NIDENT];
  Elf64_Half    e_type;
  Elf64_Half    e_machine;
  Elf64_Word    e_version;
  Elf64_Addr    e_entry;
  Elf64_Off     e_phoff;
  Elf64_Off     e_shoff;
  Elf64_Word    e_flags;
  Elf64_Half    e_ehsize;
  Elf64_Half    e_phentsize;
  Elf64_Half    e_phnum;
  Elf64_Half    e_shentsize;
  Elf64_Half    e_shnum;
  Elf64_Half    e_shstrndx;
  bool checkMagic() const {
    return (memcmp(e_ident, ElfMagic, strlen(ElfMagic))) == 0;
  }
  unsigned char getFileClass() const { return e_ident[EI_CLASS]; }
  unsigned char getDataEncoding() const { return e_ident[EI_DATA]; }
};

// File types
enum {
  ET_NONE   = 0,      // No file type
  ET_REL    = 1,      // Relocatable file
  ET_EXEC   = 2,      // Executable file
  ET_DYN    = 3,      // Shared object file
  ET_CORE   = 4,      // Core file
  ET_LOPROC = 0xff00, // Beginning of processor-specific codes
  ET_HIPROC = 0xffff  // Processor-specific
};

// Versioning
enum {
  EV_NONE = 0,
  EV_CURRENT = 1
};

// Machine architectures
enum {
  EM_NONE          = 0, // No machine
  EM_M32           = 1, // AT&T WE 32100
  EM_SPARC         = 2, // SPARC
  EM_386           = 3, // Intel 386
  EM_68K           = 4, // Motorola 68000
  EM_88K           = 5, // Motorola 88000
  EM_486           = 6, // Intel 486 (deprecated)
  EM_860           = 7, // Intel 80860
  EM_MIPS          = 8, // MIPS R3000
  EM_S370          = 9, // IBM System/370
  EM_MIPS_RS3_LE   = 10, // MIPS RS3000 Little-endian
  EM_PARISC        = 15, // Hewlett-Packard PA-RISC
  EM_VPP500        = 17, // Fujitsu VPP500
  EM_SPARC32PLUS   = 18, // Enhanced instruction set SPARC
  EM_960           = 19, // Intel 80960
  EM_PPC           = 20, // PowerPC
  EM_PPC64         = 21, // PowerPC64
  EM_S390          = 22, // IBM System/390
  EM_SPU           = 23, // IBM SPU/SPC
  EM_V800          = 36, // NEC V800
  EM_FR20          = 37, // Fujitsu FR20
  EM_RH32          = 38, // TRW RH-32
  EM_RCE           = 39, // Motorola RCE
  EM_ARM           = 40, // ARM
  EM_ALPHA         = 41, // DEC Alpha
  EM_SH            = 42, // Hitachi SH
  EM_SPARCV9       = 43, // SPARC V9
  EM_TRICORE       = 44, // Siemens TriCore
  EM_ARC           = 45, // Argonaut RISC Core
  EM_H8_300        = 46, // Hitachi H8/300
  EM_H8_300H       = 47, // Hitachi H8/300H
  EM_H8S           = 48, // Hitachi H8S
  EM_H8_500        = 49, // Hitachi H8/500
  EM_IA_64         = 50, // Intel IA-64 processor architecture
  EM_MIPS_X        = 51, // Stanford MIPS-X
  EM_COLDFIRE      = 52, // Motorola ColdFire
  EM_68HC12        = 53, // Motorola M68HC12
  EM_MMA           = 54, // Fujitsu MMA Multimedia Accelerator
  EM_PCP           = 55, // Siemens PCP
  EM_NCPU          = 56, // Sony nCPU embedded RISC processor
  EM_NDR1          = 57, // Denso NDR1 microprocessor
  EM_STARCORE      = 58, // Motorola Star*Core processor
  EM_ME16          = 59, // Toyota ME16 processor
  EM_ST100         = 60, // STMicroelectronics ST100 processor
  EM_TINYJ         = 61, // Advanced Logic Corp. TinyJ embedded processor family
  EM_X86_64        = 62, // AMD x86-64 architecture
  EM_PDSP          = 63, // Sony DSP Processor
  EM_PDP10         = 64, // Digital Equipment Corp. PDP-10
  EM_PDP11         = 65, // Digital Equipment Corp. PDP-11
  EM_FX66          = 66, // Siemens FX66 microcontroller
  EM_ST9PLUS       = 67, // STMicroelectronics ST9+ 8/16 bit microcontroller
  EM_ST7           = 68, // STMicroelectronics ST7 8-bit microcontroller
  EM_68HC16        = 69, // Motorola MC68HC16 Microcontroller
  EM_68HC11        = 70, // Motorola MC68HC11 Microcontroller
  EM_68HC08        = 71, // Motorola MC68HC08 Microcontroller
  EM_68HC05        = 72, // Motorola MC68HC05 Microcontroller
  EM_SVX           = 73, // Silicon Graphics SVx
  EM_ST19          = 74, // STMicroelectronics ST19 8-bit microcontroller
  EM_VAX           = 75, // Digital VAX
  EM_CRIS          = 76, // Axis Communications 32-bit embedded processor
  EM_JAVELIN       = 77, // Infineon Technologies 32-bit embedded processor
  EM_FIREPATH      = 78, // Element 14 64-bit DSP Processor
  EM_ZSP           = 79, // LSI Logic 16-bit DSP Processor
  EM_MMIX          = 80, // Donald Knuth's educational 64-bit processor
  EM_HUANY         = 81, // Harvard University machine-independent object files
  EM_PRISM         = 82, // SiTera Prism
  EM_AVR           = 83, // Atmel AVR 8-bit microcontroller
  EM_FR30          = 84, // Fujitsu FR30
  EM_D10V          = 85, // Mitsubishi D10V
  EM_D30V          = 86, // Mitsubishi D30V
  EM_V850          = 87, // NEC v850
  EM_M32R          = 88, // Mitsubishi M32R
  EM_MN10300       = 89, // Matsushita MN10300
  EM_MN10200       = 90, // Matsushita MN10200
  EM_PJ            = 91, // picoJava
  EM_OPENRISC      = 92, // OpenRISC 32-bit embedded processor
  EM_ARC_COMPACT   = 93, // ARC International ARCompact processor (old
                         // spelling/synonym: EM_ARC_A5)
  EM_XTENSA        = 94, // Tensilica Xtensa Architecture
  EM_VIDEOCORE     = 95, // Alphamosaic VideoCore processor
  EM_TMM_GPP       = 96, // Thompson Multimedia General Purpose Processor
  EM_NS32K         = 97, // National Semiconductor 32000 series
  EM_TPC           = 98, // Tenor Network TPC processor
  EM_SNP1K         = 99, // Trebia SNP 1000 processor
  EM_ST200         = 100, // STMicroelectronics (www.st.com) ST200
  EM_IP2K          = 101, // Ubicom IP2xxx microcontroller family
  EM_MAX           = 102, // MAX Processor
  EM_CR            = 103, // National Semiconductor CompactRISC microprocessor
  EM_F2MC16        = 104, // Fujitsu F2MC16
  EM_MSP430        = 105, // Texas Instruments embedded microcontroller msp430
  EM_BLACKFIN      = 106, // Analog Devices Blackfin (DSP) processor
  EM_SE_C33        = 107, // S1C33 Family of Seiko Epson processors
  EM_SEP           = 108, // Sharp embedded microprocessor
  EM_ARCA          = 109, // Arca RISC Microprocessor
  EM_UNICORE       = 110, // Microprocessor series from PKU-Unity Ltd. and MPRC
                          // of Peking University
  EM_EXCESS        = 111, // eXcess: 16/32/64-bit configurable embedded CPU
  EM_DXP           = 112, // Icera Semiconductor Inc. Deep Execution Processor
  EM_ALTERA_NIOS2  = 113, // Altera Nios II soft-core processor
  EM_CRX           = 114, // National Semiconductor CompactRISC CRX
  EM_XGATE         = 115, // Motorola XGATE embedded processor
  EM_C166          = 116, // Infineon C16x/XC16x processor
  EM_M16C          = 117, // Renesas M16C series microprocessors
  EM_DSPIC30F      = 118, // Microchip Technology dsPIC30F Digital Signal
                          // Controller
  EM_CE            = 119, // Freescale Communication Engine RISC core
  EM_M32C          = 120, // Renesas M32C series microprocessors
  EM_TSK3000       = 131, // Altium TSK3000 core
  EM_RS08          = 132, // Freescale RS08 embedded processor
  EM_SHARC         = 133, // Analog Devices SHARC family of 32-bit DSP
                          // processors
  EM_ECOG2         = 134, // Cyan Technology eCOG2 microprocessor
  EM_SCORE7        = 135, // Sunplus S+core7 RISC processor
  EM_DSP24         = 136, // New Japan Radio (NJR) 24-bit DSP Processor
  EM_VIDEOCORE3    = 137, // Broadcom VideoCore III processor
  EM_LATTICEMICO32 = 138, // RISC processor for Lattice FPGA architecture
  EM_SE_C17        = 139, // Seiko Epson C17 family
  EM_TI_C6000      = 140, // The Texas Instruments TMS320C6000 DSP family
  EM_TI_C2000      = 141, // The Texas Instruments TMS320C2000 DSP family
  EM_TI_C5500      = 142, // The Texas Instruments TMS320C55x DSP family
  EM_MMDSP_PLUS    = 160, // STMicroelectronics 64bit VLIW Data Signal Processor
  EM_CYPRESS_M8C   = 161, // Cypress M8C microprocessor
  EM_R32C          = 162, // Renesas R32C series microprocessors
  EM_TRIMEDIA      = 163, // NXP Semiconductors TriMedia architecture family
  EM_HEXAGON       = 164, // Qualcomm Hexagon processor
  EM_8051          = 165, // Intel 8051 and variants
  EM_STXP7X        = 166, // STMicroelectronics STxP7x family of configurable
                          // and extensible RISC processors
  EM_NDS32         = 167, // Andes Technology compact code size embedded RISC
                          // processor family
  EM_ECOG1         = 168, // Cyan Technology eCOG1X family
  EM_ECOG1X        = 168, // Cyan Technology eCOG1X family
  EM_MAXQ30        = 169, // Dallas Semiconductor MAXQ30 Core Micro-controllers
  EM_XIMO16        = 170, // New Japan Radio (NJR) 16-bit DSP Processor
  EM_MANIK         = 171, // M2000 Reconfigurable RISC Microprocessor
  EM_CRAYNV2       = 172, // Cray Inc. NV2 vector architecture
  EM_RX            = 173, // Renesas RX family
  EM_METAG         = 174, // Imagination Technologies META processor
                          // architecture
  EM_MCST_ELBRUS   = 175, // MCST Elbrus general purpose hardware architecture
  EM_ECOG16        = 176, // Cyan Technology eCOG16 family
  EM_CR16          = 177, // National Semiconductor CompactRISC CR16 16-bit
                          // microprocessor
  EM_ETPU          = 178, // Freescale Extended Time Processing Unit
  EM_SLE9X         = 179, // Infineon Technologies SLE9X core
  EM_L10M          = 180, // Intel L10M
  EM_K10M          = 181, // Intel K10M
  EM_AVR32         = 185, // Atmel Corporation 32-bit microprocessor family
  EM_STM8          = 186, // STMicroeletronics STM8 8-bit microcontroller
  EM_TILE64        = 187, // Tilera TILE64 multicore architecture family
  EM_TILEPRO       = 188, // Tilera TILEPro multicore architecture family
  EM_MICROBLAZE    = 189, // Xilinx MicroBlaze 32-bit RISC soft processor core
  EM_CUDA          = 190, // NVIDIA CUDA architecture
  EM_TILEGX        = 191, // Tilera TILE-Gx multicore architecture family
  EM_CLOUDSHIELD   = 192, // CloudShield architecture family
  EM_COREA_1ST     = 193, // KIPO-KAIST Core-A 1st generation processor family
  EM_COREA_2ND     = 194, // KIPO-KAIST Core-A 2nd generation processor family
  EM_ARC_COMPACT2  = 195, // Synopsys ARCompact V2
  EM_OPEN8         = 196, // Open8 8-bit RISC soft processor core
  EM_RL78          = 197, // Renesas RL78 family
  EM_VIDEOCORE5    = 198, // Broadcom VideoCore V processor
  EM_78KOR         = 199, // Renesas 78KOR family
  EM_56800EX       = 200, // Freescale 56800EX Digital Signal Controller (DSC)
  EM_MBLAZE        = 47787 // Xilinx MicroBlaze
};

// Object file classes.
enum {
  ELFCLASSNONE = 0,
  ELFCLASS32 = 1, // 32-bit object file
  ELFCLASS64 = 2  // 64-bit object file
};

// Object file byte orderings.
enum {
  ELFDATANONE = 0, // Invalid data encoding.
  ELFDATA2LSB = 1, // Little-endian object file
  ELFDATA2MSB = 2  // Big-endian object file
};

// OS ABI identification.
enum {
  ELFOSABI_NONE = 0,          // UNIX System V ABI
  ELFOSABI_HPUX = 1,          // HP-UX operating system
  ELFOSABI_NETBSD = 2,        // NetBSD
  ELFOSABI_LINUX = 3,         // GNU/Linux
  ELFOSABI_HURD = 4,          // GNU/Hurd
  ELFOSABI_SOLARIS = 6,       // Solaris
  ELFOSABI_AIX = 7,           // AIX
  ELFOSABI_IRIX = 8,          // IRIX
  ELFOSABI_FREEBSD = 9,       // FreeBSD
  ELFOSABI_TRU64 = 10,        // TRU64 UNIX
  ELFOSABI_MODESTO = 11,      // Novell Modesto
  ELFOSABI_OPENBSD = 12,      // OpenBSD
  ELFOSABI_OPENVMS = 13,      // OpenVMS
  ELFOSABI_NSK = 14,          // Hewlett-Packard Non-Stop Kernel
  ELFOSABI_AROS = 15,         // AROS
  ELFOSABI_FENIXOS = 16,      // FenixOS
  ELFOSABI_C6000_ELFABI = 64, // Bare-metal TMS320C6000
  ELFOSABI_C6000_LINUX = 65,  // Linux TMS320C6000
  ELFOSABI_ARM = 97,          // ARM
  ELFOSABI_STANDALONE = 255   // Standalone (embedded) application
};

// X86_64 relocations.
enum {
  R_X86_64_NONE       = 0,
  R_X86_64_64         = 1,
  R_X86_64_PC32       = 2,
  R_X86_64_GOT32      = 3,
  R_X86_64_PLT32      = 4,
  R_X86_64_COPY       = 5,
  R_X86_64_GLOB_DAT   = 6,
  R_X86_64_JUMP_SLOT  = 7,
  R_X86_64_RELATIVE   = 8,
  R_X86_64_GOTPCREL   = 9,
  R_X86_64_32         = 10,
  R_X86_64_32S        = 11,
  R_X86_64_16         = 12,
  R_X86_64_PC16       = 13,
  R_X86_64_8          = 14,
  R_X86_64_PC8        = 15,
  R_X86_64_DTPMOD64   = 16,
  R_X86_64_DTPOFF64   = 17,
  R_X86_64_TPOFF64    = 18,
  R_X86_64_TLSGD      = 19,
  R_X86_64_TLSLD      = 20,
  R_X86_64_DTPOFF32   = 21,
  R_X86_64_GOTTPOFF   = 22,
  R_X86_64_TPOFF32    = 23,
  R_X86_64_PC64       = 24,
  R_X86_64_GOTOFF64   = 25,
  R_X86_64_GOTPC32    = 26,
  R_X86_64_GOT64      = 27,
  R_X86_64_GOTPCREL64 = 28,
  R_X86_64_GOTPC64    = 29,
  R_X86_64_GOTPLT64   = 30,
  R_X86_64_PLTOFF64   = 31,
  R_X86_64_SIZE32     = 32,
  R_X86_64_SIZE64     = 33,
  R_X86_64_GOTPC32_TLSDESC = 34,
  R_X86_64_TLSDESC_CALL    = 35,
  R_X86_64_TLSDESC    = 36
};

// i386 relocations.
// TODO: this is just a subset
enum {
  R_386_NONE          = 0,
  R_386_32            = 1,
  R_386_PC32          = 2,
  R_386_GOT32         = 3,
  R_386_PLT32         = 4,
  R_386_COPY          = 5,
  R_386_GLOB_DAT      = 6,
  R_386_JUMP_SLOT     = 7,
  R_386_RELATIVE      = 8,
  R_386_GOTOFF        = 9,
  R_386_GOTPC         = 10,
  R_386_32PLT         = 11,
  R_386_TLS_TPOFF     = 14,
  R_386_TLS_IE        = 15,
  R_386_TLS_GOTIE     = 16,
  R_386_TLS_LE        = 17,
  R_386_TLS_GD        = 18,
  R_386_TLS_LDM       = 19,
  R_386_16            = 20,
  R_386_PC16          = 21,
  R_386_8             = 22,
  R_386_PC8           = 23,
  R_386_TLS_GD_32     = 24,
  R_386_TLS_GD_PUSH   = 25,
  R_386_TLS_GD_CALL   = 26,
  R_386_TLS_GD_POP    = 27,
  R_386_TLS_LDM_32    = 28,
  R_386_TLS_LDM_PUSH  = 29,
  R_386_TLS_LDM_CALL  = 30,
  R_386_TLS_LDM_POP   = 31,
  R_386_TLS_LDO_32    = 32,
  R_386_TLS_IE_32     = 33,
  R_386_TLS_LE_32     = 34,
  R_386_TLS_DTPMOD32  = 35,
  R_386_TLS_DTPOFF32  = 36,
  R_386_TLS_TPOFF32   = 37,
  R_386_TLS_GOTDESC   = 39,
  R_386_TLS_DESC_CALL = 40,
  R_386_TLS_DESC      = 41,
  R_386_IRELATIVE     = 42,
  R_386_NUM           = 43
};

// MBlaze relocations.
enum {
  R_MICROBLAZE_NONE           = 0,
  R_MICROBLAZE_32             = 1,
  R_MICROBLAZE_32_PCREL       = 2,
  R_MICROBLAZE_64_PCREL       = 3,
  R_MICROBLAZE_32_PCREL_LO    = 4,
  R_MICROBLAZE_64             = 5,
  R_MICROBLAZE_32_LO          = 6,
  R_MICROBLAZE_SRO32          = 7,
  R_MICROBLAZE_SRW32          = 8,
  R_MICROBLAZE_64_NONE        = 9,
  R_MICROBLAZE_32_SYM_OP_SYM  = 10,
  R_MICROBLAZE_GNU_VTINHERIT  = 11,
  R_MICROBLAZE_GNU_VTENTRY    = 12,
  R_MICROBLAZE_GOTPC_64       = 13,
  R_MICROBLAZE_GOT_64         = 14,
  R_MICROBLAZE_PLT_64         = 15,
  R_MICROBLAZE_REL            = 16,
  R_MICROBLAZE_JUMP_SLOT      = 17,
  R_MICROBLAZE_GLOB_DAT       = 18,
  R_MICROBLAZE_GOTOFF_64      = 19,
  R_MICROBLAZE_GOTOFF_32      = 20,
  R_MICROBLAZE_COPY           = 21
};

// ELF Relocation types for PPC32
enum {
  R_PPC_NONE                  = 0,      /* No relocation. */
  R_PPC_ADDR32                = 1,
  R_PPC_ADDR24                = 2,
  R_PPC_ADDR16                = 3,
  R_PPC_ADDR16_LO             = 4,
  R_PPC_ADDR16_HI             = 5,
  R_PPC_ADDR16_HA             = 6,
  R_PPC_ADDR14                = 7,
  R_PPC_ADDR14_BRTAKEN        = 8,
  R_PPC_ADDR14_BRNTAKEN       = 9,
  R_PPC_REL24                 = 10,
  R_PPC_REL14                 = 11,
  R_PPC_REL14_BRTAKEN         = 12,
  R_PPC_REL14_BRNTAKEN        = 13,
  R_PPC_REL32                 = 26
};

// ELF Relocation types for PPC64
enum {
  R_PPC64_ADDR16_LO           = 4,
  R_PPC64_ADDR16_HI           = 5,
  R_PPC64_ADDR14              = 7,
  R_PPC64_REL24               = 10,
  R_PPC64_ADDR64              = 38,
  R_PPC64_ADDR16_HIGHER       = 39,
  R_PPC64_ADDR16_HIGHEST      = 41,
  R_PPC64_TOC16               = 47,
  R_PPC64_TOC                 = 51,
  R_PPC64_TOC16_DS            = 63
};

// ARM Specific e_flags
enum { EF_ARM_EABIMASK = 0xFF000000U };

// ELF Relocation types for ARM
// Meets 2.08 ABI Specs.

enum {
  R_ARM_NONE                  = 0x00,
  R_ARM_PC24                  = 0x01,
  R_ARM_ABS32                 = 0x02,
  R_ARM_REL32                 = 0x03,
  R_ARM_LDR_PC_G0             = 0x04,
  R_ARM_ABS16                 = 0x05,
  R_ARM_ABS12                 = 0x06,
  R_ARM_THM_ABS5              = 0x07,
  R_ARM_ABS8                  = 0x08,
  R_ARM_SBREL32               = 0x09,
  R_ARM_THM_CALL              = 0x0a,
  R_ARM_THM_PC8               = 0x0b,
  R_ARM_BREL_ADJ              = 0x0c,
  R_ARM_TLS_DESC              = 0x0d,
  R_ARM_THM_SWI8              = 0x0e,
  R_ARM_XPC25                 = 0x0f,
  R_ARM_THM_XPC22             = 0x10,
  R_ARM_TLS_DTPMOD32          = 0x11,
  R_ARM_TLS_DTPOFF32          = 0x12,
  R_ARM_TLS_TPOFF32           = 0x13,
  R_ARM_COPY                  = 0x14,
  R_ARM_GLOB_DAT              = 0x15,
  R_ARM_JUMP_SLOT             = 0x16,
  R_ARM_RELATIVE              = 0x17,
  R_ARM_GOTOFF32              = 0x18,
  R_ARM_BASE_PREL             = 0x19,
  R_ARM_GOT_BREL              = 0x1a,
  R_ARM_PLT32                 = 0x1b,
  R_ARM_CALL                  = 0x1c,
  R_ARM_JUMP24                = 0x1d,
  R_ARM_THM_JUMP24            = 0x1e,
  R_ARM_BASE_ABS              = 0x1f,
  R_ARM_ALU_PCREL_7_0         = 0x20,
  R_ARM_ALU_PCREL_15_8        = 0x21,
  R_ARM_ALU_PCREL_23_15       = 0x22,
  R_ARM_LDR_SBREL_11_0_NC     = 0x23,
  R_ARM_ALU_SBREL_19_12_NC    = 0x24,
  R_ARM_ALU_SBREL_27_20_CK    = 0x25,
  R_ARM_TARGET1               = 0x26,
  R_ARM_SBREL31               = 0x27,
  R_ARM_V4BX                  = 0x28,
  R_ARM_TARGET2               = 0x29,
  R_ARM_PREL31                = 0x2a,
  R_ARM_MOVW_ABS_NC           = 0x2b,
  R_ARM_MOVT_ABS              = 0x2c,
  R_ARM_MOVW_PREL_NC          = 0x2d,
  R_ARM_MOVT_PREL             = 0x2e,
  R_ARM_THM_MOVW_ABS_NC       = 0x2f,
  R_ARM_THM_MOVT_ABS          = 0x30,
  R_ARM_THM_MOVW_PREL_NC      = 0x31,
  R_ARM_THM_MOVT_PREL         = 0x32,
  R_ARM_THM_JUMP19            = 0x33,
  R_ARM_THM_JUMP6             = 0x34,
  R_ARM_THM_ALU_PREL_11_0     = 0x35,
  R_ARM_THM_PC12              = 0x36,
  R_ARM_ABS32_NOI             = 0x37,
  R_ARM_REL32_NOI             = 0x38,
  R_ARM_ALU_PC_G0_NC          = 0x39,
  R_ARM_ALU_PC_G0             = 0x3a,
  R_ARM_ALU_PC_G1_NC          = 0x3b,
  R_ARM_ALU_PC_G1             = 0x3c,
  R_ARM_ALU_PC_G2             = 0x3d,
  R_ARM_LDR_PC_G1             = 0x3e,
  R_ARM_LDR_PC_G2             = 0x3f,
  R_ARM_LDRS_PC_G0            = 0x40,
  R_ARM_LDRS_PC_G1            = 0x41,
  R_ARM_LDRS_PC_G2            = 0x42,
  R_ARM_LDC_PC_G0             = 0x43,
  R_ARM_LDC_PC_G1             = 0x44,
  R_ARM_LDC_PC_G2             = 0x45,
  R_ARM_ALU_SB_G0_NC          = 0x46,
  R_ARM_ALU_SB_G0             = 0x47,
  R_ARM_ALU_SB_G1_NC          = 0x48,
  R_ARM_ALU_SB_G1             = 0x49,
  R_ARM_ALU_SB_G2             = 0x4a,
  R_ARM_LDR_SB_G0             = 0x4b,
  R_ARM_LDR_SB_G1             = 0x4c,
  R_ARM_LDR_SB_G2             = 0x4d,
  R_ARM_LDRS_SB_G0            = 0x4e,
  R_ARM_LDRS_SB_G1            = 0x4f,
  R_ARM_LDRS_SB_G2            = 0x50,
  R_ARM_LDC_SB_G0             = 0x51,
  R_ARM_LDC_SB_G1             = 0x52,
  R_ARM_LDC_SB_G2             = 0x53,
  R_ARM_MOVW_BREL_NC          = 0x54,
  R_ARM_MOVT_BREL             = 0x55,
  R_ARM_MOVW_BREL             = 0x56,
  R_ARM_THM_MOVW_BREL_NC      = 0x57,
  R_ARM_THM_MOVT_BREL         = 0x58,
  R_ARM_THM_MOVW_BREL         = 0x59,
  R_ARM_TLS_GOTDESC           = 0x5a,
  R_ARM_TLS_CALL              = 0x5b,
  R_ARM_TLS_DESCSEQ           = 0x5c,
  R_ARM_THM_TLS_CALL          = 0x5d,
  R_ARM_PLT32_ABS             = 0x5e,
  R_ARM_GOT_ABS               = 0x5f,
  R_ARM_GOT_PREL              = 0x60,
  R_ARM_GOT_BREL12            = 0x61,
  R_ARM_GOTOFF12              = 0x62,
  R_ARM_GOTRELAX              = 0x63,
  R_ARM_GNU_VTENTRY           = 0x64,
  R_ARM_GNU_VTINHERIT         = 0x65,
  R_ARM_THM_JUMP11            = 0x66,
  R_ARM_THM_JUMP8             = 0x67,
  R_ARM_TLS_GD32              = 0x68,
  R_ARM_TLS_LDM32             = 0x69,
  R_ARM_TLS_LDO32             = 0x6a,
  R_ARM_TLS_IE32              = 0x6b,
  R_ARM_TLS_LE32              = 0x6c,
  R_ARM_TLS_LDO12             = 0x6d,
  R_ARM_TLS_LE12              = 0x6e,
  R_ARM_TLS_IE12GP            = 0x6f,
  R_ARM_PRIVATE_0             = 0x70,
  R_ARM_PRIVATE_1             = 0x71,
  R_ARM_PRIVATE_2             = 0x72,
  R_ARM_PRIVATE_3             = 0x73,
  R_ARM_PRIVATE_4             = 0x74,
  R_ARM_PRIVATE_5             = 0x75,
  R_ARM_PRIVATE_6             = 0x76,
  R_ARM_PRIVATE_7             = 0x77,
  R_ARM_PRIVATE_8             = 0x78,
  R_ARM_PRIVATE_9             = 0x79,
  R_ARM_PRIVATE_10            = 0x7a,
  R_ARM_PRIVATE_11            = 0x7b,
  R_ARM_PRIVATE_12            = 0x7c,
  R_ARM_PRIVATE_13            = 0x7d,
  R_ARM_PRIVATE_14            = 0x7e,
  R_ARM_PRIVATE_15            = 0x7f,
  R_ARM_ME_TOO                = 0x80,
  R_ARM_THM_TLS_DESCSEQ16     = 0x81,
  R_ARM_THM_TLS_DESCSEQ32     = 0x82
};

// Mips Specific e_flags
enum {
  EF_MIPS_NOREORDER = 0x00000001, // Don't reorder instructions
  EF_MIPS_PIC       = 0x00000002, // Position independent code
  EF_MIPS_CPIC      = 0x00000004, // Call object with Position independent code
  EF_MIPS_ARCH_1    = 0x00000000, // MIPS1 instruction set
  EF_MIPS_ARCH_2    = 0x10000000, // MIPS2 instruction set
  EF_MIPS_ARCH_3    = 0x20000000, // MIPS3 instruction set
  EF_MIPS_ARCH_4    = 0x30000000, // MIPS4 instruction set
  EF_MIPS_ARCH_5    = 0x40000000, // MIPS5 instruction set
  EF_MIPS_ARCH_32   = 0x50000000, // MIPS32 instruction set per linux not elf.h
  EF_MIPS_ARCH_64   = 0x60000000, // MIPS64 instruction set per linux not elf.h
  EF_MIPS_ARCH_32R2 = 0x70000000, // mips32r2
  EF_MIPS_ARCH_64R2 = 0x80000000, // mips64r2
  EF_MIPS_ARCH      = 0xf0000000  // Mask for applying EF_MIPS_ARCH_ variant
};

// ELF Relocation types for Mips
// .
enum {
  R_MIPS_NONE              =  0,
  R_MIPS_16                =  1,
  R_MIPS_32                =  2,
  R_MIPS_REL32             =  3,
  R_MIPS_26                =  4,
  R_MIPS_HI16              =  5,
  R_MIPS_LO16              =  6,
  R_MIPS_GPREL16           =  7,
  R_MIPS_LITERAL           =  8,
  R_MIPS_GOT16             =  9,
  R_MIPS_GOT               =  9,
  R_MIPS_PC16              = 10,
  R_MIPS_CALL16            = 11,
  R_MIPS_GPREL32           = 12,
  R_MIPS_SHIFT5            = 16,
  R_MIPS_SHIFT6            = 17,
  R_MIPS_64                = 18,
  R_MIPS_GOT_DISP          = 19,
  R_MIPS_GOT_PAGE          = 20,
  R_MIPS_GOT_OFST          = 21,
  R_MIPS_GOT_HI16          = 22,
  R_MIPS_GOT_LO16          = 23,
  R_MIPS_SUB               = 24,
  R_MIPS_INSERT_A          = 25,
  R_MIPS_INSERT_B          = 26,
  R_MIPS_DELETE            = 27,
  R_MIPS_HIGHER            = 28,
  R_MIPS_HIGHEST           = 29,
  R_MIPS_CALL_HI16         = 30,
  R_MIPS_CALL_LO16         = 31,
  R_MIPS_SCN_DISP          = 32,
  R_MIPS_REL16             = 33,
  R_MIPS_ADD_IMMEDIATE     = 34,
  R_MIPS_PJUMP             = 35,
  R_MIPS_RELGOT            = 36,
  R_MIPS_JALR              = 37,
  R_MIPS_TLS_DTPMOD32      = 38,
  R_MIPS_TLS_DTPREL32      = 39,
  R_MIPS_TLS_DTPMOD64      = 40,
  R_MIPS_TLS_DTPREL64      = 41,
  R_MIPS_TLS_GD            = 42,
  R_MIPS_TLS_LDM           = 43,
  R_MIPS_TLS_DTPREL_HI16   = 44,
  R_MIPS_TLS_DTPREL_LO16   = 45,
  R_MIPS_TLS_GOTTPREL      = 46,
  R_MIPS_TLS_TPREL32       = 47,
  R_MIPS_TLS_TPREL64       = 48,
  R_MIPS_TLS_TPREL_HI16    = 49,
  R_MIPS_TLS_TPREL_LO16    = 50,
  R_MIPS_GLOB_DAT          = 51,
  R_MIPS_COPY              = 126,
  R_MIPS_JUMP_SLOT         = 127,
  R_MIPS_NUM               = 218
};

// Hexagon Specific e_flags
// Release 5 ABI
enum {
  // Object processor version flags, bits[3:0]
  EF_HEXAGON_MACH_V2      = 0x00000001,   // Hexagon V2
  EF_HEXAGON_MACH_V3      = 0x00000002,   // Hexagon V3
  EF_HEXAGON_MACH_V4      = 0x00000003,   // Hexagon V4
  EF_HEXAGON_MACH_V5      = 0x00000004,   // Hexagon V5

  // Highest ISA version flags
  EF_HEXAGON_ISA_MACH     = 0x00000000,   // Same as specified in bits[3:0]
                                          // of e_flags
  EF_HEXAGON_ISA_V2       = 0x00000010,   // Hexagon V2 ISA
  EF_HEXAGON_ISA_V3       = 0x00000020,   // Hexagon V3 ISA
  EF_HEXAGON_ISA_V4       = 0x00000030,   // Hexagon V4 ISA
  EF_HEXAGON_ISA_V5       = 0x00000040    // Hexagon V5 ISA
};

// Hexagon specific Section indexes for common small data
// Release 5 ABI 
enum {
  SHN_HEXAGON_SCOMMON     = 0xff00,       // Other access sizes
  SHN_HEXAGON_SCOMMON_1   = 0xff01,       // Byte-sized access
  SHN_HEXAGON_SCOMMON_2   = 0xff02,       // Half-word-sized access
  SHN_HEXAGON_SCOMMON_4   = 0xff03,       // Word-sized access
  SHN_HEXAGON_SCOMMON_8   = 0xff04        // Double-word-size access
};   

// ELF Relocation types for Hexagon
// Release 5 ABI
enum {
  R_HEX_NONE              =  0,
  R_HEX_B22_PCREL         =  1,
  R_HEX_B15_PCREL         =  2,
  R_HEX_B7_PCREL          =  3,
  R_HEX_LO16              =  4,
  R_HEX_HI16              =  5,
  R_HEX_32                =  6,
  R_HEX_16                =  7,
  R_HEX_8                 =  8,
  R_HEX_GPREL16_0         =  9,
  R_HEX_GPREL16_1         =  10,
  R_HEX_GPREL16_2         =  11,
  R_HEX_GPREL16_3         =  12,
  R_HEX_HL16              =  13,
  R_HEX_B13_PCREL         =  14,
  R_HEX_B9_PCREL          =  15,
  R_HEX_B32_PCREL_X       =  16,
  R_HEX_32_6_X            =  17,
  R_HEX_B22_PCREL_X       =  18,
  R_HEX_B15_PCREL_X       =  19,
  R_HEX_B13_PCREL_X       =  20,
  R_HEX_B9_PCREL_X        =  21,
  R_HEX_B7_PCREL_X        =  22,
  R_HEX_16_X              =  23,
  R_HEX_12_X              =  24,
  R_HEX_11_X              =  25,
  R_HEX_10_X              =  26,
  R_HEX_9_X               =  27,
  R_HEX_8_X               =  28,
  R_HEX_7_X               =  29,
  R_HEX_6_X               =  30,
  R_HEX_32_PCREL          =  31,
  R_HEX_COPY              =  32,
  R_HEX_GLOB_DAT          =  33,
  R_HEX_JMP_SLOT          =  34,
  R_HEX_RELATIVE          =  35,
  R_HEX_PLT_B22_PCREL     =  36,
  R_HEX_GOTREL_LO16       =  37,
  R_HEX_GOTREL_HI16       =  38,
  R_HEX_GOTREL_32         =  39,
  R_HEX_GOT_LO16          =  40,
  R_HEX_GOT_HI16          =  41,
  R_HEX_GOT_32            =  42,
  R_HEX_GOT_16            =  43,
  R_HEX_DTPMOD_32         =  44,
  R_HEX_DTPREL_LO16       =  45,
  R_HEX_DTPREL_HI16       =  46,
  R_HEX_DTPREL_32         =  47,
  R_HEX_DTPREL_16         =  48,
  R_HEX_GD_PLT_B22_PCREL  =  49,
  R_HEX_GD_GOT_LO16       =  50,
  R_HEX_GD_GOT_HI16       =  51,
  R_HEX_GD_GOT_32         =  52,
  R_HEX_GD_GOT_16         =  53,
  R_HEX_IE_LO16           =  54,
  R_HEX_IE_HI16           =  55,
  R_HEX_IE_32             =  56,
  R_HEX_IE_GOT_LO16       =  57,
  R_HEX_IE_GOT_HI16       =  58,
  R_HEX_IE_GOT_32         =  59,
  R_HEX_IE_GOT_16         =  60,
  R_HEX_TPREL_LO16        =  61,
  R_HEX_TPREL_HI16        =  62,
  R_HEX_TPREL_32          =  63,
  R_HEX_TPREL_16          =  64,
  R_HEX_6_PCREL_X         =  65,
  R_HEX_GOTREL_32_6_X     =  66,
  R_HEX_GOTREL_16_X       =  67,
  R_HEX_GOTREL_11_X       =  68,
  R_HEX_GOT_32_6_X        =  69,
  R_HEX_GOT_16_X          =  70,
  R_HEX_GOT_11_X          =  71,
  R_HEX_DTPREL_32_6_X     =  72,
  R_HEX_DTPREL_16_X       =  73,
  R_HEX_DTPREL_11_X       =  74,
  R_HEX_GD_GOT_32_6_X     =  75,
  R_HEX_GD_GOT_16_X       =  76,
  R_HEX_GD_GOT_11_X       =  77,
  R_HEX_IE_32_6_X         =  78,
  R_HEX_IE_16_X           =  79,
  R_HEX_IE_GOT_32_6_X     =  80,
  R_HEX_IE_GOT_16_X       =  81,
  R_HEX_IE_GOT_11_X       =  82,
  R_HEX_TPREL_32_6_X      =  83,
  R_HEX_TPREL_16_X        =  84,
  R_HEX_TPREL_11_X        =  85
};

// Section header.
struct Elf32_Shdr {
  Elf32_Word sh_name;      // Section name (index into string table)
  Elf32_Word sh_type;      // Section type (SHT_*)
  Elf32_Word sh_flags;     // Section flags (SHF_*)
  Elf32_Addr sh_addr;      // Address where section is to be loaded
  Elf32_Off  sh_offset;    // File offset of section data, in bytes
  Elf32_Word sh_size;      // Size of section, in bytes
  Elf32_Word sh_link;      // Section type-specific header table index link
  Elf32_Word sh_info;      // Section type-specific extra information
  Elf32_Word sh_addralign; // Section address alignment
  Elf32_Word sh_entsize;   // Size of records contained within the section
};

// Section header for ELF64 - same fields as ELF32, different types.
struct Elf64_Shdr {
  Elf64_Word  sh_name;
  Elf64_Word  sh_type;
  Elf64_Xword sh_flags;
  Elf64_Addr  sh_addr;
  Elf64_Off   sh_offset;
  Elf64_Xword sh_size;
  Elf64_Word  sh_link;
  Elf64_Word  sh_info;
  Elf64_Xword sh_addralign;
  Elf64_Xword sh_entsize;
};

// Special section indices.
enum {
  SHN_UNDEF     = 0,      // Undefined, missing, irrelevant, or meaningless
  SHN_LORESERVE = 0xff00, // Lowest reserved index
  SHN_LOPROC    = 0xff00, // Lowest processor-specific index
  SHN_HIPROC    = 0xff1f, // Highest processor-specific index
  SHN_LOOS      = 0xff20, // Lowest operating system-specific index
  SHN_HIOS      = 0xff3f, // Highest operating system-specific index
  SHN_ABS       = 0xfff1, // Symbol has absolute value; does not need relocation
  SHN_COMMON    = 0xfff2, // FORTRAN COMMON or C external global variables
  SHN_XINDEX    = 0xffff, // Mark that the index is >= SHN_LORESERVE
  SHN_HIRESERVE = 0xffff  // Highest reserved index
};

// Section types.
enum {
  SHT_NULL          = 0,  // No associated section (inactive entry).
  SHT_PROGBITS      = 1,  // Program-defined contents.
  SHT_SYMTAB        = 2,  // Symbol table.
  SHT_STRTAB        = 3,  // String table.
  SHT_RELA          = 4,  // Relocation entries; explicit addends.
  SHT_HASH          = 5,  // Symbol hash table.
  SHT_DYNAMIC       = 6,  // Information for dynamic linking.
  SHT_NOTE          = 7,  // Information about the file.
  SHT_NOBITS        = 8,  // Data occupies no space in the file.
  SHT_REL           = 9,  // Relocation entries; no explicit addends.
  SHT_SHLIB         = 10, // Reserved.
  SHT_DYNSYM        = 11, // Symbol table.
  SHT_INIT_ARRAY    = 14, // Pointers to initialization functions.
  SHT_FINI_ARRAY    = 15, // Pointers to termination functions.
  SHT_PREINIT_ARRAY = 16, // Pointers to pre-init functions.
  SHT_GROUP         = 17, // Section group.
  SHT_SYMTAB_SHNDX  = 18, // Indices for SHN_XINDEX entries.
  SHT_LOOS          = 0x60000000, // Lowest operating system-specific type.
  SHT_GNU_ATTRIBUTES= 0x6ffffff5, // Object attributes.
  SHT_GNU_HASH      = 0x6ffffff6, // GNU-style hash table.
  SHT_GNU_verdef    = 0x6ffffffd, // GNU version definitions.
  SHT_GNU_verneed   = 0x6ffffffe, // GNU version references.
  SHT_GNU_versym    = 0x6fffffff, // GNU symbol versions table.
  SHT_HIOS          = 0x6fffffff, // Highest operating system-specific type.
  SHT_LOPROC        = 0x70000000, // Lowest processor architecture-specific type.
  // Fixme: All this is duplicated in MCSectionELF. Why??
  // Exception Index table
  SHT_ARM_EXIDX           = 0x70000001U,
  // BPABI DLL dynamic linking pre-emption map
  SHT_ARM_PREEMPTMAP      = 0x70000002U,
  //  Object file compatibility attributes
  SHT_ARM_ATTRIBUTES      = 0x70000003U,
  SHT_ARM_DEBUGOVERLAY    = 0x70000004U,
  SHT_ARM_OVERLAYSECTION  = 0x70000005U,

  SHT_X86_64_UNWIND       = 0x70000001, // Unwind information

  SHT_HIPROC        = 0x7fffffff, // Highest processor architecture-specific type.
  SHT_LOUSER        = 0x80000000, // Lowest type reserved for applications.
  SHT_HIUSER        = 0xffffffff  // Highest type reserved for applications.
};

// Section flags.
enum {
  // Section data should be writable during execution.
  SHF_WRITE = 0x1,

  // Section occupies memory during program execution.
  SHF_ALLOC = 0x2,

  // Section contains executable machine instructions.
  SHF_EXECINSTR = 0x4,

  // The data in this section may be merged.
  SHF_MERGE = 0x10,

  // The data in this section is null-terminated strings.
  SHF_STRINGS = 0x20,

  // A field in this section holds a section header table index.
  SHF_INFO_LINK = 0x40U,

  // Adds special ordering requirements for link editors.
  SHF_LINK_ORDER = 0x80U,

  // This section requires special OS-specific processing to avoid incorrect
  // behavior.
  SHF_OS_NONCONFORMING = 0x100U,

  // This section is a member of a section group.
  SHF_GROUP = 0x200U,

  // This section holds Thread-Local Storage.
  SHF_TLS = 0x400U,

  // Start of target-specific flags.

  /// XCORE_SHF_CP_SECTION - All sections with the "c" flag are grouped
  /// together by the linker to form the constant pool and the cp register is
  /// set to the start of the constant pool by the boot code.
  XCORE_SHF_CP_SECTION = 0x800U,

  /// XCORE_SHF_DP_SECTION - All sections with the "d" flag are grouped
  /// together by the linker to form the data section and the dp register is
  /// set to the start of the section by the boot code.
  XCORE_SHF_DP_SECTION = 0x1000U,

  SHF_MASKOS   = 0x0ff00000,

  // Bits indicating processor-specific flags.
  SHF_MASKPROC = 0xf0000000,

  // If an object file section does not have this flag set, then it may not hold
  // more than 2GB and can be freely referred to in objects using smaller code
  // models. Otherwise, only objects using larger code models can refer to them.
  // For example, a medium code model object can refer to data in a section that
  // sets this flag besides being able to refer to data in a section that does
  // not set it; likewise, a small code model object can refer only to code in a
  // section that does not set this flag.
  SHF_X86_64_LARGE = 0x10000000
};

// Section Group Flags
enum {
  GRP_COMDAT = 0x1,
  GRP_MASKOS = 0x0ff00000,
  GRP_MASKPROC = 0xf0000000
};

// Symbol table entries for ELF32.
struct Elf32_Sym {
  Elf32_Word    st_name;  // Symbol name (index into string table)
  Elf32_Addr    st_value; // Value or address associated with the symbol
  Elf32_Word    st_size;  // Size of the symbol
  unsigned char st_info;  // Symbol's type and binding attributes
  unsigned char st_other; // Must be zero; reserved
  Elf32_Half    st_shndx; // Which section (header table index) it's defined in

  // These accessors and mutators correspond to the ELF32_ST_BIND,
  // ELF32_ST_TYPE, and ELF32_ST_INFO macros defined in the ELF specification:
  unsigned char getBinding() const { return st_info >> 4; }
  unsigned char getType() const { return st_info & 0x0f; }
  void setBinding(unsigned char b) { setBindingAndType(b, getType()); }
  void setType(unsigned char t) { setBindingAndType(getBinding(), t); }
  void setBindingAndType(unsigned char b, unsigned char t) {
    st_info = (b << 4) + (t & 0x0f);
  }
};

// Symbol table entries for ELF64.
struct Elf64_Sym {
  Elf64_Word      st_name;  // Symbol name (index into string table)
  unsigned char   st_info;  // Symbol's type and binding attributes
  unsigned char   st_other; // Must be zero; reserved
  Elf64_Half      st_shndx; // Which section (header table index) it's defined in
  Elf64_Addr      st_value; // Value or address associated with the symbol
  Elf64_Xword     st_size;  // Size of the symbol

  // These accessors and mutators are identical to those defined for ELF32
  // symbol table entries.
  unsigned char getBinding() const { return st_info >> 4; }
  unsigned char getType() const { return st_info & 0x0f; }
  void setBinding(unsigned char b) { setBindingAndType(b, getType()); }
  void setType(unsigned char t) { setBindingAndType(getBinding(), t); }
  void setBindingAndType(unsigned char b, unsigned char t) {
    st_info = (b << 4) + (t & 0x0f);
  }
};

// The size (in bytes) of symbol table entries.
enum {
  SYMENTRY_SIZE32 = 16, // 32-bit symbol entry size
  SYMENTRY_SIZE64 = 24  // 64-bit symbol entry size.
};

// Symbol bindings.
enum {
  STB_LOCAL = 0,   // Local symbol, not visible outside obj file containing def
  STB_GLOBAL = 1,  // Global symbol, visible to all object files being combined
  STB_WEAK = 2,    // Weak symbol, like global but lower-precedence
  STB_LOOS   = 10, // Lowest operating system-specific binding type
  STB_HIOS   = 12, // Highest operating system-specific binding type
  STB_LOPROC = 13, // Lowest processor-specific binding type
  STB_HIPROC = 15  // Highest processor-specific binding type
};

// Symbol types.
enum {
  STT_NOTYPE  = 0,   // Symbol's type is not specified
  STT_OBJECT  = 1,   // Symbol is a data object (variable, array, etc.)
  STT_FUNC    = 2,   // Symbol is executable code (function, etc.)
  STT_SECTION = 3,   // Symbol refers to a section
  STT_FILE    = 4,   // Local, absolute symbol that refers to a file
  STT_COMMON  = 5,   // An uninitialized common block
  STT_TLS     = 6,   // Thread local data object
  STT_LOOS    = 7,   // Lowest operating system-specific symbol type
  STT_HIOS    = 8,   // Highest operating system-specific symbol type
  STT_GNU_IFUNC = 10, // GNU indirect function
  STT_LOPROC  = 13,  // Lowest processor-specific symbol type
  STT_HIPROC  = 15   // Highest processor-specific symbol type
};

enum {
  STV_DEFAULT   = 0,  // Visibility is specified by binding type
  STV_INTERNAL  = 1,  // Defined by processor supplements
  STV_HIDDEN    = 2,  // Not visible to other components
  STV_PROTECTED = 3   // Visible in other components but not preemptable
};

// Relocation entry, without explicit addend.
struct Elf32_Rel {
  Elf32_Addr r_offset; // Location (file byte offset, or program virtual addr)
  Elf32_Word r_info;   // Symbol table index and type of relocation to apply

  // These accessors and mutators correspond to the ELF32_R_SYM, ELF32_R_TYPE,
  // and ELF32_R_INFO macros defined in the ELF specification:
  Elf32_Word getSymbol() const { return (r_info >> 8); }
  unsigned char getType() const { return (unsigned char) (r_info & 0x0ff); }
  void setSymbol(Elf32_Word s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(Elf32_Word s, unsigned char t) {
    r_info = (s << 8) + t;
  }
};

// Relocation entry with explicit addend.
struct Elf32_Rela {
  Elf32_Addr  r_offset; // Location (file byte offset, or program virtual addr)
  Elf32_Word  r_info;   // Symbol table index and type of relocation to apply
  Elf32_Sword r_addend; // Compute value for relocatable field by adding this

  // These accessors and mutators correspond to the ELF32_R_SYM, ELF32_R_TYPE,
  // and ELF32_R_INFO macros defined in the ELF specification:
  Elf32_Word getSymbol() const { return (r_info >> 8); }
  unsigned char getType() const { return (unsigned char) (r_info & 0x0ff); }
  void setSymbol(Elf32_Word s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(Elf32_Word s, unsigned char t) {
    r_info = (s << 8) + t;
  }
};

// Relocation entry, without explicit addend.
struct Elf64_Rel {
  Elf64_Addr r_offset; // Location (file byte offset, or program virtual addr).
  Elf64_Xword r_info;   // Symbol table index and type of relocation to apply.

  // These accessors and mutators correspond to the ELF64_R_SYM, ELF64_R_TYPE,
  // and ELF64_R_INFO macros defined in the ELF specification:
  Elf64_Xword getSymbol() const { return (r_info >> 32); }
  unsigned char getType() const {
    return (unsigned char) (r_info & 0xffffffffL);
  }
  void setSymbol(Elf32_Word s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(Elf64_Xword s, unsigned char t) {
    r_info = (s << 32) + (t&0xffffffffL);
  }
};

// Relocation entry with explicit addend.
struct Elf64_Rela {
  Elf64_Addr  r_offset; // Location (file byte offset, or program virtual addr).
  Elf64_Xword  r_info;   // Symbol table index and type of relocation to apply.
  Elf64_Sxword r_addend; // Compute value for relocatable field by adding this.

  // These accessors and mutators correspond to the ELF64_R_SYM, ELF64_R_TYPE,
  // and ELF64_R_INFO macros defined in the ELF specification:
  Elf64_Xword getSymbol() const { return (r_info >> 32); }
  unsigned char getType() const {
    return (unsigned char) (r_info & 0xffffffffL);
  }
  void setSymbol(Elf64_Xword s) { setSymbolAndType(s, getType()); }
  void setType(unsigned char t) { setSymbolAndType(getSymbol(), t); }
  void setSymbolAndType(Elf64_Xword s, unsigned char t) {
    r_info = (s << 32) + (t&0xffffffffL);
  }
};

// Program header for ELF32.
struct Elf32_Phdr {
  Elf32_Word p_type;   // Type of segment
  Elf32_Off  p_offset; // File offset where segment is located, in bytes
  Elf32_Addr p_vaddr;  // Virtual address of beginning of segment
  Elf32_Addr p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf32_Word p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf32_Word p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf32_Word p_flags;  // Segment flags
  Elf32_Word p_align;  // Segment alignment constraint
};

// Program header for ELF64.
struct Elf64_Phdr {
  Elf64_Word   p_type;   // Type of segment
  Elf64_Word   p_flags;  // Segment flags
  Elf64_Off    p_offset; // File offset where segment is located, in bytes
  Elf64_Addr   p_vaddr;  // Virtual address of beginning of segment
  Elf64_Addr   p_paddr;  // Physical address of beginning of segment (OS-specific)
  Elf64_Xword  p_filesz; // Num. of bytes in file image of segment (may be zero)
  Elf64_Xword  p_memsz;  // Num. of bytes in mem image of segment (may be zero)
  Elf64_Xword  p_align;  // Segment alignment constraint
};

// Segment types.
enum {
  PT_NULL    = 0, // Unused segment.
  PT_LOAD    = 1, // Loadable segment.
  PT_DYNAMIC = 2, // Dynamic linking information.
  PT_INTERP  = 3, // Interpreter pathname.
  PT_NOTE    = 4, // Auxiliary information.
  PT_SHLIB   = 5, // Reserved.
  PT_PHDR    = 6, // The program header table itself.
  PT_TLS     = 7, // The thread-local storage template.
  PT_LOOS    = 0x60000000, // Lowest operating system-specific pt entry type.
  PT_HIOS    = 0x6fffffff, // Highest operating system-specific pt entry type.
  PT_LOPROC  = 0x70000000, // Lowest processor-specific program hdr entry type.
  PT_HIPROC  = 0x7fffffff, // Highest processor-specific program hdr entry type.

  // x86-64 program header types.
  // These all contain stack unwind tables.
  PT_GNU_EH_FRAME  = 0x6474e550,
  PT_SUNW_EH_FRAME = 0x6474e550,
  PT_SUNW_UNWIND   = 0x6464e550,

  PT_GNU_STACK  = 0x6474e551, // Indicates stack executability.
  PT_GNU_RELRO  = 0x6474e552, // Read-only after relocation.

  // ARM program header types.
  PT_ARM_ARCHEXT = 0x70000000, // Platform architecture compatibility information
  // These all contain stack unwind tables.
  PT_ARM_EXIDX   = 0x70000001,
  PT_ARM_UNWIND  = 0x70000001
};

// Segment flag bits.
enum {
  PF_X        = 1,         // Execute
  PF_W        = 2,         // Write
  PF_R        = 4,         // Read
  PF_MASKOS   = 0x0ff00000,// Bits for operating system-specific semantics.
  PF_MASKPROC = 0xf0000000 // Bits for processor-specific semantics.
};

// Dynamic table entry for ELF32.
struct Elf32_Dyn
{
  Elf32_Sword d_tag;            // Type of dynamic table entry.
  union
  {
      Elf32_Word d_val;         // Integer value of entry.
      Elf32_Addr d_ptr;         // Pointer value of entry.
  } d_un;
};

// Dynamic table entry for ELF64.
struct Elf64_Dyn
{
  Elf64_Sxword d_tag;           // Type of dynamic table entry.
  union
  {
      Elf64_Xword d_val;        // Integer value of entry.
      Elf64_Addr  d_ptr;        // Pointer value of entry.
  } d_un;
};

// Dynamic table entry tags.
enum {
  DT_NULL         = 0,        // Marks end of dynamic array.
  DT_NEEDED       = 1,        // String table offset of needed library.
  DT_PLTRELSZ     = 2,        // Size of relocation entries in PLT.
  DT_PLTGOT       = 3,        // Address associated with linkage table.
  DT_HASH         = 4,        // Address of symbolic hash table.
  DT_STRTAB       = 5,        // Address of dynamic string table.
  DT_SYMTAB       = 6,        // Address of dynamic symbol table.
  DT_RELA         = 7,        // Address of relocation table (Rela entries).
  DT_RELASZ       = 8,        // Size of Rela relocation table.
  DT_RELAENT      = 9,        // Size of a Rela relocation entry.
  DT_STRSZ        = 10,       // Total size of the string table.
  DT_SYMENT       = 11,       // Size of a symbol table entry.
  DT_INIT         = 12,       // Address of initialization function.
  DT_FINI         = 13,       // Address of termination function.
  DT_SONAME       = 14,       // String table offset of a shared objects name.
  DT_RPATH        = 15,       // String table offset of library search path.
  DT_SYMBOLIC     = 16,       // Changes symbol resolution algorithm.
  DT_REL          = 17,       // Address of relocation table (Rel entries).
  DT_RELSZ        = 18,       // Size of Rel relocation table.
  DT_RELENT       = 19,       // Size of a Rel relocation entry.
  DT_PLTREL       = 20,       // Type of relocation entry used for linking.
  DT_DEBUG        = 21,       // Reserved for debugger.
  DT_TEXTREL      = 22,       // Relocations exist for non-writable segments.
  DT_JMPREL       = 23,       // Address of relocations associated with PLT.
  DT_BIND_NOW     = 24,       // Process all relocations before execution.
  DT_INIT_ARRAY   = 25,       // Pointer to array of initialization functions.
  DT_FINI_ARRAY   = 26,       // Pointer to array of termination functions.
  DT_INIT_ARRAYSZ = 27,       // Size of DT_INIT_ARRAY.
  DT_FINI_ARRAYSZ = 28,       // Size of DT_FINI_ARRAY.
  DT_RUNPATH      = 29,       // String table offset of lib search path.
  DT_FLAGS        = 30,       // Flags.
  DT_ENCODING     = 32,       // Values from here to DT_LOOS follow the rules
                              // for the interpretation of the d_un union.

  DT_PREINIT_ARRAY = 32,      // Pointer to array of preinit functions.
  DT_PREINIT_ARRAYSZ = 33,    // Size of the DT_PREINIT_ARRAY array.

  DT_LOOS         = 0x60000000, // Start of environment specific tags.
  DT_HIOS         = 0x6FFFFFFF, // End of environment specific tags.
  DT_LOPROC       = 0x70000000, // Start of processor specific tags.
  DT_HIPROC       = 0x7FFFFFFF, // End of processor specific tags.

  DT_RELACOUNT    = 0x6FFFFFF9, // ELF32_Rela count.
  DT_RELCOUNT     = 0x6FFFFFFA, // ELF32_Rel count.

  DT_FLAGS_1      = 0X6FFFFFFB, // Flags_1.
  DT_VERDEF       = 0X6FFFFFFC, // The address of the version definition table.
  DT_VERDEFNUM    = 0X6FFFFFFD, // The number of entries in DT_VERDEF.
  DT_VERNEED      = 0X6FFFFFFE, // The address of the version Dependency table.
  DT_VERNEEDNUM   = 0X6FFFFFFF  // The number of entries in DT_VERNEED.
};

// DT_FLAGS values.
enum {
  DF_ORIGIN     = 0x01, // The object may reference $ORIGIN.
  DF_SYMBOLIC   = 0x02, // Search the shared lib before searching the exe.
  DF_TEXTREL    = 0x04, // Relocations may modify a non-writable segment.
  DF_BIND_NOW   = 0x08, // Process all relocations on load.
  DF_STATIC_TLS = 0x10  // Reject attempts to load dynamically.
};

// State flags selectable in the `d_un.d_val' element of the DT_FLAGS_1 entry.
enum {
  DF_1_NOW        = 0x00000001, // Set RTLD_NOW for this object.
  DF_1_GLOBAL     = 0x00000002, // Set RTLD_GLOBAL for this object.
  DF_1_GROUP      = 0x00000004, // Set RTLD_GROUP for this object.
  DF_1_NODELETE   = 0x00000008, // Set RTLD_NODELETE for this object.
  DF_1_LOADFLTR   = 0x00000010, // Trigger filtee loading at runtime.
  DF_1_INITFIRST  = 0x00000020, // Set RTLD_INITFIRST for this object.
  DF_1_NOOPEN     = 0x00000040, // Set RTLD_NOOPEN for this object.
  DF_1_ORIGIN     = 0x00000080, // $ORIGIN must be handled.
  DF_1_DIRECT     = 0x00000100, // Direct binding enabled.
  DF_1_TRANS      = 0x00000200,
  DF_1_INTERPOSE  = 0x00000400, // Object is used to interpose.
  DF_1_NODEFLIB   = 0x00000800, // Ignore default lib search path.
  DF_1_NODUMP     = 0x00001000, // Object can't be dldump'ed.
  DF_1_CONFALT    = 0x00002000, // Configuration alternative created.
  DF_1_ENDFILTEE  = 0x00004000, // Filtee terminates filters search.
  DF_1_DISPRELDNE = 0x00008000, // Disp reloc applied at build time.
  DF_1_DISPRELPND = 0x00010000  // Disp reloc applied at run-time.
};

// ElfXX_VerDef structure version (GNU versioning)
enum {
  VER_DEF_NONE    = 0,
  VER_DEF_CURRENT = 1
};

// VerDef Flags (ElfXX_VerDef::vd_flags)
enum {
  VER_FLG_BASE = 0x1,
  VER_FLG_WEAK = 0x2,
  VER_FLG_INFO = 0x4
};

// Special constants for the version table. (SHT_GNU_versym/.gnu.version)
enum {
  VER_NDX_LOCAL  = 0,      // Unversioned local symbol
  VER_NDX_GLOBAL = 1,      // Unversioned global symbol
  VERSYM_VERSION = 0x7fff, // Version Index mask
  VERSYM_HIDDEN  = 0x8000  // Hidden bit (non-default version)
};

// ElfXX_VerNeed structure version (GNU versioning)
enum {
  VER_NEED_NONE = 0,
  VER_NEED_CURRENT = 1
};

} // end namespace ELF

} // end namespace llvm

#endif
