# Default ABI for MIPS32 is O32.
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips1                                    %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985 %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips2                                    %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS2 %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips3                                    %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS3,32BITMODE %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips4                                    %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS4,32BITMODE %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips5                                    %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS5,32BITMODE %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux                                                    %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS32R1 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux                                                  %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R1 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux                                                  %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R1 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r2                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r3                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R3 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r5                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R5 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r2                 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r3                 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R3 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r5                 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R5 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips32r6                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R6 %s

# Selected ABI O32 takes precedence over target triple.
# FIXME: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips1    -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985 %s
# FIXME: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips2    -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS2 %s
# FIXME: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips3    -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS3,32BITMODE %s
# FIXME: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips4    -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS4,32BITMODE %s
# FIXME: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips5    -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS5,32BITMODE %s
# FIXME: llvm-mc -filetype=obj -triple mips64-unknown-linux                  -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,O32,NAN1985,MIPS32R1 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux                -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R1 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux                -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R1 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r2 -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R2 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r3 -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R3 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r5 -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN1985,MIPS32R5 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r2 -target-abi=o32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R2 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r3 -target-abi=o32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R3 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r5 -target-abi=o32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R5 %s
# FIXME: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips32r6 -target-abi=o32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,O32,NAN2008,MIPS32R6 %s

# Default ABI for MIPS64 is N64 as opposed to GCC/GAS (N32).
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips3    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS3 %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips4    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS4 %s
# RUN: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips5    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS5 %s
# FIXME: llvm-mc -filetype=obj -triple mips-unknown-linux                    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS64R1 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux                  -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R1 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r2 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r3 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R3 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r5 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R5 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r2 -target-abi=n32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r3 -target-abi=n32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R3 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r5 -target-abi=n32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R5 %s
# RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r6 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R6 %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips3    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS3 %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips4    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS4 %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips5    -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS5 %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux                  -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,BE,N32,NAN1985,MIPS64R1 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux                -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R1 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r3 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R3 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r5 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN1985,MIPS64R5 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2 -target-abi=n32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r3 -target-abi=n32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R3 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r5 -target-abi=n32 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R5 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r6 -target-abi=n32                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF32,LE,N32,NAN2008,MIPS64R6 %s

# Default ABI for MIPS64 is N64 as opposed to GCC/GAS (N32).
# FIXME: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips3    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS3    %s
# FIXME: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips4    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS4    %s
# FIXME: llvm-mc -filetype=obj -triple mips-unknown-linux     -mcpu=mips5    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS5    %s
# FIXME: llvm-mc -filetype=obj -triple mips-unknown-linux                    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS64R1 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux                  -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R1 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r2 -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R2 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r3 -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R3 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r5 -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R5 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r2 -target-abi=n64 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R2 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r3 -target-abi=n64 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R3 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r5 -target-abi=n64 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R5 %s
# FIXME: llvm-mc -filetype=obj -triple mipsel-unknown-linux   -mcpu=mips64r6 -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R6 %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips3    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS3    %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips4    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS4    %s
# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux   -mcpu=mips5    -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS5    %s

# RUN: llvm-mc -filetype=obj -triple mips64-unknown-linux                                                  %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,BE,N64,NAN1985,MIPS64R1 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux                                                %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R1 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r3                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R3 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r5                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,MIPS64R5 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r2                 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R2 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r3                 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R3 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r5                 -mattr=+nan2008 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R5 %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=mips64r6                                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN2008,MIPS64R6 %s

# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux -mcpu=octeon   -target-abi=n64                 %s -o - | llvm-readobj -h - | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,OCTEON   %s
# RUN: llvm-mc -filetype=obj -triple mips64el-unknown-linux \
# RUN:         -mcpu=octeon+ -target-abi=n64 %s -o - \
# RUN:   | llvm-readobj -h - \
# RUN:   | FileCheck --check-prefixes=ALL,ELF64,LE,N64,NAN1985,OCTEON %s

# ALL:        ElfHeader {
# ALL-NEXT:     Ident {
# ALL-NEXT:       Magic: (7F 45 4C 46)
# ELF32-NEXT:     Class: 32-bit
# ELF64-NEXT:     Class: 64-bit
# LE-NEXT:        DataEncoding: LittleEndian
# BE-NEXT:        DataEncoding: BigEndian
# ALL-NEXT:       FileVersion: 1
# ALL-NEXT:       OS/ABI: SystemV
# ALL-NEXT:       ABIVersion: 0
# ALL-NEXT:       Unused: (00 00 00 00 00 00 00)
# ALL-NEXT:     }
# ALL-NEXT:     Type: Relocatable
# ALL-NEXT:     Machine: EM_MIPS
# ALL-NEXT:     Version: 1
# ALL-NEXT:     Entry: 0x0
# ALL-NEXT:     ProgramHeaderOffset: 0x0
# ALL-NEXT:     SectionHeaderOffset:
# ALL-NEXT:     Flags [
# 32BITMODE-NEXT: EF_MIPS_32BITMODE
# N64-NOT:        EF_MIPS_32BITMODE
# N32-NEXT:       EF_MIPS_ABI2
# O32-NEXT:       EF_MIPS_ABI_O32
# N64-NOT:        EF_MIPS_ABI2
# N64-NOT:        EF_MIPS_ABI_O32

# MIPS2-NEXT:     EF_MIPS_ARCH_2
# MIPS3-NEXT:     EF_MIPS_ARCH_3
# MIPS4-NEXT:     EF_MIPS_ARCH_4
# MIPS5-NEXT:     EF_MIPS_ARCH_5
# MIPS32R1-NEXT:  EF_MIPS_ARCH_32
# MIPS32R2-NEXT:  EF_MIPS_ARCH_32R2
# The R2 flag is reused for R3 and R5.
# MIPS32R3-NEXT:  EF_MIPS_ARCH_32R2
# MIPS32R5-NEXT:  EF_MIPS_ARCH_32R2
# MIPS32R6-NEXT:  EF_MIPS_ARCH_32R6
# MIPS64R1-NEXT:  EF_MIPS_ARCH_64
# MIPS64R2-NEXT:  EF_MIPS_ARCH_64R2
# The R2 flag is reused for R3 and R5.
# MIPS64R3-NEXT:  EF_MIPS_ARCH_64R2
# MIPS64R5-NEXT:  EF_MIPS_ARCH_64R2
# MIPS64R6-NEXT:  EF_MIPS_ARCH_64R6
# OCTEON-NEXT:    EF_MIPS_ARCH_64R2

# ALL-NEXT:       EF_MIPS_CPIC

# OCTEON-NEXT:    EF_MIPS_MACH_OCTEON

# NAN1985-NOT:    EF_MIPS_NAN2008
# NAN2008-NEXT:   EF_MIPS_NAN2008
# ALL-NEXT:     ]
