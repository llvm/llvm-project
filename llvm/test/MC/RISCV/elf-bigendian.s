# RUN: llvm-mc %s -filetype=obj -triple=riscv32be | llvm-readobj -h - \
# RUN:     | FileCheck -check-prefix=RV32BE %s
# RUN: llvm-mc %s -filetype=obj -triple=riscv64be | llvm-readobj -h - \
# RUN:     | FileCheck -check-prefix=RV64BE %s

# Test that RISC-V big-endian targets produce correct ELF headers

# RV32BE: Format: elf32-bigriscv
# RV32BE: Arch: riscv32
# RV32BE: AddressSize: 32bit
# RV32BE: ElfHeader {
# RV32BE:   Ident {
# RV32BE:     Magic: (7F 45 4C 46)
# RV32BE:     Class: 32-bit (0x1)
# RV32BE:     DataEncoding: BigEndian (0x2)
# RV32BE:     FileVersion: 1
# RV32BE:     OS/ABI: SystemV (0x0)
# RV32BE:     ABIVersion: 0
# RV32BE:   }
# RV32BE:   Type: Relocatable (0x1)
# RV32BE:   Machine: EM_RISCV (0xF3)
# RV32BE:   Version: 1
# RV32BE:   Flags [ (0x0)
# RV32BE:   ]
# RV32BE: }

# RV64BE: Format: elf64-bigriscv
# RV64BE: Arch: riscv64
# RV64BE: AddressSize: 64bit
# RV64BE: ElfHeader {
# RV64BE:   Ident {
# RV64BE:     Magic: (7F 45 4C 46)
# RV64BE:     Class: 64-bit (0x2)
# RV64BE:     DataEncoding: BigEndian (0x2)
# RV64BE:     FileVersion: 1
# RV64BE:     OS/ABI: SystemV (0x0)
# RV64BE:     ABIVersion: 0
# RV64BE:   }
# RV64BE:   Type: Relocatable (0x1)
# RV64BE:   Machine: EM_RISCV (0xF3)
# RV64BE:   Version: 1
# RV64BE:   Flags [ (0x0)
# RV64BE:   ]
# RV64BE: }

nop
