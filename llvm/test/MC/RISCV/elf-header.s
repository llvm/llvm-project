# RUN: llvm-mc %s -filetype=obj -triple=riscv32 | llvm-readobj -h - \
# RUN:     | FileCheck -check-prefix=RV32 %s
# RUN: llvm-mc %s -filetype=obj -triple=riscv64 | llvm-readobj -h - \
# RUN:     | FileCheck -check-prefix=RV64 %s

# RV32: Format: elf32-littleriscv
# RV32: Arch: riscv32
# RV32: AddressSize: 32bit
# RV32: ElfHeader {
# RV32:   Ident {
# RV32:     Magic: (7F 45 4C 46)
# RV32:     Class: 32-bit (0x1)
# RV32:     DataEncoding: LittleEndian (0x1)
# RV32:     FileVersion: 1
# RV32:     OS/ABI: SystemV (0x0)
# RV32:     ABIVersion: 0
# RV32:   }
# RV32:   Type: Relocatable (0x1)
# RV32:   Machine: EM_RISCV (0xF3)
# RV32:   Version: 1
# RV32:   Flags [ (0x0)
# RV32:   ]
# RV32: }

# RV64: Format: elf64-littleriscv
# RV64: Arch: riscv64
# RV64: AddressSize: 64bit
# RV64: ElfHeader {
# RV64:   Ident {
# RV64:     Magic: (7F 45 4C 46)
# RV64:     Class: 64-bit (0x2)
# RV64:     DataEncoding: LittleEndian (0x1)
# RV64:     FileVersion: 1
# RV64:     OS/ABI: SystemV (0x0)
# RV64:     ABIVersion: 0
# RV64:   }
# RV64:   Type: Relocatable (0x1)
# RV64:   Machine: EM_RISCV (0xF3)
# RV64:   Version: 1
# RV64:   Flags [ (0x0)
# RV64:   ]
# RV64: }
