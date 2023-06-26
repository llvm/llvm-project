; RUN: sed -e "s,VERSION,0,g"   %s | llvm-mc -triple=arm64e-apple-ios -filetype=obj - -o - | llvm-objdump --macho -d -p - | FileCheck %s --check-prefix=V0
; RUN: sed -e "s,VERSION,15,g" %s | llvm-mc -triple=arm64e-apple-ios -filetype=obj - -o - | llvm-objdump --macho -d -p - | FileCheck %s --check-prefix=V15
; RUN: sed -e "s,VERSION,64,g" %s | not llvm-mc -triple=arm64e-apple-ios -filetype=obj - -o - 2>&1 | FileCheck %s --check-prefix=V64

; V15: Mach header
; V15:       magic cputype cpusubtype  caps    filetype ncmds sizeofcmds      flags
; V15: MH_MAGIC_64   ARM64          E  PAC15     OBJECT     3        256 0x00000000

; V0: Mach header
; V0:       magic cputype cpusubtype  caps    filetype ncmds sizeofcmds      flags
; V0: MH_MAGIC_64   ARM64          E  PAC00     OBJECT     3        256 0x00000000

; V64: error: invalid ptrauth ABI version number

.ptrauth_abi_version VERSION
