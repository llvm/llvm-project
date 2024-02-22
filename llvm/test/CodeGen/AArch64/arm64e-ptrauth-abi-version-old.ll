; RUN: llc -mtriple arm64e-apple-darwin -o - %s | FileCheck %s
; RUN: llc -filetype=obj -mtriple arm64e-apple-darwin -o - %s | llvm-objdump --macho -d -p - | FileCheck %s --check-prefix=OBJ

; CHECK-NOT: .ptrauth_abi_version

; OBJ: Mach header
; OBJ:       magic cputype cpusubtype  caps    filetype ncmds sizeofcmds      flags
; OBJ: MH_MAGIC_64   ARM64          E  0x00      OBJECT     3        256 SUBSECTIONS_VIA_SYMBOLS

!0 = !{ i32 -1, i1 false }
!1 = !{ !0 }
!2 = !{ i32 6, !"ptrauth.abi-version", !1 }
!llvm.module.flags = !{ !2 }
