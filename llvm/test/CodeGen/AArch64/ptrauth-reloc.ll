; RUN: rm -rf %t && split-file %s %t && cd %t

;--- ok.ll

; RUN: llc < ok.ll -mtriple arm64e-apple-darwin \
; RUN:   | FileCheck %s --check-prefix=CHECK-MACHO
; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   | FileCheck %s --check-prefix=CHECK-ELF

; RUN: llc < ok.ll -mtriple arm64e-apple-darwin \
; RUN:   -global-isel -verify-machineinstrs -global-isel-abort=1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-MACHO
; RUN: llc < ok.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel -verify-machineinstrs -global-isel-abort=1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ELF

@g = external global i32

@g_weak = extern_weak global i32

@g_strong_def = constant i32 42

; CHECK-ELF-LABEL:     .globl g.ref.ia.0
; CHECK-ELF-NEXT:      .p2align 4
; CHECK-ELF-NEXT:    g.ref.ia.0:
; CHECK-ELF-NEXT:      .xword 5
; CHECK-ELF-NEXT:      .xword g@AUTH(ia,0)
; CHECK-ELF-NEXT:      .xword 6

; CHECK-MACHO-LABEL:   .section __DATA,__const
; CHECK-MACHO-NEXT:    .globl _g.ref.ia.0
; CHECK-MACHO-NEXT:    .p2align 4
; CHECK-MACHO-NEXT:  _g.ref.ia.0:
; CHECK-MACHO-NEXT:    .quad 5
; CHECK-MACHO-NEXT:    .quad _g@AUTH(ia,0)
; CHECK-MACHO-NEXT:    .quad 6

@g.ref.ia.0 = constant { i64, ptr, i64 } { i64 5, ptr ptrauth (ptr @g, i32 0), i64 6 }

; CHECK-ELF-LABEL:     .globl g.ref.ia.42
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g.ref.ia.42:
; CHECK-ELF-NEXT:      .xword g@AUTH(ia,42)

; CHECK-MACHO-LABEL:   .globl _g.ref.ia.42
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g.ref.ia.42:
; CHECK-MACHO-NEXT:    .quad _g@AUTH(ia,42)

@g.ref.ia.42 = constant ptr ptrauth (ptr @g, i32 0, i64 42)

; CHECK-ELF-LABEL:     .globl g.ref.ib.0
; CHECK-ELF-NEXT:      .p2align 4
; CHECK-ELF-NEXT:    g.ref.ib.0:
; CHECK-ELF-NEXT:      .xword 5
; CHECK-ELF-NEXT:      .xword g@AUTH(ib,0)
; CHECK-ELF-NEXT:      .xword 6

; CHECK-MACHO-LABEL:   .globl _g.ref.ib.0
; CHECK-MACHO-NEXT:    .p2align 4
; CHECK-MACHO-NEXT:  _g.ref.ib.0:
; CHECK-MACHO-NEXT:    .quad 5
; CHECK-MACHO-NEXT:    .quad _g@AUTH(ib,0)
; CHECK-MACHO-NEXT:    .quad 6

@g.ref.ib.0 = constant { i64, ptr, i64 } { i64 5, ptr ptrauth (ptr @g, i32 1, i64 0), i64 6 }

; CHECK-ELF-LABEL:     .globl g.ref.da.42.addr
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g.ref.da.42.addr:
; CHECK-ELF-NEXT:      .xword g@AUTH(da,42,addr)

; CHECK-MACHO-LABEL:   .globl _g.ref.da.42.addr
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g.ref.da.42.addr:
; CHECK-MACHO-NEXT:    .quad _g@AUTH(da,42,addr)

@g.ref.da.42.addr = constant ptr ptrauth (ptr @g, i32 2, i64 42, ptr @g.ref.da.42.addr)

; CHECK-ELF-LABEL:     .globl g.offset.ref.da.0
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g.offset.ref.da.0:
; CHECK-ELF-NEXT:      .xword (g+16)@AUTH(da,0)

; CHECK-MACHO-LABEL:   .globl _g.offset.ref.da.0
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g.offset.ref.da.0:
; CHECK-MACHO-NEXT:    .quad (_g+16)@AUTH(da,0)

@g.offset.ref.da.0 = constant ptr ptrauth (i8* getelementptr (i8, ptr @g, i64 16), i32 2)

; CHECK-ELF-LABEL:     .globl g.big_offset.ref.da.0
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g.big_offset.ref.da.0:
; CHECK-ELF-NEXT:      .xword (g+2147549185)@AUTH(da,0)

; CHECK-MACHO-LABEL:   .globl _g.big_offset.ref.da.0
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g.big_offset.ref.da.0:
; CHECK-MACHO-NEXT:    .quad (_g+2147549185)@AUTH(da,0)

@g.big_offset.ref.da.0 = constant ptr ptrauth (i8* getelementptr (i8, ptr @g, i64 add (i64 2147483648, i64 65537)), i32 2)

; CHECK-ELF-LABEL:     .globl g.weird_ref.da.0
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g.weird_ref.da.0:
; CHECK-ELF-NEXT:      .xword (g+16)@AUTH(da,0)

; CHECK-MACHO-LABEL:   .globl _g.weird_ref.da.0
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g.weird_ref.da.0:
; CHECK-MACHO-NEXT:    .quad (_g+16)@AUTH(da,0)

@g.weird_ref.da.0 = constant i64 ptrtoint (ptr inttoptr (i64 ptrtoint (ptr ptrauth (i8* getelementptr (i8, ptr @g, i64 16), i32 2) to i64) to ptr) to i64)

; CHECK-ELF-LABEL:     .globl g_weak.ref.ia.42
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g_weak.ref.ia.42:
; CHECK-ELF-NEXT:      .xword g_weak@AUTH(ia,42)

; CHECK-MACHO-LABEL:   .globl _g_weak.ref.ia.42
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g_weak.ref.ia.42:
; CHECK-MACHO-NEXT:    .quad _g_weak@AUTH(ia,42)

@g_weak.ref.ia.42 = constant ptr ptrauth (ptr @g_weak, i32 0, i64 42)

; CHECK-ELF-LABEL:     .globl g_strong_def.ref.da.0
; CHECK-ELF-NEXT:      .p2align 3
; CHECK-ELF-NEXT:    g_strong_def.ref.da.0:
; CHECK-ELF-NEXT:      .xword g_strong_def@AUTH(da,0)

; CHECK-MACHO-LABEL:   .globl _g_strong_def.ref.da.0
; CHECK-MACHO-NEXT:    .p2align 3
; CHECK-MACHO-NEXT:  _g_strong_def.ref.da.0:
; CHECK-MACHO-NEXT:    .quad _g_strong_def@AUTH(da,0)

@g_strong_def.ref.da.0 = constant ptr ptrauth (ptr @g_strong_def, i32 2)

;--- err-key.ll

; RUN: not --crash llc < err-key.ll -mtriple arm64e-apple-darwin 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-KEY
; RUN: not --crash llc < err-key.ll -mtriple aarch64-elf -mattr=+pauth 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-KEY

; RUN: not --crash llc < err-key.ll -mtriple arm64e-apple-darwin \
; RUN:   -global-isel -verify-machineinstrs -global-isel-abort=1  2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-KEY
; RUN: not --crash llc < err-key.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel -verify-machineinstrs -global-isel-abort=1 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-KEY

; CHECK-ERR-KEY: LLVM ERROR: AArch64 PAC Key ID '4' out of range [0, 3]

@g = external global i32
@g.ref.4.0 = constant ptr ptrauth (ptr @g, i32 4, i64 0)

;--- err-disc.ll

; RUN: not --crash llc < err-disc.ll -mtriple arm64e-apple-darwin 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-DISC
; RUN: not --crash llc < err-disc.ll -mtriple aarch64-elf -mattr=+pauth 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-DISC

; RUN: not --crash llc < err-disc.ll -mtriple arm64e-apple-darwin \
; RUN:   -global-isel -verify-machineinstrs -global-isel-abort=1  2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-DISC
; RUN: not --crash llc < err-disc.ll -mtriple aarch64-elf -mattr=+pauth \
; RUN:   -global-isel -verify-machineinstrs -global-isel-abort=1 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-DISC

; CHECK-ERR-DISC: LLVM ERROR: AArch64 PAC Discriminator '65536' out of range [0, 0xFFFF]

@g = external global i32
@g.ref.ia.65536 = constant ptr ptrauth (ptr @g, i32 0, i64 65536)
