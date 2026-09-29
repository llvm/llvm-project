; RUN: llc -global-isel=0 -fast-isel=0 -O0         --relocation-model=pic < %s \
; RUN:   -mattr=+pauth -mattr=+fpac | FileCheck %s --check-prefixes=CHECK,DAGISEL,NOTRAP,DAGISEL-NOTRAP
; RUN: llc -global-isel=0 -fast-isel=0 -O0         --relocation-model=pic < %s \
; RUN:   -mattr=+pauth              | FileCheck %s --check-prefixes=CHECK,DAGISEL,TRAP,DAGISEL-TRAP

; RUN: llc -global-isel=0 -fast-isel=1 -O0         --relocation-model=pic < %s \
; RUN:   -mattr=+pauth -mattr=+fpac | FileCheck %s --check-prefixes=CHECK,DAGISEL,NOTRAP,DAGISEL-NOTRAP
; RUN: llc -global-isel=0 -fast-isel=1 -O0         --relocation-model=pic < %s \
; RUN:   -mattr=+pauth              | FileCheck %s --check-prefixes=CHECK,DAGISEL,TRAP,DAGISEL-TRAP

; RUN: llc -global-isel=1 -global-isel-abort=1 -O0 --relocation-model=pic < %s \
; RUN:   -mattr=+pauth -mattr=+fpac | FileCheck %s --check-prefixes=CHECK,GISEL,NOTRAP,GISEL-NOTRAP
; RUN: llc -global-isel=1 -global-isel-abort=1 -O0 --relocation-model=pic < %s \
; RUN:   -mattr=+pauth              | FileCheck %s --check-prefixes=CHECK,GISEL,TRAP,GISEL-TRAP

;; Note: for FastISel, we fall back to SelectionDAG

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-android"

@global = external global i32
declare void @func()

define ptr @global_addr() #0 {
; CHECK-LABEL: global_addr:
; CHECK:         adrp  x17, :got_auth:global
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:global
; NOTRAP-NEXT:   ldr   x0,  [x17]
; NOTRAP-NEXT:   autda x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_0
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_0:
; TRAP-NEXT:     mov   x0,  x16
; CHECK-NEXT:    ret

  ret ptr @global
}

define i32 @global_load() #0 {
; CHECK-LABEL: global_load:
; CHECK:         adrp  x17, :got_auth:global
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:global
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAP-NEXT:   autda x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_1
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_1:
; TRAP-NEXT:     mov   x8,  x16
; CHECK-NEXT:    ldr   w0,  [x8]
; CHECK-NEXT:    ret

  %load = load i32, ptr @global
  ret i32 %load
}

define void @global_store() #0 {
; CHECK-LABEL:       global_store:
; CHECK:               adrp  x17, :got_auth:global
; CHECK-NEXT:          add   x17, x17, :got_auth_lo12:global
; GISEL-NOTRAP-NEXT:   ldr   x8,  [x17]
; GISEL-NOTRAP-NEXT:   autda x8,  x17
; GISEL-TRAP-NEXT:     ldr   x16, [x17]
; GISEL-TRAP-NEXT:     autda x16, x17
; DAGISEL-NOTRAP-NEXT: ldr   x9,  [x17]
; DAGISEL-NOTRAP-NEXT: autda x9,  x17
; DAGISEL-TRAP-NEXT:   ldr   x16, [x17]
; DAGISEL-TRAP-NEXT:   autda x16, x17
; TRAP-NEXT:           mov   x17, x16
; TRAP-NEXT:           xpacd x17
; TRAP-NEXT:           cmp   x16, x17
; TRAP-NEXT:           b.eq  .Lauth_success_2
; TRAP-NEXT:           brk   #0xc472
; TRAP-NEXT:         .Lauth_success_2:
; GISEL-TRAP-NEXT:     mov   x8,  x16
; DAGISEL-TRAP-NEXT:   mov   x9,  x16
; GISEL-NEXT:          str   wzr, [x8]
; DAGISEL-NEXT:        mov   w8,  wzr
; DAGISEL-NEXT:        str   w8,  [x9]
; CHECK-NEXT:          ret
  store i32 0, ptr @global
  ret void
}

define ptr @func_addr() #0 {
; CHECK-LABEL: func_addr:
; CHECK:         adrp  x17, :got_auth:func
; CHECK-NEXT:    add   x17, x17, :got_auth_lo12:func
; NOTRAP-NEXT:   ldr   x0,  [x17]
; NOTRAP-NEXT:   autia x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autia x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_3
; TRAP-NEXT:     brk   #0xc470
; TRAP-NEXT:   .Lauth_success_3:
; TRAP-NEXT:     mov   x0,  x16
; CHECK-NEXT:    ret

  ret ptr @func
}

attributes #0 = { "target-features"="+tagged-globals" }

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
