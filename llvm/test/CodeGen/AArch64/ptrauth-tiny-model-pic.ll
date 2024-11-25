; RUN: llc -verify-machineinstrs -mtriple=aarch64 -mattr=+pauth -mattr=+fpac -code-model=tiny \
; RUN:   -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,NOTRAP %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64 -mattr=+pauth              -code-model=tiny \
; RUN:   -relocation-model=pic < %s | FileCheck --check-prefixes=CHECK,TRAP %s

; RUN: llc -verify-machineinstrs -mtriple=aarch64 -mattr=+pauth -mattr=+fpac -code-model=tiny \
; RUN:   -relocation-model=pic -fast-isel < %s | FileCheck --check-prefixes=CHECK,NOTRAP %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64 -mattr=+pauth              -code-model=tiny \
; RUN:   -relocation-model=pic -fast-isel < %s | FileCheck --check-prefixes=CHECK,TRAP %s

; RUN: llc -verify-machineinstrs -mtriple=aarch64 -mattr=+pauth -mattr=+fpac -code-model=tiny \
; RUN:   -relocation-model=pic -global-isel -global-isel-abort=1 < %s | FileCheck --check-prefixes=CHECK,NOTRAP %s
; RUN: llc -verify-machineinstrs -mtriple=aarch64 -mattr=+pauth              -code-model=tiny \
; RUN:   -relocation-model=pic -global-isel -global-isel-abort=1 < %s | FileCheck --check-prefixes=CHECK,TRAP %s

; Note: fast-isel tests here will fall back to isel

@src = external local_unnamed_addr global [65536 x i8], align 1
@dst = external global [65536 x i8], align 1
@ptr = external local_unnamed_addr global ptr, align 8

define dso_preemptable void @foo1() {
; CHECK-LABEL: foo1:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr   x17, :got_auth:src
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAP-NEXT:   autda x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_0
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_0:
; TRAP-NEXT:     mov   x8,  x16
; CHECK-NEXT:    ldrb  w8,  [x8]
; CHECK-NEXT:    adr   x17, :got_auth:dst
; NOTRAP-NEXT:   ldr   x9,  [x17]
; NOTRAP-NEXT:   autda x9,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_1
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_1:
; TRAP-NEXT:     mov   x9,  x16
; CHECK-NEXT:    strb  w8,  [x9]
; CHECK-NEXT:    ret

entry:
  %0 = load i8, ptr @src, align 1
  store i8 %0, ptr @dst, align 1
  ret void
}

define dso_preemptable void @foo2() {
; CHECK-LABEL: foo2:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr   x17, :got_auth:ptr
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAP-NEXT:   autda x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_2
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_2:
; TRAP-NEXT:     mov   x8,  x16
; CHECK-NEXT:    adr   x17, :got_auth:dst
; NOTRAP-NEXT:   ldr   x9,  [x17]
; NOTRAP-NEXT:   autda x9,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_3
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_3:
; TRAP-NEXT:     mov   x9,  x16
; CHECK-NEXT:    str   x9,  [x8]
; CHECK-NEXT:    ret

entry:
  store ptr @dst, ptr @ptr, align 8
  ret void
}

define dso_preemptable void @foo3() {
; CHECK-LABEL: foo3:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr   x17, :got_auth:src
; NOTRAP-NEXT:   ldr   x8,  [x17]
; NOTRAP-NEXT:   autda x8,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_4
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_4:
; TRAP-NEXT:     mov   x8,  x16
; CHECK-NEXT:    ldrb  w8,  [x8]
; CHECK-NEXT:    adr   x17, :got_auth:ptr
; NOTRAP-NEXT:   ldr   x9,  [x17]
; NOTRAP-NEXT:   autda x9,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autda x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpacd x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_5
; TRAP-NEXT:     brk   #0xc472
; TRAP-NEXT:   .Lauth_success_5:
; TRAP-NEXT:     mov   x9,  x16
; CHECK-NEXT:    ldr   x9,  [x9]
; CHECK-NEXT:    strb  w8,  [x9]
; CHECK-NEXT:    ret

entry:
  %0 = load i8, ptr @src, align 1
  %1 = load ptr, ptr @ptr, align 8
  store i8 %0, ptr %1, align 1
  ret void
}

@lsrc = internal global i8 0, align 4
@ldst = internal global i8 0, align 4
@lptr = internal global ptr null, align 8

declare void @func(...)

define dso_preemptable ptr @externfuncaddr() {
; CHECK-LABEL: externfuncaddr:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr   x17, :got_auth:func
; NOTRAP-NEXT:   ldr   x0,  [x17]
; NOTRAP-NEXT:   autia x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autia x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_6
; TRAP-NEXT:     brk   #0xc470
; TRAP-NEXT:   .Lauth_success_6:
; TRAP-NEXT:     mov   x0,  x16
; CHECK-NEXT:    ret

entry:
  ret ptr @func
}

define dso_preemptable ptr @localfuncaddr() {
; CHECK-LABEL: localfuncaddr:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    adr   x17, :got_auth:externfuncaddr
; NOTRAP-NEXT:   ldr   x0,  [x17]
; NOTRAP-NEXT:   autia x0,  x17
; TRAP-NEXT:     ldr   x16, [x17]
; TRAP-NEXT:     autia x16, x17
; TRAP-NEXT:     mov   x17, x16
; TRAP-NEXT:     xpaci x17
; TRAP-NEXT:     cmp   x16, x17
; TRAP-NEXT:     b.eq  .Lauth_success_7
; TRAP-NEXT:     brk   #0xc470
; TRAP-NEXT:   .Lauth_success_7:
; TRAP-NEXT:     mov   x0,  x16
; CHECK-NEXT:    ret

entry:
  ret ptr @externfuncaddr
}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"ptrauth-elf-got", i32 1}
