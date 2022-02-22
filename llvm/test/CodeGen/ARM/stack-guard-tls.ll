; RUN: split-file %s %t
; RUN: cat %t/main.ll %t/a.ll > %t/a2.ll
; RUN: cat %t/main.ll %t/b.ll > %t/b2.ll
; RUN: llc %t/a2.ll -mtriple=armv7-unknown-linux-gnueabihf -mattr=+read-tp-hard -o - | \
; RUN: FileCheck --check-prefixes=CHECK,CHECK-SMALL %s
; RUN: llc %t/a2.ll -mtriple=thumbv7-unknown-linux-gnueabihf -mattr=+read-tp-hard -o - | \
; RUN: FileCheck --check-prefixes=CHECK,CHECK-SMALL %s
; RUN: llc %t/b2.ll -mtriple=armv7-unknown-linux-gnueabihf -mattr=+read-tp-hard -o - | \
; RUN: FileCheck --check-prefixes=CHECK,CHECK-LARGE %s
; RUN: llc %t/b2.ll -mtriple=thumbv7-unknown-linux-gnueabihf -mattr=+read-tp-hard -o - | \
; RUN: FileCheck --check-prefixes=CHECK,CHECK-LARGE %s

;--- main.ll
declare void @baz(i32*)

define void @foo(i64 %t) sspstrong {
  %vla = alloca i32, i64 %t, align 4
  call void @baz(i32* nonnull %vla)
  ret void
}
!llvm.module.flags = !{!1, !2}
!1 = !{i32 2, !"stack-protector-guard", !"tls"}

;--- a.ll
!2 = !{i32 2, !"stack-protector-guard-offset", i32 1296}

;--- b.ll
!2 = !{i32 2, !"stack-protector-guard-offset", i32 4296}

; CHECK: mrc p15, #0, [[REG1:r[0-9]+]], c13, c0, #3
; CHECK-SMALL-NEXT: ldr{{(\.w)?}} [[REG1]], [[[REG1]], #1296]
; CHECK-LARGE-NEXT: add{{(\.w)?}} [[REG1]], [[REG1]], #4096
; CHECK-LARGE-NEXT: ldr{{(\.w)?}} [[REG1]], [[[REG1]], #200]
; CHECK: bl baz
; CHECK: mrc p15, #0, [[REG2:r[0-9]+]], c13, c0, #3
; CHECK-SMALL-NEXT: ldr{{(\.w)?}} [[REG2]], [[[REG2]], #1296]
; CHECK-LARGE-NEXT: add{{(\.w)?}} [[REG2]], [[REG2]], #4096
; CHECK-LARGE-NEXT: ldr{{(\.w)?}} [[REG2]], [[[REG2]], #200]
