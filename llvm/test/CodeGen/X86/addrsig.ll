; RUN: llc < %s -mtriple=x86_64-unknown-linux | FileCheck --check-prefix=NO-ADDRSIG %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux -addrsig | FileCheck %s

; NO-ADDRSIG-NOT: .addrsig

; CHECK: .addrsig

; CHECK: .addrsig_sym f1
define void @f1() {
  unreachable
}

; CHECK-NOT: .addrsig_sym f2
define internal void @f2() local_unnamed_addr {
  unreachable
}

; CHECK-NOT: .addrsig_sym f3
declare void @f3() unnamed_addr

; CHECK: .addrsig_sym g1
@g1 = global i32 0
; CHECK-NOT: .addrsig_sym g2
@g2 = internal local_unnamed_addr global i32 0
; CHECK-NOT: .addrsig_sym g3
@g3 = external unnamed_addr global i32

; CHECK-NOT: .addrsig_sym tls
@tls = thread_local global i32 0

; CHECK: .addrsig_sym a1
@a1 = alias i32, i32* @g1
; CHECK-NOT: .addrsig_sym a2
@a2 = internal local_unnamed_addr alias i32, i32* @g2

; CHECK: .addrsig_sym i1
@i1 = ifunc void(), void()* @f1
; CHECK-NOT: .addrsig_sym i2
@i2 = internal local_unnamed_addr ifunc void(), void()* @f2
