; RUN: llc < %s -mtriple=x86_64-unknown-linux | FileCheck --check-prefix=NO-ADDRSIG %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux -addrsig | FileCheck %s

; NO-ADDRSIG-NOT: .addrsig

; CHECK: .addrsig

; CHECK: .addrsig_sym f1
define void @f1() {
  %f1 = bitcast void()* @f1 to i8*
  %f2 = bitcast void()* @f2 to i8*
  %f3 = bitcast void()* @f3 to i8*
  %g1 = bitcast i32* @g1 to i8*
  %g2 = bitcast i32* @g2 to i8*
  %g3 = bitcast i32* @g3 to i8*
  %dllimport = bitcast i32* @dllimport to i8*
  %tls = bitcast i32* @tls to i8*
  %a1 = bitcast i32* @a1 to i8*
  %a2 = bitcast i32* @a2 to i8*
  %i1 = bitcast void()* @i1 to i8*
  %i2 = bitcast void()* @i2 to i8*
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

; CHECK-NOT: .addrsig_sym unref
@unref = external global i32

; CHECK-NOT: .addrsig_sym dllimport
@dllimport = external dllimport global i32

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
