; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -O0 -debug-only=selectiondag -o - < %s 2>&1 | \
; RUN:  FileCheck %s
; CHECK-NOT: lxvdsx
; CHECK-NOT: LD_SPLAT
; REQUIRES: asserts

define weak_odr dso_local void @unpack(ptr noalias noundef %packed_in) local_unnamed_addr {
entry:
  %ld = load <2 x i32>, ptr %packed_in, align 2
  %shuf = shufflevector <2 x i32> %ld, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 0>
  %ie = insertelement <4 x i32> %shuf, i32 7, i32 2
  store <4 x i32> %shuf, ptr %packed_in, align 2
  ret void
}
