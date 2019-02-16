; RUN: opt < %s  -instcombine -S | FileCheck %s

define void @matching_phi(i64 %a, float* %b, i1 %cond) {
; CHECK-LABEL: @matching_phi
entry:
  %cmp1 = icmp  eq i1 %cond, 0
  %add.int = add i64 %a, 1
  %add = inttoptr i64 %add.int to float *

  %addb = getelementptr inbounds float, float* %b, i64 2
  %addb.int = ptrtoint float* %addb to i64
  br i1 %cmp1, label %A, label %B
A:
  br label %C
B:
  store float 1.0e+01, float* %add, align 4
  br label %C

C:
  %a.addr.03 = phi float* [ %addb, %A ], [ %add, %B ]
  %b.addr.02 = phi i64 [ %addb.int, %A ], [ %add.int, %B ]
  %tmp = inttoptr i64 %b.addr.02 to float*
; CHECK: %a.addr.03 = phi
; CHECK-NEXT: = load
  %tmp1 = load float, float* %tmp, align 4
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  ret void
}

define void @no_matching_phi(i64 %a, float* %b, i1 %cond) {
; CHECK-LABEL: @no_matching_phi
entry:
  %cmp1 = icmp  eq i1 %cond, 0
  %add.int = add i64 %a, 1
  %add = inttoptr i64 %add.int to float *

  %addb = getelementptr inbounds float, float* %b, i64 2
  %addb.int = ptrtoint float* %addb to i64
  br i1 %cmp1, label %A, label %B
A:
  br label %C
B:
  store float 1.0e+01, float* %add, align 4
  br label %C

C:
  %a.addr.03 = phi float* [ %addb, %A ], [ %add, %B ]
  %b.addr.02 = phi i64 [ %addb.int, %B ], [ %add.int, %A ]
  %tmp = inttoptr i64 %b.addr.02 to float*
  %tmp1 = load float, float* %tmp, align 4
; CHECK: %a.addr.03 = phi
; CHECK-NEXT: %b.addr.02.ptr = phi
; CHECK-NEXT: = load
  %mul.i = fmul float %tmp1, 4.200000e+01
  store float %mul.i, float* %a.addr.03, align 4
  ret void
}
