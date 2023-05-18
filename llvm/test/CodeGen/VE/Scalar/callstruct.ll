; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

%struct.a = type { i32, i32 }

@A = common global %struct.a zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @fun(ptr noalias nocapture sret(%struct.a) %a, i32 %p1, i32 %p2) {
; CHECK-LABEL: fun:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    stl %s2, 4(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store i32 %p1, ptr %a, align 4
  %a.one = getelementptr inbounds %struct.a, ptr %a, i64 0, i32 1
  store i32 %p2, ptr %a.one, align 4
  ret void
}

; Function Attrs: nounwind
define void @caller() {
; CHECK-LABEL: caller:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s0, callee@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, callee@hi(, %s0)
; CHECK-NEXT:    lea %s0, 248(, %s11)
; CHECK-NEXT:    or %s1, 3, (0)1
; CHECK-NEXT:    or %s2, 4, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s0, 248(, %s11)
; CHECK-NEXT:    lea %s1, A@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, A@hi(, %s1)
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    or %s11, 0, %s9
  %a = alloca i64, align 8
  call void @callee(ptr nonnull sret(%struct.a) %a, i32 3, i32 4)
  %a.val = load i64, ptr %a, align 8
  store i64 %a.val, ptr @A, align 4
  ret void
}

declare void @callee(ptr sret(%struct.a), i32, i32)
