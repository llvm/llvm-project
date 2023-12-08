; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-linux-gnu | FileCheck %s

declare void @bar(i32)

define void @test_float(float %a, float %b) {
; CHECK-LABEL: test_float:

  %tst1 = fcmp oeq float %a, %b
  br i1 %tst1, label %end, label %t2
; CHECK: fcmp {{s[0-9]+}}, {{s[0-9]+}}
; CHECK: b.eq .L

t2:
  %tst2 = fcmp une float %b, 0.0
  br i1 %tst2, label %t3, label %end
; CHECK: fcmp {{s[0-9]+}}, #0.0
; CHECK: b.eq .L


t3:
; This test can't be implemented with just one A64 conditional
; branch. LLVM converts "ordered and not equal" to "unordered or
; equal" before instruction selection, which is what we currently
; test. Obviously, other sequences are valid.
  %tst3 = fcmp one float %a,  %b
  br i1 %tst3, label %t4, label %end
; CHECK: fcmp {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NEXT: b.eq .[[T4:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: b.vs .[[T4]]
t4:
  %tst4 = fcmp uge float %a, -0.0
  br i1 %tst4, label %t5, label %end
; CHECK-NOT: fcmp {{s[0-9]+}}, #0.0
; CHECK: b.mi .LBB

t5:
  call void @bar(i32 0)
  ret void
end:
  ret void

}

define void @test_double(double %a, double %b) {
; CHECK-LABEL: test_double:

  %tst1 = fcmp oeq double %a, %b
  br i1 %tst1, label %end, label %t2
; CHECK: fcmp {{d[0-9]+}}, {{d[0-9]+}}
; CHECK: b.eq .L

t2:
  %tst2 = fcmp une double %b, 0.0
  br i1 %tst2, label %t3, label %end
; CHECK: fcmp {{d[0-9]+}}, #0.0
; CHECK: b.eq .L


t3:
; This test can't be implemented with just one A64 conditional
; branch. LLVM converts "ordered and not equal" to "unordered or
; equal" before instruction selection, which is what we currently
; test. Obviously, other sequences are valid.
  %tst3 = fcmp one double %a,  %b
  br i1 %tst3, label %t4, label %end
; CHECK: fcmp {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NEXT: b.eq .[[T4:LBB[0-9]+_[0-9]+]]
; CHECK-NEXT: b.vs .[[T4]]
t4:
  %tst4 = fcmp uge double %a, -0.0
  br i1 %tst4, label %t5, label %end
; CHECK-NOT: fcmp {{d[0-9]+}}, #0.0
; CHECK: b.mi .LBB

t5:
  call void @bar(i32 0)
  ret void
end:
  ret void

}
