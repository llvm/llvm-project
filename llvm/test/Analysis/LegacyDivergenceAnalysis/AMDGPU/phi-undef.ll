; RUN: opt -mtriple=amdgcn-- -amdgpu-use-legacy-divergence-analysis -analyze -divergence %s | FileCheck %s

; CHECK-LABEL: 'test1':
; CHECK-NEXT: DIVERGENT: i32 %bound
; CHECK: {{^  *}}%counter =
; CHECK-NEXT: DIVERGENT: %break = icmp sge i32 %counter, %bound
; CHECK-NEXT: DIVERGENT: br i1 %break, label %footer, label %body
; CHECK: {{^  *}}%counter.next =
; CHECK: {{^  *}}%counter.footer =
; CHECK: DIVERGENT: br i1 %break, label %end, label %header
; Note: %counter is not divergent!
define amdgpu_ps void @test1(i32 %bound) {
entry:
  br label %header

header:
  %counter = phi i32 [ 0, %entry ], [ %counter.footer, %footer ]
  %break = icmp sge i32 %counter, %bound
  br i1 %break, label %footer, label %body

body:
  %counter.next = add i32 %counter, 1
  br label %footer

footer:
  %counter.footer = phi i32 [ %counter.next, %body ], [ undef, %header ]
  br i1 %break, label %end, label %header

end:
  ret void
}
