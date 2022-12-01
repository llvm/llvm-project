; RUN: opt -passes=indvars -S < %s | FileCheck %s

define void @infer_via_ranges(i32 *%arr, i32 %n) {
; CHECK-LABEL: @infer_via_ranges
 entry:
  %first.itr.check = icmp sgt i32 %n, 0
  %start = sub i32 %n, 1
  br i1 %first.itr.check, label %loop, label %exit

 loop:
; CHECK-LABEL: loop:
  %idx = phi i32 [ %start, %entry ] , [ %idx.dec, %in.bounds ]
  %idx.dec = sub i32 %idx, 1
  %abc = icmp sge i32 %idx, 0
; CHECK: br i1 true, label %in.bounds, label %out.of.bounds
  br i1 %abc, label %in.bounds, label %out.of.bounds

 in.bounds:
; CHECK-LABEL: in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %idx
  store i32 0, i32* %addr
  %next = icmp sgt i32 %idx.dec, -1
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}
