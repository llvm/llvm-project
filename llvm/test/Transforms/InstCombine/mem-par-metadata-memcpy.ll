; RUN: opt < %s -instcombine -S | FileCheck %s
;
; Make sure the llvm.access.group meta-data is preserved
; when a memcpy is replaced with a load+store by instcombine
;
; #include <string.h>
; void test(char* out, long size)
; {
;     #pragma clang loop vectorize(assume_safety)
;     for (long i = 0; i < size; i+=2) {
;         memcpy(&(out[i]), &(out[i+size]), 2);
;     }
; }

; CHECK: for.body:
; CHECK:  %{{.*}} = load i16, i16* %{{.*}}, align 1, !llvm.access.group !1
; CHECK:  store i16 %{{.*}}, i16* %{{.*}}, align 1, !llvm.access.group !1


; ModuleID = '<stdin>'
source_filename = "memcpy.pragma.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @_Z4testPcl(i8* %out, i64 %size) #0 {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %add2, %for.inc ]
  %cmp = icmp slt i64 %i.0, %size
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i8, i8* %out, i64 %i.0
  %add = add nsw i64 %i.0, %size
  %arrayidx1 = getelementptr inbounds i8, i8* %out, i64 %add
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %arrayidx, i8* %arrayidx1, i64 2, i1 false), !llvm.access.group !4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %add2 = add nsw i64 %i.0, 2
  br label %for.cond, !llvm.loop !2

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (cfe/trunk 277751)"}
!1 = distinct !{!1, !2, !3, !{!"llvm.loop.parallel_accesses", !4}}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
!4 = distinct !{} ; access group
