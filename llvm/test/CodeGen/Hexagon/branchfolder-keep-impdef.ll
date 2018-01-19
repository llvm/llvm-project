; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
;
; Check that the testcase compiles successfully. Expect that if-conversion
; took place.
; CHECK-LABEL: fred:
; CHECK: if (!p0) r1 = memw(r0+#0)

target triple = "hexagon"

define void @fred(i32 %p0) local_unnamed_addr align 2 {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  %t0 = load i8*, i8** undef, align 4
  br label %b2

b2:                                               ; preds = %b1, %b0
  %t1 = phi i8* [ %t0, %b1 ], [ undef, %b0 ]
  %t2 = getelementptr inbounds i8, i8* %t1, i32 %p0
  tail call void @llvm.memmove.p0i8.p0i8.i32(i8* undef, i8* %t2, i32 undef, i1 false) #1
  unreachable
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) #0

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind }

