; RUN: opt < %s -passes="slsr" -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"

%struct.B = type { i16 }
%struct.A = type { %struct.B, %struct.B, %struct.B }

define void @path_compression(i32 %a, ptr %base, i16 %r, i1 %cond) {
; CHECK-LABEL: @path_compression(
; CHECK: [[I:%.*]] = sext i32 %a to i64
; CHECK: [[GEP1:%.*]] = getelementptr inbounds %struct.A, ptr %base, i64 [[I]]
; CHECK: br
; CHECK-LABEL: next
; compress the path to use GEP1 as the Basis instead of GEP2
; CHECK: [[GEP2:%.*]] = getelementptr inbounds i8, ptr [[GEP1]], i64 2
; CHECK: [[GEP3:%.*]] = getelementptr inbounds i8, ptr [[GEP1]], i64 4


  %1 = sext i32 %a to i64
  %2 = add i64 %1, 1
  %getElem1 = getelementptr inbounds %struct.A, ptr %base, i64 %1
  br i1 %cond, label %next, label %ret

next:
  %getElem2 = getelementptr inbounds %struct.A, ptr %base, i64 %1, i32 1
  %offset = sub i64 %2, 1
  %getElem3 = getelementptr inbounds %struct.A, ptr %base, i64 %offset, i32 2
  store i16 %r, ptr %getElem1, align 2
  store i16 %r, ptr %getElem2, align 2
  store i16 %r, ptr %getElem3, align 2
  br label %ret

ret:
  ret void
}
