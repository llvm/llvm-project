; REQUIRES: asserts
; RUN: opt -mtriple=nvptx64-nvidia-cuda -mcpu=sm_100 -passes=slsr -stats \
; RUN:   -disable-output < %s 2>&1 | FileCheck %s

; CHECK: 2 slsr - Number of SCEV candidate differences computed by SLSR

declare i64 @source(i32)
declare void @use(ptr)

define void @base_delta(ptr %root, i64 %common) {
  %offset.0 = call i64 @source(i32 0)
  %base.0 = getelementptr i8, ptr %root, i64 %offset.0
  %candidate.0 = getelementptr i8, ptr %base.0, i64 %common
  call void @use(ptr %candidate.0)

  %delta.1 = call i64 @source(i32 1)
  %offset.1 = add i64 %offset.0, %delta.1
  %base.1 = getelementptr i8, ptr %root, i64 %offset.1
  %candidate.1 = getelementptr i8, ptr %base.1, i64 %common
  call void @use(ptr %candidate.1)
  ret void
}
