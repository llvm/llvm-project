; REQUIRES: asserts
; RUN: opt -passes=slsr -stats -disable-output < %s 2>&1 | FileCheck %s

; CHECK: 8 slsr - Number of candidate-basis SCEV differences computed by SLSR

target triple = "nvptx64-nvidia-cuda"

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

  %delta.2 = call i64 @source(i32 2)
  %offset.2 = add i64 %offset.1, %delta.2
  %base.2 = getelementptr i8, ptr %root, i64 %offset.2
  %candidate.2 = getelementptr i8, ptr %base.2, i64 %common
  call void @use(ptr %candidate.2)
  ret void
}
