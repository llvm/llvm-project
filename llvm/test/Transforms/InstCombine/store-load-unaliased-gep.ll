; RUN: opt -passes=instcombine %s -S 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes=instcombine %s -S 2>&1 | FileCheck %s

; Checking successful store-load optimization of array length.
; Function below should deduce just to "return length".
; Doable only if instcombine has access to alias-analysis.

define i32 @test1(i32 %length) {
; CHECK-LABEL: entry:
entry:
  %array = alloca i32, i32 2
  ; CHECK-NOT: %array

  %value_gep = getelementptr inbounds i32, ptr %array, i32 1
  store i32 %length, ptr %array
  store i32 0, ptr %value_gep
  %loaded_length = load i32, ptr %array
  ; CHECK-NOT: %loaded_length = load i32

  ret i32 %loaded_length
  ; CHECK: ret i32 %length
}
