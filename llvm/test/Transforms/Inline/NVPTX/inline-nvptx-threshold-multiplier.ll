; RUN: opt -S -mtriple=nvptx64-nvidia-cuda -passes=inline -inline-threshold=5 < %s | FileCheck %s

; This test verifies that the multiplier is at least 29 by checking that
; the callee is inlined.

define i32 @callee(i32 %x) {
entry:
  %v0 = or i32 %x, 1
  %v1 = or i32 %x, 1
  %v2 = or i32 %x, 1
  %v3 = or i32 %x, 1
  %v4 = or i32 %x, 1
  %v5 = or i32 %x, 1
  %v6 = or i32 %x, 1
  %v7 = or i32 %x, 1
  %v8 = or i32 %x, 1
  %v9 = or i32 %x, 1
  %v10 = or i32 %x, 1
  %v11 = or i32 %x, 1
  %v12 = or i32 %x, 1
  %v13 = or i32 %x, 1
  %v14 = or i32 %x, 1
  %v15 = or i32 %x, 1
  %v16 = or i32 %x, 1
  %v17 = or i32 %x, 1
  %v18 = or i32 %x, 1
  %v19 = or i32 %x, 1
  %v20 = or i32 %x, 1
  %v21 = or i32 %x, 1
  %v22 = or i32 %x, 1
  %v23 = or i32 %x, 1
  %v24 = or i32 %x, 1
  %v25 = or i32 %x, 1
  %v26 = or i32 %x, 1
  %v27 = or i32 %x, 1
  %v28 = or i32 %x, 1
  %v29 = or i32 %x, 1
  %v30 = or i32 %x, 1
  %v31 = or i32 %x, 1
  %v32 = or i32 %x, 1
  %v33 = or i32 %x, 1
  %v34 = or i32 %x, 1
  %v35 = or i32 %x, 1
  %v36 = or i32 %x, 1
  %v37 = or i32 %x, 1
  %v38 = or i32 %x, 1
  %v39 = or i32 %x, 1
  %v40 = or i32 %x, 1
  %v41 = or i32 %x, 1
  %v42 = or i32 %x, 1
  %v43 = or i32 %x, 1
  %v44 = or i32 %x, 1
  %v45 = or i32 %x, 1
  %v46 = or i32 %x, 1
  %v47 = or i32 %x, 1
  %v48 = or i32 %x, 1
  ret i32 %v48
}

; CHECK-LABEL: define i32 @caller
; CHECK-NOT: call i32 @callee
; CHECK: ret i32
define i32 @caller(i32 %y) {
  %result = call i32 @callee(i32 %y)
  ret i32 %result
}
