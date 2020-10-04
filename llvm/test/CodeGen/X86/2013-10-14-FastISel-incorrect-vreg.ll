; RUN: llc -mtriple x86_64-apple-darwin -O0 < %s -o - | FileCheck %s
;
; During X86 fastisel, the address of indirect call was resolved
; through bitcast, ptrtoint, and inttoptr instructions. This is valid
; only if the related instructions are in that same basic block, otherwise
; we may reference variables that were not live across basic blocks
; resulting in undefined virtual registers.
;
; In this example, this is illustrated by a spill/reload of the
; LOADED_PTR_SLOT.
;
; Before this patch, the compiler was accessing two different spill
; slots.
; <rdar://problem/15192473>

; CHECK-LABEL: @test_bitcast
; Load the value of the function pointer: %loaded_ptr
; Spill %arg2.
; CHECK: movq %rdx, [[ARG2_SLOT:[0-9]*\(%[a-z]+\)]]
; CHECK: movq (%rdi), [[LOADED_PTR:%[a-z]+]]
; Spill %loaded_ptr.
; CHECK: movq [[LOADED_PTR]], [[LOADED_PTR_SLOT:[0-9]*\(%[a-z]+\)]]
; Perform the indirect call.
; Load the function pointer.
; CHECK: movq [[LOADED_PTR_SLOT]], [[FCT_PTR:%[a-z]+]]
; Load the third argument
; CHECK: movq [[ARG2_SLOT]], %rdx
; Load the first argument
; CHECK: movq %rdx, %rdi
; Load the second argument
; CHECK: movq %rdx, %rsi
; Call.
; CHECK: callq *[[FCT_PTR]]
; CHECK: ret
define i64 @test_bitcast(i64 (i64, i64, i64)** %arg, i1 %bool, i64 %arg2) {
entry:
  %loaded_ptr = load i64 (i64, i64, i64)*, i64 (i64, i64, i64)** %arg, align 8
  %raw = bitcast i64 (i64, i64, i64)* %loaded_ptr to i8*
  switch i1 %bool, label %default [
    i1 true, label %label_true
    i1 false, label %label_end
  ]
default:
  br label %label_end

label_true:
  br label %label_end

label_end:
  %fct_ptr = bitcast i8* %raw to i64 (i64, i64, i64)*
  %res = call i64 %fct_ptr(i64 %arg2, i64 %arg2, i64 %arg2)
  ret i64 %res
}

; CHECK-LABEL: @test_inttoptr
; Load the value of the function pointer: %loaded_ptr
; CHECK: movq %rdx, [[ARG2_SLOT:[0-9]*\(%[a-z]+\)]]
; Spill %loaded_ptr.
; CHECK: movq (%rdi), [[LOADED_PTR:%[a-z]+]]
; Spill %arg2.
; CHECK: movq [[LOADED_PTR]], [[LOADED_PTR_SLOT:[0-9]*\(%[a-z]+\)]]
; Perform the indirect call.
; Load the function pointer.
; CHECK: movq [[LOADED_PTR_SLOT]], [[FCT_PTR:%[a-z]+]]
; Load the third argument
; CHECK: movq [[ARG2_SLOT]], %rdx
; Load the first argument
; CHECK: movq %rdx, %rdi
; Load the second argument
; CHECK: movq %rdx, %rsi
; Call.
; CHECK: callq *[[FCT_PTR]]
; CHECK: ret
define i64 @test_inttoptr(i64 (i64, i64, i64)** %arg, i1 %bool, i64 %arg2) {
entry:
  %loaded_ptr = load i64 (i64, i64, i64)*, i64 (i64, i64, i64)** %arg, align 8
  %raw = ptrtoint i64 (i64, i64, i64)* %loaded_ptr to i64
  switch i1 %bool, label %default [
    i1 true, label %label_true
    i1 false, label %label_end
  ]
default:
  br label %label_end

label_true:
  br label %label_end

label_end:
  %fct_ptr = inttoptr i64 %raw to i64 (i64, i64, i64)*
  %res = call i64 %fct_ptr(i64 %arg2, i64 %arg2, i64 %arg2)
  ret i64 %res
}

; CHECK-LABEL: @test_ptrtoint
; Spill %arg2.
; CHECK: movq %rdx, [[ARG2_SLOT:[0-9]*\(%[a-z]+\)]]
; Load the value of the function pointer: %loaded_ptr
; CHECK: movq (%rdi), [[LOADED_PTR:%[a-z]+]]
; Spill %loaded_ptr.
; CHECK: movq [[LOADED_PTR]], [[LOADED_PTR_SLOT:[0-9]*\(%[a-z]+\)]]
; Perform the indirect call.
; Load the function pointer.
; CHECK: movq [[LOADED_PTR_SLOT]], [[FCT_PTR:%[a-z]+]]
; Load the third argument
; CHECK: movq [[ARG2_SLOT]], %rdx
; Load the first argument
; CHECK: movq %rdx, %rdi
; Load the second argument
; CHECK: movq %rdx, %rsi
; Call.
; CHECK: callq *[[FCT_PTR]]
; CHECK: ret
define i64 @test_ptrtoint(i64 (i64, i64, i64)** %arg, i1 %bool, i64 %arg2) {
entry:
  %loaded_ptr = load i64 (i64, i64, i64)*, i64 (i64, i64, i64)** %arg, align 8
  %raw = bitcast i64 (i64, i64, i64)* %loaded_ptr to i8*
  switch i1 %bool, label %default [
    i1 true, label %label_true
    i1 false, label %label_end
  ]
default:
  br label %label_end

label_true:
  br label %label_end

label_end:
  %fct_int = ptrtoint i8* %raw to i64
  %fct_ptr = inttoptr i64 %fct_int to i64 (i64, i64, i64)*
  %res = call i64 %fct_ptr(i64 %arg2, i64 %arg2, i64 %arg2)
  ret i64 %res
}
