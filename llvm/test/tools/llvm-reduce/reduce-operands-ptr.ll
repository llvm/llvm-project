; Test that llvm-reduce can reduce pointer operands
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ONE %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; RUN: llvm-reduce --abort-on-invalid-reduction --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,ZERO %s < %t

; CHECK-LABEL: define void @foo(

; ONE: load i32, ptr %a0
; ONE: load i32, ptr @g
; ONE: extractelement <4 x ptr> <ptr @g, ptr null, ptr @g, ptr @g>, i32 11

; ZERO: load i32, ptr null
; ZERO: load i32, ptr null
; ZERO: extractelement <4 x ptr> zeroinitializer, i32 11

@g = global i32 0

define void @foo(ptr %a0) {
  ; CHECK-INTERESTINGNESS: load i32
  %v0 = load i32, ptr %a0
  ; CHECK-INTERESTINGNESS: load i32
  %v1 = load i32, ptr @g

  ; CHECK-INTERESTINGNESS: extractelement{{.*}}i32 11
  %v2 = extractelement <4 x ptr> <ptr @g, ptr null, ptr @g, ptr @g>, i32 11

  ret void
}
