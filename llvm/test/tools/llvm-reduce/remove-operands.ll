; Test that llvm-reduce can reduce operands
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-undef --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s --check-prefixes=CHECK,UNDEF
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-one --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s --check-prefixes=CHECK,ONE
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=operands-zero --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s --check-prefixes=CHECK,ZERO
; RUN: llvm-reduce --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: cat %t | FileCheck %s --check-prefixes=CHECK,ZERO

; CHECK-INTERESTINGNESS: inttoptr
; CHECK-INTERESTINGNESS: inttoptr
; CHECK-INTERESTINGNESS: inttoptr
; CHECK-INTERESTINGNESS: inttoptr
; CHECK-INTERESTINGNESS: br label
; CHECK-INTERESTINGNESS: ret i32

%t = type { i32, i8 }

; CHECK-LABEL: define i32 @main
define i32 @main(%t* %a, i32 %a2) {

; CHECK-LABEL: lb1:
; UNDEF: inttoptr i16 0
; UNDEF: inttoptr i16 1
; UNDEF: inttoptr i16 2
; UNDEF: inttoptr i16 undef
; ONE: inttoptr i16 0
; ONE: inttoptr i16 1
; ONE: inttoptr i16 1
; ONE: inttoptr i16 1
; ZERO: inttoptr i16 0
; ZERO: inttoptr i16 0
; ZERO: inttoptr i16 0
; ZERO: inttoptr i16 0
; CHECK: br label %lb2
lb1:
  %b = getelementptr %t, %t* %a, i32 1, i32 0
  %i1 = inttoptr i16 0 to i8*
  %i2 = inttoptr i16 1 to i8*
  %i3 = inttoptr i16 2 to i8*
  %i4 = inttoptr i16 undef to i8*
  br label %lb2

; CHECK-LABEL: lb2:
; UNDEF: ret i32 undef
; ONE: ret i32 1
; ZERO: ret i32 0
lb2:
  ret i32 %a2
}
