; RUN: split-file %s %t
; RUN: not llvm-as < %s %t/instr_smaller_id.ll 2>&1 | FileCheck %s --check-prefix=INSTR-SMALLER-ID
; RUN: not llvm-as < %s %t/instr_too_large_skip.ll 2>&1 | FileCheck %s --check-prefix=INSTR-TOO-LARGE-SKIP
; RUN: not llvm-as < %s %t/arg_smaller_id.ll 2>&1 | FileCheck %s --check-prefix=ARG-SMALLER-ID
; RUN: not llvm-as < %s %t/arg_too_large_skip.ll 2>&1 | FileCheck %s --check-prefix=ARG-TOO-LARGE-SKIP
; RUN: not llvm-as < %s %t/block_smaller_id.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-SMALLER-ID
; RUN: not llvm-as < %s %t/block_too_large_skip.ll 2>&1 | FileCheck %s --check-prefix=BLOCK-TOO-LARGE-SKIP

;--- instr_smaller_id.ll

; INSTR-SMALLER-ID: error: instruction expected to be numbered '%11' or greater
define i32 @test() {
  %10 = add i32 1, 2
  %5 = add i32 %10, 3
  ret i32 %5
}

;--- instr_too_large_skip.ll

; INSTR-TOO-LARGE-SKIP: error: value numbers can skip at most 1000 values
define i32 @test() {
  %10 = add i32 1, 2
  %1000000 = add i32 %10, 3
  ret i32 %1000000
}

;--- arg_smaller_id.ll

; ARG-SMALLER-ID: error: argument expected to be numbered '%11' or greater
define i32 @test(i32 %10, i32 %5) {
  ret i32 %5
}

;--- arg_too_large_skip.ll

; ARG-TOO-LARGE-SKIP: error: value numbers can skip at most 1000 values
define i32 @test(i32 %10, i32 %1000000) {
  ret i32 %1000000
}

;--- block_smaller_id.ll

; BLOCK-SMALLER-ID: error: label expected to be numbered '11' or greater
define i32 @test() {
10:
  br label %5

5:
  ret i32 0
}

;--- block_too_large_skip.ll

; BLOCK-TOO-LARGE-SKIP: error: value numbers can skip at most 1000 values
define i32 @test() {
10:
  br label %1000000

1000000:
  ret i32 0
}
