; RUN: not llc -mtriple=x86_64-pc-linux -stackrealign -verify-machineinstrs %s -o - 2>&1 | FileCheck %s

declare cc 11 i64 @hipe2(i64, i64, i64, i64, i64, i64, i64)

; Test with many arguments, so some of them are passed from stack. The spilling
; of rbp should not disturb stack arguments.
; fixme: current generated code is wrong because rbp is used to load passed in
;        argument after rbp is assigned argument for function call, it is caused
;        by x86-cf-opt.

; CHECK: <unknown>:0: error: Interference usage of base pointer/frame pointer.
; CHECK: <unknown>:0: error: Interference usage of base pointer/frame pointer.
define i64 @test3(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7) {
  %x = call cc 11 i64 @hipe2(i64 %a0, i64 %a1, i64 %a2, i64 %a3, i64 %a4, i64 %a5, i64 %a6, i64 %a7)
  ret i64 %x
}
