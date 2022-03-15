; This testcase ensures that CFLSteensAA doesn't try to access out of bounds indices
; when given functions with large amounts of arguments (specifically, more
; arguments than the StratifiedAttrs bitset can handle)
;
; Because the result on failure is effectively crashing the compiler, output
; checking is minimal.

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-may-aliases -disable-output 2>&1 | FileCheck %s

; CHECK: Function: test
define void @test(i1 %cond,
                  i32* %arg1, i32* %arg2, i32* %arg3, i32* %arg4, i32* %arg5,
                  i32* %arg6, i32* %arg7, i32* %arg8, i32* %arg9, i32* %arg10,
                  i32* %arg11, i32* %arg12, i32* %arg13, i32* %arg14, i32* %arg15,
                  i32* %arg16, i32* %arg17, i32* %arg18, i32* %arg19, i32* %arg20,
                  i32* %arg21, i32* %arg22, i32* %arg23, i32* %arg24, i32* %arg25,
                  i32* %arg26, i32* %arg27, i32* %arg28, i32* %arg29, i32* %arg30,
                  i32* %arg31, i32* %arg32, i32* %arg33, i32* %arg34, i32* %arg35) {

  ; CHECK: 45 Total Alias Queries Performed
  ; CHECK: 9 no alias responses (20.0%)
  %a = alloca i32, align 4
  load i32, i32* %a
  load i32, i32* %arg27
  load i32, i32* %arg28
  load i32, i32* %arg29
  load i32, i32* %arg30
  load i32, i32* %arg31
  load i32, i32* %arg32
  load i32, i32* %arg33
  load i32, i32* %arg34
  load i32, i32* %arg35

  ret void
}
