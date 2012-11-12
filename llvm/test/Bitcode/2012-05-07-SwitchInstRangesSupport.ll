; RUN: rm -f %t.bc
; RUN: rm -f %t.ll
; RUN: rm -f %t2.bc
; RUN: rm -f %t2.ll
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | tail -n +2 > %t.ll
; RUN: llvm-as %t.ll -o %t2.bc
; RUN: llvm-dis %t2.bc -o - | tail -n +2 > %t2.ll
; RUN: llvm-diff %t.ll %t2.ll

define void @test() {
  %mem = alloca i32
  store i32 2, i32* %mem
  %c = load i32* %mem
  switch i32 %c, label %exit [
      i32 1, label %exit
      i32 2, label %exit
  ]
exit:
  ret void
}
define void @test_wide() {
  %mem = alloca i256
  store i256 2, i256* %mem
  %c = load i256* %mem
  switch i256 %c, label %exit [
      i256 123456789012345678901234567890, label %exit
      i256 2, label %exit
  ]
exit:
  ret void
}

