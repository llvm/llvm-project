; RUN: llc -mtriple aarch64-unknown-windows-msvc %s -filetype asm -o - | FileCheck %s
; RUN: llc -mtriple aarch64-unknown-linux-gnu %s -filetype asm -o - | FileCheck %s
; RUN: llc -mtriple aarch64-apple-macosx %s -filetype asm -o - | FileCheck %s
; Regression test for https://github.com/llvm/llvm-project/issues/59751

define void @"func"(i32** swifterror %0) #0 {
; CHECK-LABEL: func:
; CHECK:       {{.*}}%bb.0:
; CHECK-NEXT:    b {{\.?}}LBB0_2
; CHECK-NEXT:  {{\.?}}LBB0_1:{{.*}}%thirtythree
; CHECK-NEXT:  {{.*}}=>This Inner Loop Header: Depth=1
; CHECK-NEXT:    b {{\.?}}LBB0_1
; CHECK-NEXT:  {{\.?}}LBB0_2:{{.*}}%thirtyeight
; CHECK-NEXT:    b {{\.?}}LBB0_3
; CHECK-NEXT:  {{\.?}}LBB0_3:{{.*}}%thirtythree.preheader
; CHECK-NEXT:    b {{\.?}}LBB0_1
  br label %thirtyeight

five:
  br label %UelOc2l.exit

common.ret:
  ret void

UelOc2l.exit:
  %a = getelementptr inbounds [754 x i8*], [754 x i8*]* undef, i32 undef, i32 undef
  %b = load i8*, i8** %a, align 8
  %c = bitcast i8* %b to void ()*
  call void %c()
  br label %common.ret

thirtythree:
  br i1 false, label %UelOc2l.exit, label %thirtythree

thirtyeight:
  br i1 undef, label %thirtyeight, label %thirtythree
}

attributes #0 = { noinline optnone }
