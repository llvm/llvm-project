; RUN: llc -mtriple aarch64-unknown-windows-msvc %s -filetype asm -o - | FileCheck %s
; RUN: llc -mtriple aarch64-unknown-linux-gnu %s -filetype asm -o - | FileCheck %s
; RUN: llc -mtriple aarch64-apple-macosx %s -filetype asm -o - | FileCheck %s
; Regression test for https://github.com/llvm/llvm-project/issues/59751

define void @"func"(ptr swifterror %0) #0 {
; CHECK-LABEL: func:
; CHECK:  {{.*}}%bb.0:
; CHECK:    b {{.*}}LBB0_5
; CHECK:  {{.*}}LBB0_1:{{.*}}%common.ret
; CHECK:  {{.*}}LBB0_2:
; CHECK:    b {{.*}}LBB0_1
; CHECK:  {{.*}}LBB0_4:{{.*}}%thirtythree
; CHECK:    {{.*}}=>This Inner Loop Header: Depth=1
; CHECK:    b {{.*}}LBB0_4
; CHECK:  {{.*}}LBB0_5:{{.*}}%thirtyeight
; CHECK:    {{.*}}=>This Inner Loop Header: Depth=1
; CHECK:    b {{.*}}LBB0_6
; CHECK:  {{.*}}LBB0_6:{{.*}}%thirtythree.preheader
; CHECK:    b {{.*}}LBB0_4
  br label %thirtyeight

five:
  br label %UelOc2l.exit

common.ret:
  ret void

UelOc2l.exit:
  %a = getelementptr inbounds [754 x ptr], ptr undef, i32 undef, i32 undef
  %b = load ptr, ptr %a, align 8
  %c = bitcast ptr %b to ptr
  call void %c()
  br label %common.ret

thirtythree:
  br i1 false, label %UelOc2l.exit, label %thirtythree

thirtyeight:
  br i1 poison, label %thirtyeight, label %thirtythree
}

attributes #0 = { noinline optnone }
