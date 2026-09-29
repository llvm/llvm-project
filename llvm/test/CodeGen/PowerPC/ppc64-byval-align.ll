; RUN: llc -verify-machineinstrs -O1 < %s -mcpu=pwr7 | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.test = type { i64, [8 x i8] }
%struct.pad = type { [8 x i64] }

@gt = common global %struct.test zeroinitializer, align 16
@gp = common global %struct.pad zeroinitializer, align 8

define signext i32 @callee1(i32 signext %x, ptr byval(%struct.test) align 16 nocapture readnone %y, i32 signext %z) {
entry:
  ret i32 %z
}
; CHECK-LABEL: @callee1
; CHECK: mr 3, 7
; CHECK: blr

declare signext i32 @test1(i32 signext, ptr byval(%struct.test) align 16, i32 signext)
define void @caller1(i32 signext %z) {
entry:
  %call = tail call signext i32 @test1(i32 signext 0, ptr byval(%struct.test) align 16 @gt, i32 signext %z)
  ret void
}
; CHECK-LABEL: @caller1
; CHECK: mr 7, 3
; CHECK: bl test1

define i64 @callee2(ptr byval(%struct.pad) nocapture readnone %x, i32 signext %y, ptr byval(%struct.test) align 16 nocapture readonly %z) {
entry:
  %0 = load i64, ptr %z, align 16
  ret i64 %0
}
; CHECK-LABEL: @callee2
; CHECK: ld {{[0-9]+}}, 128(1)
; CHECK: blr

declare i64 @test2(ptr byval(%struct.pad), i32 signext, ptr byval(%struct.test) align 16)
define void @caller2(i64 %z) {
entry:
  %tmp = alloca %struct.test, align 16
  store i64 %z, ptr %tmp, align 16
  %call = call i64 @test2(ptr byval(%struct.pad) @gp, i32 signext 0, ptr byval(%struct.test) align 16 %tmp)
  ret void
}
; CHECK-LABEL: @caller2
; CHECK-DAG: std 3, [[OFF:[0-9]+]](1)
; CHECK-DAG: addi [[REG1:[0-9]+]], 1, [[OFF]]
;
; CHECK-DAG: lxvw4x [[REG2:[0-9]+]], 0, [[REG1]]
; CHECK-DAG: li [[REG3:[0-9]+]], 128
; CHECK:     stxvw4x 0, 1, [[REG3]]
; CHECK:     bl test2

