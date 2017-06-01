; RUN: llc -verify-machineinstrs -O1 < %s -march=ppc64 -mcpu=pwr7 | FileCheck %s

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.test = type { i64, [8 x i8] }
%struct.pad = type { [8 x i64] }

@gt = common global %struct.test zeroinitializer, align 16
@gp = common global %struct.pad zeroinitializer, align 8

define signext i32 @callee1(i32 signext %x, %struct.test* byval align 16 nocapture readnone %y, i32 signext %z) {
entry:
  ret i32 %z
}
; CHECK-LABEL: @callee1
; CHECK: mr 3, 7
; CHECK: blr

declare signext i32 @test1(i32 signext, %struct.test* byval align 16, i32 signext)
define void @caller1(i32 signext %z) {
entry:
  %call = tail call signext i32 @test1(i32 signext 0, %struct.test* byval align 16 @gt, i32 signext %z)
  ret void
}
; CHECK-LABEL: @caller1
; CHECK: mr [[REG:[0-9]+]], 3
; CHECK: mr 7, [[REG]]
; CHECK: bl test1

define i64 @callee2(%struct.pad* byval nocapture readnone %x, i32 signext %y, %struct.test* byval align 16 nocapture readonly %z) {
entry:
  %x1 = getelementptr inbounds %struct.test, %struct.test* %z, i64 0, i32 0
  %0 = load i64, i64* %x1, align 16
  ret i64 %0
}
; CHECK-LABEL: @callee2
; CHECK: ld 3, 128(1)
; CHECK: blr

declare i64 @test2(%struct.pad* byval, i32 signext, %struct.test* byval align 16)
define void @caller2(i64 %z) {
entry:
  %tmp = alloca %struct.test, align 16
  %.compoundliteral.sroa.0.0..sroa_idx = getelementptr inbounds %struct.test, %struct.test* %tmp, i64 0, i32 0
  store i64 %z, i64* %.compoundliteral.sroa.0.0..sroa_idx, align 16
  %call = call i64 @test2(%struct.pad* byval @gp, i32 signext 0, %struct.test* byval align 16 %tmp)
  ret void
}
; CHECK-LABEL: @caller2
; CHECK: std 3, [[OFF:[0-9]+]](1)
; CHECK: addi [[REG1:[0-9]+]], 1, [[OFF]]
; CHECK: lxvw4x [[REG2:[0-9]+]], 0, [[REG1]]
; CHECK: li [[REG3:[0-9]+]], 128
; CHECK: stxvw4x 0, 1, [[REG3]]
; CHECK: bl test2

