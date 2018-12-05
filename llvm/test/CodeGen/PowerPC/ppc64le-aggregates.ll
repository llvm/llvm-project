; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mcpu=pwr8 \
; RUN:   -mattr=+altivec -mattr=-vsx | FileCheck %s
; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mattr=+altivec \
; RUN:   -mattr=-vsx | FileCheck %s
; RUN: llc -relocation-model=pic -verify-machineinstrs < %s -mcpu=pwr9 \
; RUN:   -mattr=-direct-move -mattr=+altivec | FileCheck %s

; Currently VSX support is disabled for this test because we generate lxsdx
; instead of lfd, and stxsdx instead of stfd.  That is a poor choice when we
; have reg+imm addressing, and is on the list of things to be fixed.
; The second run step is to ensure that -march=ppc64le is adequate to select
; the same feature set as with -mcpu=pwr8 since that is the baseline for ppc64le.

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

;
; Verify use of registers for float/vector aggregate return.
;

define [8 x float] @return_float([8 x float] %x) {
entry:
  ret [8 x float] %x
}
; CHECK-LABEL: @return_float
; CHECK: %entry
; CHECK-NEXT: blr

define [8 x double] @return_double([8 x double] %x) {
entry:
  ret [8 x double] %x
}
; CHECK-LABEL: @return_double
; CHECK: %entry
; CHECK-NEXT: blr

define [4 x ppc_fp128] @return_ppcf128([4 x ppc_fp128] %x) {
entry:
  ret [4 x ppc_fp128] %x
}
; CHECK-LABEL: @return_ppcf128
; CHECK: %entry
; CHECK-NEXT: blr

define [8 x <4 x i32>] @return_v4i32([8 x <4 x i32>] %x) {
entry:
  ret [8 x <4 x i32>] %x
}
; CHECK-LABEL: @return_v4i32
; CHECK: %entry
; CHECK-NEXT: blr


;
; Verify amount of space taken up by aggregates in the parameter save area.
;

define i64 @callee_float([7 x float] %a, [7 x float] %b, i64 %c) {
entry:
  ret i64 %c
}
; CHECK-LABEL: @callee_float
; CHECK: ld 3, 96(1)
; CHECK: blr

define void @caller_float(i64 %x, [7 x float] %y) {
entry:
  tail call void @test_float([7 x float] %y, [7 x float] %y, i64 %x)
  ret void
}
; CHECK-LABEL: @caller_float
; CHECK: std 3, 96(1)
; CHECK: bl test_float

declare void @test_float([7 x float], [7 x float], i64)

define i64 @callee_double(i64 %a, [7 x double] %b, i64 %c) {
entry:
  ret i64 %c
}
; CHECK-LABEL: @callee_double
; CHECK: ld 3, 96(1)
; CHECK: blr

define void @caller_double(i64 %x, [7 x double] %y) {
entry:
  tail call void @test_double(i64 %x, [7 x double] %y, i64 %x)
  ret void
}
; CHECK-LABEL: @caller_double
; CHECK: std 3, 96(1)
; CHECK: bl test_double

declare void @test_double(i64, [7 x double], i64)

define i64 @callee_ppcf128(i64 %a, [4 x ppc_fp128] %b, i64 %c) {
entry:
  ret i64 %c
}
; CHECK-LABEL: @callee_ppcf128
; CHECK: ld 3, 104(1)
; CHECK: blr

define void @caller_ppcf128(i64 %x, [4 x ppc_fp128] %y) {
entry:
  tail call void @test_ppcf128(i64 %x, [4 x ppc_fp128] %y, i64 %x)
  ret void
}
; CHECK-LABEL: @caller_ppcf128
; CHECK: std 3, 104(1)
; CHECK: bl test_ppcf128

declare void @test_ppcf128(i64, [4 x ppc_fp128], i64)

define i64 @callee_i64(i64 %a, [7 x i64] %b, i64 %c) {
entry:
  ret i64 %c
}
; CHECK-LABEL: @callee_i64
; CHECK: ld 3, 96(1)
; CHECK: blr

define void @caller_i64(i64 %x, [7 x i64] %y) {
entry:
  tail call void @test_i64(i64 %x, [7 x i64] %y, i64 %x)
  ret void
}
; CHECK-LABEL: @caller_i64
; CHECK: std 3, 96(1)
; CHECK: bl test_i64

declare void @test_i64(i64, [7 x i64], i64)

define i64 @callee_i128(i64 %a, [4 x i128] %b, i64 %c) {
entry:
  ret i64 %c
}
; CHECK-LABEL: @callee_i128
; CHECK: ld 3, 112(1)
; CHECK: blr

define void @caller_i128(i64 %x, [4 x i128] %y) {
entry:
  tail call void @test_i128(i64 %x, [4 x i128] %y, i64 %x)
  ret void
}
; CHECK-LABEL: @caller_i128
; CHECK: std 3, 112(1)
; CHECK: bl test_i128

declare void @test_i128(i64, [4 x i128], i64)

define i64 @callee_v4i32(i64 %a, [4 x <4 x i32>] %b, i64 %c) {
entry:
  ret i64 %c
}
; CHECK-LABEL: @callee_v4i32
; CHECK: ld 3, 112(1)
; CHECK: blr

define void @caller_v4i32(i64 %x, [4 x <4 x i32>] %y) {
entry:
  tail call void @test_v4i32(i64 %x, [4 x <4 x i32>] %y, i64 %x)
  ret void
}
; CHECK-LABEL: @caller_v4i32
; CHECK: std 3, 112(1)
; CHECK: bl test_v4i32

declare void @test_v4i32(i64, [4 x <4 x i32>], i64)


;
; Verify handling of floating point arguments in GPRs
;

%struct.float8 = type { [8 x float] }
%struct.float5 = type { [5 x float] }
%struct.float2 = type { [2 x float] }

@g8 = common global %struct.float8 zeroinitializer, align 4
@g5 = common global %struct.float5 zeroinitializer, align 4
@g2 = common global %struct.float2 zeroinitializer, align 4

define float @callee0([7 x float] %a, [7 x float] %b) {
entry:
  %b.extract = extractvalue [7 x float] %b, 6
  ret float %b.extract
}
; CHECK-LABEL: @callee0
; CHECK: stw 10, [[OFF:.*]](1)
; CHECK: lfs 1, [[OFF]](1)
; CHECK: blr

define void @caller0([7 x float] %a) {
entry:
  tail call void @test0([7 x float] %a, [7 x float] %a)
  ret void
}
; CHECK-LABEL: @caller0
; CHECK-DAG: fmr 8, 1
; CHECK-DAG: fmr 9, 2
; CHECK-DAG: fmr 10, 3
; CHECK-DAG: fmr 11, 4
; CHECK-DAG: fmr 12, 5
; CHECK-DAG: fmr 13, 6
; CHECK-DAG: stfs 7, [[OFF:[0-9]+]](1)
; CHECK-DAG: lwz 10, [[OFF]](1)
; CHECK: bl test0

declare void @test0([7 x float], [7 x float])

define float @callee1([8 x float] %a, [8 x float] %b) {
entry:
  %b.extract = extractvalue [8 x float] %b, 7
  ret float %b.extract
}
; CHECK-LABEL: @callee1
; CHECK: rldicl [[REG:[0-9]+]], 10, 32, 32
; CHECK: stw [[REG]], [[OFF:.*]](1)
; CHECK: lfs 1, [[OFF]](1)
; CHECK: blr

define void @caller1([8 x float] %a) {
entry:
  tail call void @test1([8 x float] %a, [8 x float] %a)
  ret void
}
; CHECK-LABEL: @caller1
; CHECK-DAG: fmr 9, 1
; CHECK-DAG: fmr 10, 2
; CHECK-DAG: fmr 11, 3
; CHECK-DAG: fmr 12, 4
; CHECK-DAG: fmr 13, 5
; CHECK-DAG: stfs 5, [[OFF0:[0-9]+]](1)
; CHECK-DAG: stfs 6, [[OFF1:[0-9]+]](1)
; CHECK-DAG: stfs 7, [[OFF2:[0-9]+]](1)
; CHECK-DAG: stfs 8, [[OFF3:[0-9]+]](1)
; CHECK-DAG: lwz 9, [[OFF0]](1)
; CHECK-DAG: lwz [[REG1:[0-9]+]], [[OFF1]](1)
; CHECK-DAG: lwz 10, [[OFF2]](1)
; CHECK-DAG: lwz [[REG3:[0-9]+]], [[OFF3]](1)
; CHECK-DAG: rldimi 9, [[REG1]], 32, 0
; CHECK-DAG: rldimi 10, [[REG3]], 32, 0
; CHECK: bl test1

declare void @test1([8 x float], [8 x float])

define float @callee2([8 x float] %a, [5 x float] %b, [2 x float] %c) {
entry:
  %c.extract = extractvalue [2 x float] %c, 1
  ret float %c.extract
}
; CHECK-LABEL: @callee2
; CHECK: rldicl [[REG:[0-9]+]], 10, 32, 32
; CHECK: stw [[REG]], [[OFF:.*]](1)
; CHECK: lfs 1, [[OFF]](1)
; CHECK: blr

define void @caller2() {
entry:
  %0 = load [8 x float], [8 x float]* getelementptr inbounds (%struct.float8, %struct.float8* @g8, i64 0, i32 0), align 4
  %1 = load [5 x float], [5 x float]* getelementptr inbounds (%struct.float5, %struct.float5* @g5, i64 0, i32 0), align 4
  %2 = load [2 x float], [2 x float]* getelementptr inbounds (%struct.float2, %struct.float2* @g2, i64 0, i32 0), align 4
  tail call void @test2([8 x float] %0, [5 x float] %1, [2 x float] %2)
  ret void
}
; CHECK-LABEL: @caller2
; CHECK: ld {{[0-9]+}}, .LC
; CHECK-DAG: lfs 1, 0({{[0-9]+}})
; CHECK-DAG: lfs 2, 4({{[0-9]+}})
; CHECK-DAG: lfs 3, 8({{[0-9]+}})
; CHECK-DAG: lfs 4, 12({{[0-9]+}})
; CHECK-DAG: lfs 5, 16({{[0-9]+}})
; CHECK-DAG: lfs 6, 20({{[0-9]+}})
; CHECK-DAG: lfs 7, 24({{[0-9]+}})
; CHECK-DAG: lfs 8, 28({{[0-9]+}})

; CHECK-DAG: lfs 9, 0({{[0-9]+}})
; CHECK-DAG: lfs 10, 4({{[0-9]+}})
; CHECK-DAG: lfs 11, 8({{[0-9]+}})
; CHECK-DAG: lfs 12, 12({{[0-9]+}})
; CHECK-DAG: lfs 13, 16({{[0-9]+}})

; CHECK-DAG: ld 10, 0({{[0-9]+}})
; CHECK: bl test2

declare void @test2([8 x float], [5 x float], [2 x float])

define double @callee3([8 x float] %a, [5 x float] %b, double %c) {
entry:
  ret double %c
}
; CHECK-LABEL: @callee3
; CHECK: std 10, [[OFF:.*]](1)
; CHECK: lfd 1, [[OFF]](1)
; CHECK: blr

define void @caller3(double %d) {
entry:
  %0 = load [8 x float], [8 x float]* getelementptr inbounds (%struct.float8, %struct.float8* @g8, i64 0, i32 0), align 4
  %1 = load [5 x float], [5 x float]* getelementptr inbounds (%struct.float5, %struct.float5* @g5, i64 0, i32 0), align 4
  tail call void @test3([8 x float] %0, [5 x float] %1, double %d)
  ret void
}
; CHECK-LABEL: @caller3
; CHECK: stfd 1, [[OFF:.*]](1)
; CHECK: ld 10, [[OFF]](1)
; CHECK: bl test3

declare void @test3([8 x float], [5 x float], double)

define float @callee4([8 x float] %a, [5 x float] %b, float %c) {
entry:
  ret float %c
}
; CHECK-LABEL: @callee4
; CHECK: stw 10, [[OFF:.*]](1)
; CHECK: lfs 1, [[OFF]](1)
; CHECK: blr

define void @caller4(float %f) {
entry:
  %0 = load [8 x float], [8 x float]* getelementptr inbounds (%struct.float8, %struct.float8* @g8, i64 0, i32 0), align 4
  %1 = load [5 x float], [5 x float]* getelementptr inbounds (%struct.float5, %struct.float5* @g5, i64 0, i32 0), align 4
  tail call void @test4([8 x float] %0, [5 x float] %1, float %f)
  ret void
}
; CHECK-LABEL: @caller4
; CHECK: stfs 1, [[OFF:.*]](1)
; CHECK: lwz 10, [[OFF]](1)
; CHECK: bl test4

declare void @test4([8 x float], [5 x float], float)

