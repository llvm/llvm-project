; RUN: llc < %s -mtriple=thumbv7-apple-ios -mcpu=cortex-a8 -pre-RA-sched=source -disable-post-ra | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6m-apple-ios -mcpu=cortex-m0 -pre-RA-sched=source -disable-post-ra -mattr=+strict-align | FileCheck %s -check-prefix=CHECK-T1
%struct.x = type { i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }

@src = external global %struct.x
@dst = external global %struct.x

@.str1 = private unnamed_addr constant [31 x i8] c"DHRYSTONE PROGRAM, SOME STRING\00", align 1
@.str2 = private unnamed_addr constant [36 x i8] c"DHRYSTONE PROGRAM, SOME STRING BLAH\00", align 1
@.str3 = private unnamed_addr constant [24 x i8] c"DHRYSTONE PROGRAM, SOME\00", align 1
@.str4 = private unnamed_addr constant [18 x i8] c"DHRYSTONE PROGR  \00", align 1
@.str5 = private unnamed_addr constant [7 x i8] c"DHRYST\00", align 1
@.str6 = private unnamed_addr constant [14 x i8] c"/tmp/rmXXXXXX\00", align 1
@spool.splbuf = internal global [512 x i8] zeroinitializer, align 16

define i32 @t0() {
entry:
; CHECK-LABEL: t0:
; CHECK: vldr [[REG1:d[0-9]+]],
; CHECK: vstr [[REG1]],
; CHECK-T1-LABEL: t0:
; CHECK-T1: ldrb [[TREG1:r[0-9]]],
; CHECK-T1: strb [[TREG1]],
; CHECK-T1: ldrh [[TREG2:r[0-9]]],
; CHECK-T1: strh [[TREG2]]
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds (%struct.x, %struct.x* @dst, i32 0, i32 0), i8* getelementptr inbounds (%struct.x, %struct.x* @src, i32 0, i32 0), i32 11, i32 8, i1 false)
  ret i32 0
}

define void @t1(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: movs [[INC:r[0-9]+]], #15
; CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r1], [[INC]]
; CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r0], [[INC]]
; CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r1]
; CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str1, i64 0, i64 0), i64 31, i32 1, i1 false)
  ret void
}

define void @t2(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r1]!
; CHECK: movs [[INC:r[0-9]+]], #32
; CHECK: add.w   r3, r0, #16
; CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r0], [[INC]]
; CHECK: movw [[REG2:r[0-9]+]], #16716
; CHECK: movt [[REG2:r[0-9]+]], #72
; CHECK: str [[REG2]], [r0]
; CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r1]
; CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r3]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str2, i64 0, i64 0), i64 36, i32 1, i1 false)
  ret void
}

define void @t3(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: vld1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r1]!
; CHECK: vst1.8 {d{{[0-9]+}}, d{{[0-9]+}}}, [r0]!
; CHECK: vldr d{{[0-9]+}}, [r1]
; CHECK: vst1.8 {d{{[0-9]+}}}, [r0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str3, i64 0, i64 0), i64 24, i32 1, i1 false)
  ret void
}

define void @t4(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: vld1.64 {[[REG3:d[0-9]+]], [[REG4:d[0-9]+]]}, [r1]
; CHECK: vst1.8 {[[REG3]], [[REG4]]}, [r0]!
; CHECK: strh [[REG5:r[0-9]+]], [r0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str4, i64 0, i64 0), i64 18, i32 1, i1 false)
  ret void
}

define void @t5(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: movs [[REG5:r[0-9]+]], #0
; CHECK: strb [[REG5]], [r0, #6]
; CHECK: movw [[REG6:r[0-9]+]], #21587
; CHECK: strh [[REG6]], [r0, #4]
; CHECK: movw [[REG7:r[0-9]+]], #18500
; CHECK: movt [[REG7:r[0-9]+]], #22866
; CHECK: str [[REG7]]
; CHECK-T1-LABEL: t5:
; CHECK-T1: movs [[TREG3:r[0-9]]],
; CHECK-T1: strb [[TREG3]],
; CHECK-T1: movs [[TREG4:r[0-9]]],
; CHECK-T1: strb [[TREG4]],
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str5, i64 0, i64 0), i64 7, i32 1, i1 false)
  ret void
}

define void @t6() nounwind {
entry:
; CHECK-LABEL: t6:
; CHECK: vldr [[REG9:d[0-9]+]], [r0]
; CHECK: vstr [[REG9]], [r1]
; CHECK: adds r1, #6
; CHECK: adds r0, #6
; CHECK: vld1.16
; CHECK: vst1.16
; CHECK-T1-LABEL: t6:
; CHECK-T1: movs [[TREG5:r[0-9]]],
; CHECK-T1: strh [[TREG5]],
; CHECK-T1: ldr [[TREG6:r[0-9]]],
; CHECK-T1: str [[TREG6]]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([512 x i8], [512 x i8]* @spool.splbuf, i64 0, i64 0), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str6, i64 0, i64 0), i64 14, i32 1, i1 false)
  ret void
}

%struct.Foo = type { i32, i32, i32, i32 }

define void @t7(%struct.Foo* nocapture %a, %struct.Foo* nocapture %b) nounwind {
entry:
; CHECK-LABEL: t7:
; CHECK: vld1.32
; CHECK: vst1.32
; CHECK-T1-LABEL: t7:
; CHECK-T1: ldr
; CHECK-T1: str
  %0 = bitcast %struct.Foo* %a to i8*
  %1 = bitcast %struct.Foo* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* %1, i32 16, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
