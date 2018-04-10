; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cyclone | FileCheck %s

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
; CHECK: ldrb [[REG0:w[0-9]+]], [x[[BASEREG:[0-9]+]], #10]
; CHECK: strb [[REG0]], [x[[BASEREG2:[0-9]+]], #10]
; CHECK: ldrh [[REG1:w[0-9]+]], [x[[BASEREG]], #8]
; CHECK: strh [[REG1]], [x[[BASEREG2]], #8]
; CHECK: ldr [[REG2:x[0-9]+]],
; CHECK: str [[REG2]],
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 getelementptr inbounds (%struct.x, %struct.x* @dst, i32 0, i32 0), i8* align 8 getelementptr inbounds (%struct.x, %struct.x* @src, i32 0, i32 0), i32 11, i1 false)
  ret i32 0
}

define void @t1(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t1:
; CHECK: ldur [[DEST:q[0-9]+]], [x[[BASEREG:[0-9]+]], #15]
; CHECK: stur [[DEST]], [x0, #15]
; CHECK: ldr [[DEST:q[0-9]+]], [x[[BASEREG]]]
; CHECK: str [[DEST]], [x0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str1, i64 0, i64 0), i64 31, i1 false)
  ret void
}

define void @t2(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: mov [[REG3:w[0-9]+]]
; CHECK: movk [[REG3]],
; CHECK: str [[REG3]], [x0, #32]
; CHECK: ldp [[DEST1:q[0-9]+]], [[DEST2:q[0-9]+]], [x{{[0-9]+}}]
; CHECK: stp [[DEST1]], [[DEST2]], [x0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str2, i64 0, i64 0), i64 36, i1 false)
  ret void
}

define void @t3(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: ldr [[REG4:x[0-9]+]], [x[[BASEREG:[0-9]+]], #16]
; CHECK: str [[REG4]], [x0, #16]
; CHECK: ldr [[DEST:q[0-9]+]], [x[[BASEREG]]]
; CHECK: str [[DEST]], [x0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str3, i64 0, i64 0), i64 24, i1 false)
  ret void
}

define void @t4(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: orr [[REG5:w[0-9]+]], wzr, #0x20
; CHECK: strh [[REG5]], [x0, #16]
; CHECK: ldr [[REG6:q[0-9]+]], [x{{[0-9]+}}]
; CHECK: str [[REG6]], [x0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str4, i64 0, i64 0), i64 18, i1 false)
  ret void
}

define void @t5(i8* nocapture %C) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: strb wzr, [x0, #6]
; CHECK: mov [[REG7:w[0-9]+]], #21587
; CHECK: strh [[REG7]], [x0, #4]
; CHECK: mov [[REG8:w[0-9]+]],
; CHECK: movk [[REG8]],
; CHECK: str [[REG8]], [x0]
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %C, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str5, i64 0, i64 0), i64 7, i1 false)
  ret void
}

define void @t6() nounwind {
entry:
; CHECK-LABEL: t6:
; CHECK: ldur [[REG9:x[0-9]+]], [x{{[0-9]+}}, #6]
; CHECK: stur [[REG9]], [x{{[0-9]+}}, #6]
; CHECK: ldr
; CHECK: str
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* getelementptr inbounds ([512 x i8], [512 x i8]* @spool.splbuf, i64 0, i64 0), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str6, i64 0, i64 0), i64 14, i1 false)
  ret void
}

%struct.Foo = type { i32, i32, i32, i32 }

define void @t7(%struct.Foo* nocapture %a, %struct.Foo* nocapture %b) nounwind {
entry:
; CHECK: t7
; CHECK: ldr [[REG10:q[0-9]+]], [x1]
; CHECK: str [[REG10]], [x0]
  %0 = bitcast %struct.Foo* %a to i8*
  %1 = bitcast %struct.Foo* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %0, i8* align 4 %1, i32 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind
