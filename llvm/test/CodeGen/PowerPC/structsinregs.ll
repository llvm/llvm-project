; RUN: llc -verify-machineinstrs -mcpu=ppc64 -O0 -frame-pointer=all -fast-isel=false < %s | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.s1 = type { i8 }
%struct.s2 = type { i16 }
%struct.s4 = type { i32 }
%struct.t1 = type { i8 }
%struct.t3 = type <{ i16, i8 }>
%struct.t5 = type <{ i32, i8 }>
%struct.t6 = type <{ i32, i16 }>
%struct.t7 = type <{ i32, i16, i8 }>
%struct.s3 = type { i16, i8 }
%struct.s5 = type { i32, i8 }
%struct.s6 = type { i32, i16 }
%struct.s7 = type { i32, i16, i8 }
%struct.t2 = type <{ i16 }>
%struct.t4 = type <{ i32 }>

@caller1.p1 = private unnamed_addr constant %struct.s1 { i8 1 }, align 1
@caller1.p2 = private unnamed_addr constant %struct.s2 { i16 2 }, align 2
@caller1.p3 = private unnamed_addr constant { i16, i8, i8 } { i16 4, i8 8, i8 undef }, align 2
@caller1.p4 = private unnamed_addr constant %struct.s4 { i32 16 }, align 4
@caller1.p5 = private unnamed_addr constant { i32, i8, [3 x i8] } { i32 32, i8 64, [3 x i8] undef }, align 4
@caller1.p6 = private unnamed_addr constant { i32, i16, [2 x i8] } { i32 128, i16 256, [2 x i8] undef }, align 4
@caller1.p7 = private unnamed_addr constant { i32, i16, i8, i8 } { i32 512, i16 1024, i8 -3, i8 undef }, align 4
@caller2.p1 = private unnamed_addr constant %struct.t1 { i8 1 }, align 1
@caller2.p2 = private unnamed_addr constant { i16 } { i16 2 }, align 1
@caller2.p3 = private unnamed_addr constant %struct.t3 <{ i16 4, i8 8 }>, align 1
@caller2.p4 = private unnamed_addr constant { i32 } { i32 16 }, align 1
@caller2.p5 = private unnamed_addr constant %struct.t5 <{ i32 32, i8 64 }>, align 1
@caller2.p6 = private unnamed_addr constant %struct.t6 <{ i32 128, i16 256 }>, align 1
@caller2.p7 = private unnamed_addr constant %struct.t7 <{ i32 512, i16 1024, i8 -3 }>, align 1

define i32 @caller1() nounwind {
entry:
  %p1 = alloca %struct.s1
  %p2 = alloca %struct.s2
  %p3 = alloca %struct.s3
  %p4 = alloca %struct.s4
  %p5 = alloca %struct.s5
  %p6 = alloca %struct.s6
  %p7 = alloca %struct.s7
  call void @llvm.memcpy.p0.p0.i64(ptr %p1, ptr @caller1.p1, i64 1, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %p2, ptr align 2 @caller1.p2, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 2 %p3, ptr align 2 @caller1.p3, i64 4, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %p4, ptr align 4 @caller1.p4, i64 4, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %p5, ptr align 4 @caller1.p5, i64 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %p6, ptr align 4 @caller1.p6, i64 8, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %p7, ptr align 4 @caller1.p7, i64 8, i1 false)
  %call = call i32 @callee1(ptr byval(%struct.s1) %p1, ptr byval(%struct.s2) %p2, ptr byval(%struct.s3) %p3, ptr byval(%struct.s4) %p4, ptr byval(%struct.s5) %p5, ptr byval(%struct.s6) %p6, ptr byval(%struct.s7) %p7)
  ret i32 %call

; CHECK-LABEL: caller1
; CHECK: ld 9, 112(31)
; CHECK: ld 8, 120(31)
; CHECK: ld 7, 128(31)
; CHECK: lwz 6, 136(31)
; CHECK: lwz 5, 144(31)
; CHECK: lhz 4, 152(31)
; CHECK: lbz 3, 160(31)
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

define internal i32 @callee1(ptr byval(%struct.s1) %v1, ptr byval(%struct.s2) %v2, ptr byval(%struct.s3) %v3, ptr byval(%struct.s4) %v4, ptr byval(%struct.s5) %v5, ptr byval(%struct.s6) %v6, ptr byval(%struct.s7) %v7) nounwind {
entry:
  %0 = load i8, ptr %v1, align 1
  %conv = zext i8 %0 to i32
  %1 = load i16, ptr %v2, align 2
  %conv2 = sext i16 %1 to i32
  %add = add nsw i32 %conv, %conv2
  %2 = load i16, ptr %v3, align 2
  %conv4 = sext i16 %2 to i32
  %add5 = add nsw i32 %add, %conv4
  %3 = load i32, ptr %v4, align 4
  %add7 = add nsw i32 %add5, %3
  %4 = load i32, ptr %v5, align 4
  %add9 = add nsw i32 %add7, %4
  %5 = load i32, ptr %v6, align 4
  %add11 = add nsw i32 %add9, %5
  %6 = load i32, ptr %v7, align 4
  %add13 = add nsw i32 %add11, %6
  ret i32 %add13

; CHECK-LABEL: callee1
; CHECK-DAG: std 9, 96(1)
; CHECK-DAG: std 8, 88(1)
; CHECK-DAG: std 7, 80(1)
; CHECK-DAG: stw 6, 76(1)
; CHECK-DAG: stw 5, 68(1)
; CHECK-DAG: sth 4, 62(1)
; CHECK-DAG: stb 3, 55(1)
; CHECK-DAG: lha {{[0-9]+}}, 62(1)
; CHECK-DAG: lha {{[0-9]+}}, 68(1)
; CHECK-DAG: lbz {{[0-9]+}}, 55(1)
; CHECK-DAG: lwz {{[0-9]+}}, 76(1)
; CHECK-DAG: lwz {{[0-9]+}}, 80(1)
; CHECK-DAG: lwz {{[0-9]+}}, 88(1)
; CHECK-DAG: lwz {{[0-9]+}}, 96(1)
}

define i32 @caller2() nounwind {
entry:
  %p1 = alloca %struct.t1
  %p2 = alloca %struct.t2
  %p3 = alloca %struct.t3
  %p4 = alloca %struct.t4
  %p5 = alloca %struct.t5
  %p6 = alloca %struct.t6
  %p7 = alloca %struct.t7
  call void @llvm.memcpy.p0.p0.i64(ptr %p1, ptr @caller2.p1, i64 1, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %p2, ptr @caller2.p2, i64 2, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %p3, ptr @caller2.p3, i64 3, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %p4, ptr @caller2.p4, i64 4, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %p5, ptr @caller2.p5, i64 5, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %p6, ptr @caller2.p6, i64 6, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %p7, ptr @caller2.p7, i64 7, i1 false)
  %call = call i32 @callee2(ptr byval(%struct.t1) %p1, ptr byval(%struct.t2) %p2, ptr byval(%struct.t3) %p3, ptr byval(%struct.t4) %p4, ptr byval(%struct.t5) %p5, ptr byval(%struct.t6) %p6, ptr byval(%struct.t7) %p7)
  ret i32 %call

; CHECK-LABEL: caller2
; CHECK: stb {{[0-9]+}}, 71(1)
; CHECK: sth {{[0-9]+}}, 69(1)
; CHECK: stb {{[0-9]+}}, 87(1)
; CHECK: stw {{[0-9]+}}, 83(1)
; CHECK: sth {{[0-9]+}}, 94(1)
; CHECK: stw {{[0-9]+}}, 90(1)
; CHECK: stw {{[0-9]+}}, 100(1)
; CHECK: stw {{[0-9]+}}, 97(1)
; CHECK: ld 9, 96(1)
; CHECK: ld 8, 88(1)
; CHECK: ld 7, 80(1)
; CHECK: lwz 6, 136(31)
; CHECK: ld 5, 64(1)
; CHECK: lhz 4, 152(31)
; CHECK: lbz 3, 160(31)
}

define internal i32 @callee2(ptr byval(%struct.t1) %v1, ptr byval(%struct.t2) %v2, ptr byval(%struct.t3) %v3, ptr byval(%struct.t4) %v4, ptr byval(%struct.t5) %v5, ptr byval(%struct.t6) %v6, ptr byval(%struct.t7) %v7) nounwind {
entry:
  %0 = load i8, ptr %v1, align 1
  %conv = zext i8 %0 to i32
  %1 = load i16, ptr %v2, align 1
  %conv2 = sext i16 %1 to i32
  %add = add nsw i32 %conv, %conv2
  %2 = load i16, ptr %v3, align 1
  %conv4 = sext i16 %2 to i32
  %add5 = add nsw i32 %add, %conv4
  %3 = load i32, ptr %v4, align 1
  %add7 = add nsw i32 %add5, %3
  %4 = load i32, ptr %v5, align 1
  %add9 = add nsw i32 %add7, %4
  %5 = load i32, ptr %v6, align 1
  %add11 = add nsw i32 %add9, %5
  %6 = load i32, ptr %v7, align 1
  %add13 = add nsw i32 %add11, %6
  ret i32 %add13

; CHECK-LABEL: callee2
; CHECK:     stb 9, 103(1)
; CHECK:     rldicl 10, 9, 56, 8
; CHECK:     sth 10, 101(1)
; CHECK:     rldicl 9, 9, 40, 24
; CHECK:     stw 9, 97(1)
; CHECK:     sth 8, 94(1)
; CHECK:     rldicl 8, 8, 48, 16
; CHECK:     stw 8, 90(1)
; CHECK:     stb 7, 87(1)
; CHECK:     rldicl 7, 7, 56, 8
; CHECK:     stw 7, 83(1)
; CHECK:     stb 5, 71(1)
; CHECK:     rldicl 5, 5, 56, 8
; CHECK:     sth 5, 69(1)
; CHECK:     stw 6, 76(1)
; CHECK:     sth 4, 62(1)
; CHECK:     stb 3, 55(1)
; CHECK-DAG: lha {{[0-9]+}}, 62(1)
; CHECK-DAG: lha {{[0-9]+}}, 69(1)
; CHECK-DAG: lbz {{[0-9]+}}, 55(1)
; CHECK-DAG: lwz {{[0-9]+}}, 76(1)
; CHECK-DAG: lwz {{[0-9]+}}, 83(1)
; CHECK-DAG: lwz {{[0-9]+}}, 90(1)
; CHECK-DAG: lwz {{[0-9]+}}, 97(1)
}
