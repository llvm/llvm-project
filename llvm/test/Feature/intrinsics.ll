; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: FileCheck %s < %t1.ll

declare i1 @llvm.isunordered.f32(float, float)

declare i1 @llvm.isunordered.f64(double, double)


declare i8 @llvm.ctpop.i8(i8)

declare i16 @llvm.ctpop.i16(i16)

declare i32 @llvm.ctpop.i32(i32)

declare i64 @llvm.ctpop.i64(i64)

declare i8 @llvm.cttz.i8(i8, i1)

declare i16 @llvm.cttz.i16(i16, i1)

declare i32 @llvm.cttz.i32(i32, i1)

declare i64 @llvm.cttz.i64(i64, i1)

declare i8 @llvm.ctlz.i8(i8, i1)

declare i16 @llvm.ctlz.i16(i16, i1)

declare i32 @llvm.ctlz.i32(i32, i1)

declare i64 @llvm.ctlz.i64(i64, i1)

declare float @llvm.sqrt.f32(float)

declare double @llvm.sqrt.f64(double)

; Test llvm intrinsics
;
define void @libm() {
        fcmp uno float 1.000000e+00, 2.000000e+00               ; <i1>:1 [#uses=0]
        fcmp uno double 3.000000e+00, 4.000000e+00              ; <i1>:2 [#uses=0]
        call float @llvm.sqrt.f32( float 5.000000e+00 )         ; <float>:3 [#uses=0]
        call double @llvm.sqrt.f64( double 6.000000e+00 )               ; <double>:4 [#uses=0]
        call i8  @llvm.ctpop.i8( i8 10 )                ; <i32>:5 [#uses=0]
        call i16 @llvm.ctpop.i16( i16 11 )              ; <i32>:6 [#uses=0]
        call i32 @llvm.ctpop.i32( i32 12 )              ; <i32>:7 [#uses=0]
        call i64 @llvm.ctpop.i64( i64 13 )              ; <i32>:8 [#uses=0]
        call i8  @llvm.ctlz.i8( i8 14, i1 true )         ; <i32>:9 [#uses=0]
        call i16 @llvm.ctlz.i16( i16 15, i1 true )               ; <i32>:10 [#uses=0]
        call i32 @llvm.ctlz.i32( i32 16, i1 true )               ; <i32>:11 [#uses=0]
        call i64 @llvm.ctlz.i64( i64 17, i1 true )               ; <i32>:12 [#uses=0]
        call i8  @llvm.cttz.i8( i8 18, i1 true )         ; <i32>:13 [#uses=0]
        call i16 @llvm.cttz.i16( i16 19, i1 true )               ; <i32>:14 [#uses=0]
        call i32 @llvm.cttz.i32( i32 20, i1 true )               ; <i32>:15 [#uses=0]
        call i64 @llvm.cttz.i64( i64 21, i1 true )               ; <i32>:16 [#uses=0]
        ret void
}

; FIXME: test ALL the intrinsics in this file.

; rdar://11542750
; CHECK: declare void @llvm.trap() noreturn nounwind
declare void @llvm.trap()

define void @trap() {
  call void @llvm.trap()
  ret void
}
