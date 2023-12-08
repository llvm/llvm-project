; RUN: llc < %s -disable-block-placement | FileCheck %s
; RUN: llc < %s -disable-block-placement -disable-sparc-leaf-proc=0 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-n32:64-S128"
target triple = "sparcv9-sun-solaris"

; CHECK: varargsfunc
; 128 byte save ares + 1 alloca rounded up to 16 bytes alignment.
; CHECK: save %sp, -144, %sp
; Store the ... arguments to the argument array. The order is not important.
; CHECK: stx %i5, [%fp+2215]
; CHECK: stx %i4, [%fp+2207]
; CHECK: stx %i3, [%fp+2199]
; CHECK: stx %i2, [%fp+2191]
; Store the address of the ... args to %ap at %fp+BIAS+128-8
; add %fp, 2191, [[R:[gilo][0-7]]]
; stx [[R]], [%fp+2039]
define double @varargsfunc(i8* nocapture %fmt, double %sum, ...) {
entry:
  %ap = alloca i8*, align 4
  %ap1 = bitcast i8** %ap to i8*
  call void @llvm.va_start(i8* %ap1)
  br label %for.cond

for.cond:
  %fmt.addr.0 = phi i8* [ %fmt, %entry ], [ %incdec.ptr, %for.cond.backedge ]
  %sum.addr.0 = phi double [ %sum, %entry ], [ %sum.addr.0.be, %for.cond.backedge ]
  %incdec.ptr = getelementptr inbounds i8, i8* %fmt.addr.0, i64 1
  %0 = load i8, i8* %fmt.addr.0, align 1
  %conv = sext i8 %0 to i32
  switch i32 %conv, label %sw.default [
    i32 105, label %sw.bb
    i32 102, label %sw.bb3
  ]

; CHECK: sw.bb
; ldx [%fp+2039], %[[AP:[gilo][0-7]]]
; add %[[AP]], 4, %[[AP2:[gilo][0-7]]]
; stx %[[AP2]], [%fp+2039]
; ld [%[[AP]]]
sw.bb:
  %1 = va_arg i8** %ap, i32
  %conv2 = sitofp i32 %1 to double
  br label %for.cond.backedge

; CHECK: sw.bb3
; ldx [%fp+2039], %[[AP:[gilo][0-7]]]
; add %[[AP]], 8, %[[AP2:[gilo][0-7]]]
; stx %[[AP2]], [%fp+2039]
; ldd [%[[AP]]]
sw.bb3:
  %2 = va_arg i8** %ap, double
  br label %for.cond.backedge

for.cond.backedge:
  %.pn = phi double [ %2, %sw.bb3 ], [ %conv2, %sw.bb ]
  %sum.addr.0.be = fadd double %.pn, %sum.addr.0
  br label %for.cond

sw.default:
  ret double %sum.addr.0
}

declare void @llvm.va_start(i8*)

@.str = private unnamed_addr constant [4 x i8] c"abc\00", align 1

; CHECK: call_1d
; The fixed-arg double goes in %d2, the second goes in %o2.
; CHECK: sethi 1048576
; CHECK: , %f2
; CHECK: , %o2
define i32 @call_1d() #0 {
entry:
  %call = call double (i8*, double, ...) @varargsfunc(i8* undef, double 1.000000e+00, double 2.000000e+00)
  ret i32 1
}
