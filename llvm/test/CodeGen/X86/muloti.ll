; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
%0 = type { i64, i64 }
%1 = type { i128, i1 }

define %0 @x(i64 %a.coerce0, i64 %a.coerce1, i64 %b.coerce0, i64 %b.coerce1) nounwind uwtable ssp {
; CHECK: x
entry:
  %tmp16 = zext i64 %a.coerce0 to i128
  %tmp11 = zext i64 %a.coerce1 to i128
  %tmp12 = shl nuw i128 %tmp11, 64
  %ins14 = or i128 %tmp12, %tmp16
  %tmp6 = zext i64 %b.coerce0 to i128
  %tmp3 = zext i64 %b.coerce1 to i128
  %tmp4 = shl nuw i128 %tmp3, 64
  %ins = or i128 %tmp4, %tmp6
  %0 = tail call %1 @llvm.smul.with.overflow.i128(i128 %ins14, i128 %ins)
; CHECK: callq   ___muloti4
  %1 = extractvalue %1 %0, 0
  %2 = extractvalue %1 %0, 1
  br i1 %2, label %overflow, label %nooverflow

overflow:                                         ; preds = %entry
  tail call void @llvm.trap()
  unreachable

nooverflow:                                       ; preds = %entry
  %tmp20 = trunc i128 %1 to i64
  %tmp21 = insertvalue %0 undef, i64 %tmp20, 0
  %tmp22 = lshr i128 %1, 64
  %tmp23 = trunc i128 %tmp22 to i64
  %tmp24 = insertvalue %0 %tmp21, i64 %tmp23, 1
  ret %0 %tmp24
}

define %0 @foo(i64 %a.coerce0, i64 %a.coerce1, i64 %b.coerce0, i64 %b.coerce1) nounwind uwtable ssp {
entry:
; CHECK: foo
  %retval = alloca i128, align 16
  %coerce = alloca i128, align 16
  %a.addr = alloca i128, align 16
  %coerce1 = alloca i128, align 16
  %b.addr = alloca i128, align 16
  %0 = bitcast i128* %coerce to %0*
  %1 = getelementptr %0* %0, i32 0, i32 0
  store i64 %a.coerce0, i64* %1
  %2 = getelementptr %0* %0, i32 0, i32 1
  store i64 %a.coerce1, i64* %2
  %a = load i128* %coerce, align 16
  store i128 %a, i128* %a.addr, align 16
  %3 = bitcast i128* %coerce1 to %0*
  %4 = getelementptr %0* %3, i32 0, i32 0
  store i64 %b.coerce0, i64* %4
  %5 = getelementptr %0* %3, i32 0, i32 1
  store i64 %b.coerce1, i64* %5
  %b = load i128* %coerce1, align 16
  store i128 %b, i128* %b.addr, align 16
  %tmp = load i128* %a.addr, align 16
  %tmp2 = load i128* %b.addr, align 16
  %6 = call %1 @llvm.umul.with.overflow.i128(i128 %tmp, i128 %tmp2)
; CHECK: cmov
; CHECK: divti3
  %7 = extractvalue %1 %6, 0
  %8 = extractvalue %1 %6, 1
  br i1 %8, label %overflow, label %nooverflow

overflow:                                         ; preds = %entry
  call void @llvm.trap()
  unreachable

nooverflow:                                       ; preds = %entry
  store i128 %7, i128* %retval
  %9 = bitcast i128* %retval to %0*
  %10 = load %0* %9, align 1
  ret %0 %10
}

declare %1 @llvm.umul.with.overflow.i128(i128, i128) nounwind readnone

declare %1 @llvm.smul.with.overflow.i128(i128, i128) nounwind readnone

declare void @llvm.trap() nounwind
