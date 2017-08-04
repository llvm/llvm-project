; RUN: not llc < %s -march=bpfel | FileCheck %s

@foo_printf.fmt = private unnamed_addr constant [9 x i8] c"hello  \0A\00", align 1

; Function Attrs: nounwind readnone uwtable
define i32 @foo_int(i32 %a, i32 %b) #0 {
  %1 = add nsw i32 %b, %a
  ret i32 %1
; CHECK-LABEL: foo_int:
; CHECK: r2 += r1
}

; Function Attrs: nounwind readnone uwtable
define signext i8 @foo_char(i8 signext %a, i8 signext %b) #0 {
  %1 = add i8 %b, %a
  ret i8 %1
; CHECK-LABEL: foo_char:
; CHECK: r2 += r1
; CHECK: r2 <<= 56
; CHECK: r2 s>>= 56
}

; Function Attrs: nounwind readnone uwtable
define i64 @foo_ll(i64 %a, i64 %b, i64 %c) #0 {
  %1 = add nsw i64 %b, %a
  %2 = sub i64 %1, %c
  ret i64 %2
; CHECK-LABEL: foo_ll:
; CHECK: r2 += r1
; CHECK: r2 -= r3
; CHECK: r0 = r2
}

; Function Attrs: nounwind uwtable
define void @foo_call2(i32 %a, i32 %b) #1 {
  %1 = trunc i32 %b to i8
  tail call void @foo_2arg(i8 signext %1, i32 %a) #3
  ret void
; CHECK-LABEL: foo_call2:
; CHECK: r2 <<= 56
; CHECK: r2 s>>= 56
; CHECK: r1 = r2
}

declare void @foo_2arg(i8 signext, i32) #2

; Function Attrs: nounwind uwtable
define i32 @foo_call5(i8 signext %a, i16 signext %b, i32 %c, i64 %d) #1 {
  %1 = tail call i32 @bar(i8 signext %a, i16 signext %b, i32 %c, i64 %d) #3
  ret i32 0
; CHECK-LABEL: foo_call5:
; CHECK: call bar
}

declare i32 @bar(i8 signext, i16 signext, i32, i64) #2

; Function Attrs: nounwind readnone uwtable
define signext i8 @foo_cmp(i8 signext %a, i8 signext %b) #0 {
  %1 = icmp slt i8 %a, %b
  %a.b = select i1 %1, i8 %a, i8 %b
  ret i8 %a.b
; CHECK-LABEL: foo_cmp:
; CHECK: if r2 s> r1
}

; Function Attrs: nounwind readnone uwtable
define i32 @foo_muldiv(i8 signext %a, i16 signext %b, i32 %c, i64 %d) #0 {
  %1 = icmp eq i8 %a, 0
  br i1 %1, label %5, label %2

; <label>:2                                       ; preds = %0
  %3 = sext i16 %b to i32
  %4 = mul nsw i32 %3, %c
  br label %8

; <label>:5                                       ; preds = %0
  %6 = trunc i64 %d to i32
  %7 = udiv i32 %6, %c
  br label %8

; <label>:8                                       ; preds = %5, %2
  %.0 = phi i32 [ %4, %2 ], [ %7, %5 ]
  ret i32 %.0
; CHECK-LABEL: foo_muldiv:
; CHECK: r2 *= r3
}

; Function Attrs: nounwind uwtable
define i32 @foo_optimized() #1 {
  %1 = tail call i32 @manyarg(i32 1, i32 2, i32 3, i32 4, i32 5) #3
  ret i32 %1
; CHECK-LABEL: foo_optimized:
; CHECK: r1 = 1
; CHECK: r2 = 2
; CHECK: r3 = 3
; CHECK: r4 = 4
; CHECK: r5 = 5
}

declare i32 @manyarg(i32, i32, i32, i32, i32) #2

; Function Attrs: nounwind uwtable
define void @foo_printf() #1 {
  %fmt = alloca [9 x i8], align 1
  %1 = getelementptr inbounds [9 x i8], [9 x i8]* %fmt, i64 0, i64 0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %1, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @foo_printf.fmt, i64 0, i64 0), i64 9, i32 1, i1 false)
; CHECK-LABEL: foo_printf:
; CHECK: r1 = 729618802566522216ll
  %2 = call i32 (i8*, ...) @printf(i8* %1) #3
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) #3

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture, ...) #4
