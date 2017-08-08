; RUN: llc -o - %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@a = external global i32, align 4
@d = external global i32*, align 4
@k = external global i32**, align 4
@j = external global i32***, align 4
@h = external global i32, align 4
@c = external global i32, align 4
@i = external global i32, align 4
@b = external global i32, align 4
@f = external global i64, align 8
@e = external global i64, align 8
@g = external global i32, align 4

; Function Attrs: norecurse nounwind optsize readnone
declare i32 @fn1(i32 returned) #0


; CHECK-LABEL: fn2
; CHECK: calll putchar
; CHECK: addl $1,
; CHECK: adcl $0,
; Function Attrs: nounwind optsize
define void @fn2() #1 {
entry:
  %putchar = tail call i32 @putchar(i32 48)
  %0 = load volatile i32, i32* @h, align 4
  %1 = load i32, i32* @c, align 4, !tbaa !2
  %2 = load i32***, i32**** @j, align 4
  %3 = load i32**, i32*** %2, align 4
  %4 = load i32*, i32** %3, align 4
  %5 = load i32, i32* %4, align 4
  %cmp = icmp sgt i32 %1, %5
  %conv = zext i1 %cmp to i32
  %6 = load i32, i32* @i, align 4
  %cmp1 = icmp sgt i32 %6, %conv
  %conv2 = zext i1 %cmp1 to i32
  store i32 %conv2, i32* @b, align 4
  %cmp3 = icmp sgt i32 %0, %conv2
  %conv4 = zext i1 %cmp3 to i32
  %7 = load i32, i32* @a, align 4
  %or = xor i32 %7, %conv4
  store i32 %or, i32* @a, align 4
  %8 = load i32*, i32** @d, align 4
  %9 = load i32, i32* %8, align 4
  %conv6 = sext i32 %9 to i64
  %10 = load i64, i64* @e, align 8
  %and = and i64 %10, %conv6
  store i64 %and, i64* @e, align 8
  %11 = load i32, i32* @g, align 4
  %dec = add nsw i32 %11, -1
  store i32 %dec, i32* @g, align 4
  %12 = load i64, i64* @f, align 8
  %inc = add nsw i64 %12, 1
  store i64 %inc, i64* @f, align 8
  ret void
}

; Function Attrs: nounwind optsize
declare i32 @main() #1

; Function Attrs: nounwind
declare i32 @putchar(i32) #2

attributes #0 = { optsize readnone }
attributes #1 = { optsize }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"NumRegisterParameters", i32 0}
!1 = !{!"clang version 5.0.0 (trunk 300074) (llvm/trunk 300078)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !4, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"long long", !4, i64 0}
