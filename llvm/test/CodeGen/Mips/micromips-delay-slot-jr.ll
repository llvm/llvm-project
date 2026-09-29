; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=static -O2 < %s | FileCheck %s

@main.L = internal unnamed_addr constant [3 x ptr] [ptr blockaddress(@main, %L1), ptr blockaddress(@main, %L2), ptr null], align 4
@str = private unnamed_addr constant [2 x i8] c"A\00"
@str2 = private unnamed_addr constant [2 x i8] c"B\00"

define i32 @main() #0 {
entry:
  br label %L1

L1:                                               ; preds = %entry, %L1
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L1 ]
  %puts = tail call i32 @puts(ptr @str)
  %inc = add i32 %i.0, 1
  %arrayidx = getelementptr inbounds [3 x ptr], ptr @main.L, i32 0, i32 %i.0
  %0 = load ptr, ptr %arrayidx, align 4, !tbaa !1
  indirectbr ptr %0, [label %L1, label %L2]

L2:                                               ; preds = %L1
  %puts2 = tail call i32 @puts(ptr @str2)
  ret i32 0
}

declare i32 @puts(ptr nocapture readonly) #1

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}

; CHECK:      jrc

%struct.foostruct = type { [3 x float] }
%struct.barstruct = type { %struct.foostruct, float }
@bar_ary = common global [4 x %struct.barstruct] zeroinitializer, align 4
define ptr @spooky(i32 signext %i) #0 {

  %safe = getelementptr inbounds [4 x %struct.barstruct], ptr @bar_ary, i32 0, i32 %i, i32 1
  store float 1.420000e+02, ptr %safe, align 4, !tbaa !1
  ret ptr %safe
}

; CHECK:      spooky:
; CHECK:      jrc $ra

