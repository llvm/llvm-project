; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=static -O2 < %s | FileCheck %s

@main.L = internal unnamed_addr constant [3 x i8*] [i8* blockaddress(@main, %L1), i8* blockaddress(@main, %L2), i8* null], align 4
@str = private unnamed_addr constant [2 x i8] c"A\00"
@str2 = private unnamed_addr constant [2 x i8] c"B\00"

define i32 @main() #0 {
entry:
  br label %L1

L1:                                               ; preds = %entry, %L1
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L1 ]
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str, i32 0, i32 0))
  %inc = add i32 %i.0, 1
  %arrayidx = getelementptr inbounds [3 x i8*], [3 x i8*]* @main.L, i32 0, i32 %i.0
  %0 = load i8*, i8** %arrayidx, align 4, !tbaa !1
  indirectbr i8* %0, [label %L1, label %L2]

L2:                                               ; preds = %L1
  %puts2 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str2, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture readonly) #1

!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}

; CHECK:      jrc

%struct.foostruct = type { [3 x float] }
%struct.barstruct = type { %struct.foostruct, float }
@bar_ary = common global [4 x %struct.barstruct] zeroinitializer, align 4
define float* @spooky(i32 signext %i) #0 {

  %safe = getelementptr inbounds [4 x %struct.barstruct], [4 x %struct.barstruct]* @bar_ary, i32 0, i32 %i, i32 1
  store float 1.420000e+02, float* %safe, align 4, !tbaa !1
  ret float* %safe
}

; CHECK:      spooky:
; CHECK:      jrc $ra

