; RUN: opt < %s -tbaa -gvn -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%structA = type { %structB }
%structB = type { i32*, %classT }
%classT = type { %classO, %classJ*, i8 }
%classO = type { i32 }
%classJ = type { i8 }
%classA = type { %classB }
%classB = type { i8 }
%classC = type { %classD, %structA }
%classD = type { %structA* }

; Function Attrs: ssp uwtable
define %structA** @test(%classA* %this, i32** %p1) #0 align 2 {
entry:
; CHECK-LABEL: @test
; CHECK: load i32*, i32** %p1, align 8, !tbaa
; CHECK: load i32*, i32** getelementptr (%classC, %classC* null, i32 0, i32 1, i32 0, i32 0), align 8, !tbaa
; CHECK: call void @callee
  %0 = load i32*, i32** %p1, align 8, !tbaa !1
  %1 = load i32*, i32** getelementptr (%classC, %classC* null, i32 0, i32 1, i32 0, i32 0), align 8, !tbaa !5
  call void @callee(i32* %0, i32* %1)
  unreachable
}

declare void @callee(i32*, i32*) #1

attributes #0 = { ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.4"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !2, i64 8}
!6 = !{!"_ZTSN12_GLOBAL__N_11RINS_1FIPi8TreeIterN1I1S1LENS_1KINS_1DIKS2_S3_EEEEE1GEPSD_EE", !7, i64 8}
!7 = !{!"_ZTSN12_GLOBAL__N_11FIPi8TreeIterN1I1S1LENS_1KINS_1DIKS1_S2_EEEEE1GE", !8, i64 0}
!8 = !{!"_ZTSN12_GLOBAL__N_11DIKPi8TreeIterEE", !2, i64 0, !9, i64 8}
!9 = !{!"_ZTS8TreeIter", !2, i64 8, !10, i64 16}
!10 = !{!"bool", !3, i64 0}
