; RUN: opt -basicaa -tbaa -gvn -instcombine -S < %s | FileCheck %s --check-prefix=TBAA
; RUN: opt -tbaa -basicaa -gvn -instcombine -S < %s | FileCheck %s --check-prefix=BASICAA

; According to the TBAA metadata the load and store don't alias. However,
; according to the actual code, they do. The order of the alias analysis
; passes should determine which of these takes precedence.

target datalayout = "e-p:64:64:64"

; Test for simple MustAlias aliasing.

; TBAA:    @trouble
; TBAA:      ret i32 0
; BASICAA: @trouble
; BASICAA:   ret i32 1075000115
define i32 @trouble(i32* %x) nounwind {
entry:
  store i32 0, i32* %x, !tbaa !0
  %0 = bitcast i32* %x to float*
  store float 0x4002666660000000, float* %0, !tbaa !3
  %tmp3 = load i32* %x, !tbaa !0
  ret i32 %tmp3
}

; Test for PartialAlias aliasing. GVN doesn't yet eliminate the load
; in the BasicAA case.

; TBAA:    @offset
; TBAA:      ret i64 0
; BASICAA: @offset
; BASICAA:   ret i64 %tmp3
define i64 @offset(i64* %x) nounwind {
entry:
  store i64 0, i64* %x, !tbaa !4
  %0 = bitcast i64* %x to i8*
  %1 = getelementptr i8* %0, i64 1
  store i8 1, i8* %1, !tbaa !5
  %tmp3 = load i64* %x, !tbaa !4
  ret i64 %tmp3
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"simple"}
!3 = metadata !{metadata !"float", metadata !1}
!4 = metadata !{metadata !"long", metadata !1}
!5 = metadata !{metadata !"small", metadata !1}
