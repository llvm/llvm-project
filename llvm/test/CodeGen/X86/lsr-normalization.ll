; REQUIRES: asserts
; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s --check-prefix=ASM
; RUN: llc -debug -o /dev/null < %s -mtriple=x86_64-- 2>&1 | FileCheck %s --check-prefix=DBG
; rdar://8168938

; This testcase involves SCEV normalization with the exit value from
; one loop involved with the increment value for an addrec on another
; loop. The expression should be properly normalized and simplified,
; and require only a single division.

; DBG-NOT: DISCARDING (NORMALIZATION ISN'T INVERTIBLE)
; ASM: div
; ASM-NOT: div

%0 = type { %0*, %0* }

@0 = private constant [13 x i8] c"Result: %lu\0A\00" ; <[13 x i8]*> [#uses=1]
@1 = internal constant [5 x i8] c"Huh?\00"        ; <[5 x i8]*> [#uses=1]

define i32 @main(i32 %arg, i8** nocapture %arg1) nounwind {
bb:
  %tmp = alloca %0, align 8                       ; <%0*> [#uses=11]
  %tmp2 = bitcast %0* %tmp to i8*                 ; <i8*> [#uses=1]
  call void @llvm.memset.p0i8.i64(i8* align 8 %tmp2, i8 0, i64 16, i1 false) nounwind
  %tmp3 = getelementptr inbounds %0, %0* %tmp, i64 0, i32 0 ; <%0**> [#uses=3]
  store %0* %tmp, %0** %tmp3
  %tmp4 = getelementptr inbounds %0, %0* %tmp, i64 0, i32 1 ; <%0**> [#uses=1]
  store %0* %tmp, %0** %tmp4
  %tmp5 = call noalias i8* @_Znwm(i64 24) nounwind ; <i8*> [#uses=2]
  %tmp6 = getelementptr inbounds i8, i8* %tmp5, i64 16 ; <i8*> [#uses=2]
  %tmp7 = icmp eq i8* %tmp6, null                 ; <i1> [#uses=1]
  br i1 %tmp7, label %bb10, label %bb8

bb8:                                              ; preds = %bb
  %tmp9 = bitcast i8* %tmp6 to i32*               ; <i32*> [#uses=1]
  store i32 1, i32* %tmp9
  br label %bb10

bb10:                                             ; preds = %bb8, %bb
  %tmp11 = bitcast i8* %tmp5 to %0*               ; <%0*> [#uses=1]
  call void @_ZNSt15_List_node_base4hookEPS_(%0* %tmp11, %0* %tmp) nounwind
  %tmp12 = load %0*, %0** %tmp3                        ; <%0*> [#uses=3]
  %tmp13 = icmp eq %0* %tmp12, %tmp               ; <i1> [#uses=1]
  br i1 %tmp13, label %bb14, label %bb16

bb14:                                             ; preds = %bb10
  %tmp15 = call i32 @puts(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @1, i64 0, i64 0))
  br label %bb35

bb16:                                             ; preds = %bb16, %bb10
  %tmp17 = phi i64 [ %tmp22, %bb16 ], [ 0, %bb10 ] ; <i64> [#uses=1]
  %tmp18 = phi %0* [ %tmp20, %bb16 ], [ %tmp12, %bb10 ] ; <%0*> [#uses=1]
  %tmp19 = getelementptr inbounds %0, %0* %tmp18, i64 0, i32 0 ; <%0**> [#uses=1]
  %tmp20 = load %0*, %0** %tmp19                       ; <%0*> [#uses=2]
  %tmp21 = icmp eq %0* %tmp20, %tmp               ; <i1> [#uses=1]
  %tmp22 = add i64 %tmp17, 1                      ; <i64> [#uses=2]
  br i1 %tmp21, label %bb23, label %bb16

bb23:                                             ; preds = %bb16
  %tmp24 = udiv i64 100, %tmp22                   ; <i64> [#uses=1]
  br label %bb25

bb25:                                             ; preds = %bb25, %bb23
  %tmp26 = phi i64 [ %tmp31, %bb25 ], [ 0, %bb23 ] ; <i64> [#uses=1]
  %tmp27 = phi %0* [ %tmp29, %bb25 ], [ %tmp12, %bb23 ] ; <%0*> [#uses=1]
  %tmp28 = getelementptr inbounds %0, %0* %tmp27, i64 0, i32 0 ; <%0**> [#uses=1]
  %tmp29 = load %0*, %0** %tmp28                       ; <%0*> [#uses=2]
  %tmp30 = icmp eq %0* %tmp29, %tmp               ; <i1> [#uses=1]
  %tmp31 = add i64 %tmp26, 1                      ; <i64> [#uses=2]
  br i1 %tmp30, label %bb32, label %bb25

bb32:                                             ; preds = %bb25
  %tmp33 = mul i64 %tmp31, %tmp24                 ; <i64> [#uses=1]
  %tmp34 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @0, i64 0, i64 0), i64 %tmp33) nounwind
  br label %bb35

bb35:                                             ; preds = %bb32, %bb14
  %tmp36 = load %0*, %0** %tmp3                        ; <%0*> [#uses=2]
  %tmp37 = icmp eq %0* %tmp36, %tmp               ; <i1> [#uses=1]
  br i1 %tmp37, label %bb44, label %bb38

bb38:                                             ; preds = %bb38, %bb35
  %tmp39 = phi %0* [ %tmp41, %bb38 ], [ %tmp36, %bb35 ] ; <%0*> [#uses=2]
  %tmp40 = getelementptr inbounds %0, %0* %tmp39, i64 0, i32 0 ; <%0**> [#uses=1]
  %tmp41 = load %0*, %0** %tmp40                       ; <%0*> [#uses=2]
  %tmp42 = bitcast %0* %tmp39 to i8*              ; <i8*> [#uses=1]
  call void @_ZdlPv(i8* %tmp42) nounwind
  %tmp43 = icmp eq %0* %tmp41, %tmp               ; <i1> [#uses=1]
  br i1 %tmp43, label %bb44, label %bb38

bb44:                                             ; preds = %bb38, %bb35
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @_ZNSt15_List_node_base4hookEPS_(%0*, %0*)

declare noalias i8* @_Znwm(i64)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind

declare void @_ZdlPv(i8*) nounwind

declare i32 @puts(i8* nocapture) nounwind
