; RUN: llc < %s -verify-machineinstrs
; PR16110
;
; This test case contains a value that is split into two connected components
; by rematerialization during coalescing. It also contains a DBG_VALUE
; instruction which must be updated during
; ConnectedVNInfoEqClasses::Distribute().

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios3.0.0"

@c = common global i32 0, align 4
@b = common global i32 0, align 4
@a = common global i64 0, align 8
@d = common global i32 0, align 4

; Function Attrs: nounwind ssp
define i32 @pr16110() #0 {
for.cond1.preheader:
  store i32 0, i32* @c, align 4, !dbg !21
  br label %for.cond1.outer, !dbg !26

for.cond1:                                        ; preds = %for.end9, %for.cond1.outer
  %storemerge11 = phi i32 [ 0, %for.end9 ], [ %storemerge11.ph, %for.cond1.outer ]
  %cmp = icmp slt i32 %storemerge11, 1, !dbg !26
  br i1 %cmp, label %for.body2, label %for.end9, !dbg !26

for.body2:                                        ; preds = %for.cond1
  store i32 %storemerge11, i32* @b, align 4, !dbg !26
  tail call void @llvm.dbg.value(metadata !27, i64 0, metadata !11), !dbg !28
  %0 = load i64* @a, align 8, !dbg !29
  %xor = xor i64 %0, %e.1.ph, !dbg !29
  %conv3 = trunc i64 %xor to i32, !dbg !29
  tail call void @llvm.dbg.value(metadata !{i32 %conv3}, i64 0, metadata !10), !dbg !29
  %tobool4 = icmp eq i32 %conv3, 0, !dbg !29
  br i1 %tobool4, label %land.end, label %land.rhs, !dbg !29

land.rhs:                                         ; preds = %for.body2
  %call = tail call i32 bitcast (i32 (...)* @fn3 to i32 ()*)() #3, !dbg !29
  %tobool5 = icmp ne i32 %call, 0, !dbg !29
  br label %land.end

land.end:                                         ; preds = %land.rhs, %for.body2
  %1 = phi i1 [ false, %for.body2 ], [ %tobool5, %land.rhs ]
  %land.ext = zext i1 %1 to i32
  %call6 = tail call i32 bitcast (i32 (...)* @fn2 to i32 (i32, i32*)*)(i32 %land.ext, i32* null) #3
  %2 = load i32* @b, align 4, !dbg !26
  %inc8 = add nsw i32 %2, 1, !dbg !26
  %phitmp = and i64 %xor, 4294967295, !dbg !26
  br label %for.cond1.outer, !dbg !26

for.cond1.outer:                                  ; preds = %land.end, %for.cond1.preheader
  %storemerge11.ph = phi i32 [ %inc8, %land.end ], [ 0, %for.cond1.preheader ]
  %e.1.ph = phi i64 [ %phitmp, %land.end ], [ 0, %for.cond1.preheader ]
  %3 = load i32* @d, align 4, !dbg !31
  %tobool10 = icmp eq i32 %3, 0, !dbg !31
  br label %for.cond1

for.end9:                                         ; preds = %for.cond1
  br i1 %tobool10, label %if.end, label %for.cond1, !dbg !31

if.end:                                           ; preds = %for.end9
  store i32 %storemerge11, i32* @b, align 4, !dbg !26
  ret i32 0, !dbg !32
}

declare i32 @fn2(...) #1

declare i32 @fn3(...) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #2

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 182024) (llvm/trunk 182023)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !15, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/d/b/pr16110.c] [DW_LANG_C99]
!1 = metadata !{metadata !"pr16110.c", metadata !"/d/b"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"pr16110", metadata !"pr16110", metadata !"", i32 7, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 true, i32 ()* @pr16110, null, null, metadata !9, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [pr16110]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/d/b/pr16110.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10, metadata !11}
!10 = metadata !{i32 786688, metadata !4, metadata !"e", metadata !5, i32 8, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [e] [line 8]
!11 = metadata !{i32 786688, metadata !12, metadata !"f", metadata !5, i32 13, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [f] [line 13]
!12 = metadata !{i32 786443, metadata !1, metadata !13, i32 12, i32 0, i32 2} ; [ DW_TAG_lexical_block ] [/d/b/pr16110.c]
!13 = metadata !{i32 786443, metadata !1, metadata !4, i32 12, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/d/b/pr16110.c]
!14 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from int]
!15 = metadata !{metadata !16, metadata !18, metadata !19, metadata !20}
!16 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !5, i32 1, metadata !17, i32 0, i32 1, i64* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!17 = metadata !{i32 786468, null, null, metadata !"long long int", i32 0, i64 64, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [long long int] [line 0, size 64, align 32, offset 0, enc DW_ATE_signed]
!18 = metadata !{i32 786484, i32 0, null, metadata !"b", metadata !"b", metadata !"", metadata !5, i32 2, metadata !8, i32 0, i32 1, i32* @b, null} ; [ DW_TAG_variable ] [b] [line 2] [def]
!19 = metadata !{i32 786484, i32 0, null, metadata !"c", metadata !"c", metadata !"", metadata !5, i32 3, metadata !8, i32 0, i32 1, i32* @c, null} ; [ DW_TAG_variable ] [c] [line 3] [def]
!20 = metadata !{i32 786484, i32 0, null, metadata !"d", metadata !"d", metadata !"", metadata !5, i32 4, metadata !8, i32 0, i32 1, i32* @d, null} ; [ DW_TAG_variable ] [d] [line 4] [def]
!21 = metadata !{i32 10, i32 0, metadata !22, null}
!22 = metadata !{i32 786443, metadata !1, metadata !4, i32 10, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/d/b/pr16110.c]
!26 = metadata !{i32 12, i32 0, metadata !13, null}
!27 = metadata !{i32* null}
!28 = metadata !{i32 13, i32 0, metadata !12, null}
!29 = metadata !{i32 14, i32 0, metadata !12, null}
!31 = metadata !{i32 16, i32 0, metadata !4, null}
!32 = metadata !{i32 18, i32 0, metadata !4, null}
!33 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
