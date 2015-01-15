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
  tail call void @llvm.dbg.value(metadata i32* null, i64 0, metadata !11, metadata !{!"0x102"}), !dbg !28
  %0 = load i64* @a, align 8, !dbg !29
  %xor = xor i64 %0, %e.1.ph, !dbg !29
  %conv3 = trunc i64 %xor to i32, !dbg !29
  tail call void @llvm.dbg.value(metadata i32 %conv3, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !29
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
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!33}

!0 = !{!"0x11\0012\00clang version 3.4 (trunk 182024) (llvm/trunk 182023)\001\00\000\00\000", !1, !2, !2, !3, !15, !2} ; [ DW_TAG_compile_unit ] [/d/b/pr16110.c] [DW_LANG_C99]
!1 = !{!"pr16110.c", !"/d/b"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00pr16110\00pr16110\00\007\000\001\000\006\000\001\007", !1, !5, !6, null, i32 ()* @pr16110, null, null, !9} ; [ DW_TAG_subprogram ] [line 7] [def] [pr16110]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/d/b/pr16110.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!10, !11}
!10 = !{!"0x100\00e\008\000", !4, !5, !8} ; [ DW_TAG_auto_variable ] [e] [line 8]
!11 = !{!"0x100\00f\0013\000", !12, !5, !14} ; [ DW_TAG_auto_variable ] [f] [line 13]
!12 = !{!"0xb\0012\000\002", !1, !13} ; [ DW_TAG_lexical_block ] [/d/b/pr16110.c]
!13 = !{!"0xb\0012\000\001", !1, !4} ; [ DW_TAG_lexical_block ] [/d/b/pr16110.c]
!14 = !{!"0xf\00\000\0032\0032\000\000", null, null, !8} ; [ DW_TAG_pointer_type ] [line 0, size 32, align 32, offset 0] [from int]
!15 = !{!16, !18, !19, !20}
!16 = !{!"0x34\00a\00a\00\001\000\001", null, !5, !17, i64* @a, null} ; [ DW_TAG_variable ] [a] [line 1] [def]
!17 = !{!"0x24\00long long int\000\0064\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [long long int] [line 0, size 64, align 32, offset 0, enc DW_ATE_signed]
!18 = !{!"0x34\00b\00b\00\002\000\001", null, !5, !8, i32* @b, null} ; [ DW_TAG_variable ] [b] [line 2] [def]
!19 = !{!"0x34\00c\00c\00\003\000\001", null, !5, !8, i32* @c, null} ; [ DW_TAG_variable ] [c] [line 3] [def]
!20 = !{!"0x34\00d\00d\00\004\000\001", null, !5, !8, i32* @d, null} ; [ DW_TAG_variable ] [d] [line 4] [def]
!21 = !MDLocation(line: 10, scope: !22)
!22 = !{!"0xb\0010\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [/d/b/pr16110.c]
!26 = !MDLocation(line: 12, scope: !13)
!27 = !{i32* null}
!28 = !MDLocation(line: 13, scope: !12)
!29 = !MDLocation(line: 14, scope: !12)
!31 = !MDLocation(line: 16, scope: !4)
!32 = !MDLocation(line: 18, scope: !4)
!33 = !{i32 1, !"Debug Info Version", i32 2}
