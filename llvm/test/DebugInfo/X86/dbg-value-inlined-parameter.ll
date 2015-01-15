; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s
; RUN: llc -mtriple=x86_64-linux-gnu < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=LINUX %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj -regalloc=basic \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "nums"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "nums"

;CHECK: DW_TAG_inlined_subroutine
;CHECK-NEXT: DW_AT_abstract_origin {{.*}} "foo"
;CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]
;CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]
;CHECK-NEXT: DW_AT_call_file
;CHECK-NEXT: DW_AT_call_line

;CHECK: DW_TAG_formal_parameter
;FIXME: Linux shouldn't drop this parameter either...
;CHECK-NOT: DW_TAG
;DARWIN:   DW_AT_abstract_origin {{.*}} "sp"
;DARWIN: DW_TAG_formal_parameter
;CHECK: DW_AT_abstract_origin {{.*}} "nums"
;CHECK-NOT: DW_TAG_formal_parameter

%struct.S1 = type { float*, i32 }

@p = common global %struct.S1 zeroinitializer, align 8

define i32 @foo(%struct.S1* nocapture %sp, i32 %nums) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata %struct.S1* %sp, i64 0, metadata !9, metadata !{!"0x102"}), !dbg !20
  tail call void @llvm.dbg.value(metadata i32 %nums, i64 0, metadata !18, metadata !{!"0x102"}), !dbg !21
  %tmp2 = getelementptr inbounds %struct.S1* %sp, i64 0, i32 1, !dbg !22
  store i32 %nums, i32* %tmp2, align 4, !dbg !22
  %call = tail call float* @bar(i32 %nums) nounwind optsize, !dbg !27
  %tmp5 = getelementptr inbounds %struct.S1* %sp, i64 0, i32 0, !dbg !27
  store float* %call, float** %tmp5, align 8, !dbg !27
  %cmp = icmp ne float* %call, null, !dbg !29
  %cond = zext i1 %cmp to i32, !dbg !29
  ret i32 %cond, !dbg !29
}

declare float* @bar(i32) optsize

define void @foobar() nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata %struct.S1* @p, i64 0, metadata !9, metadata !{!"0x102"}) nounwind, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !18, metadata !{!"0x102"}) nounwind, !dbg !35
  store i32 1, i32* getelementptr inbounds (%struct.S1* @p, i64 0, i32 1), align 8, !dbg !36
  %call.i = tail call float* @bar(i32 1) nounwind optsize, !dbg !37
  store float* %call.i, float** getelementptr inbounds (%struct.S1* @p, i64 0, i32 0), align 8, !dbg !37
  ret void, !dbg !38
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!43}

!0 = !{!"0x2e\00foo\00foo\00\008\000\001\000\006\00256\001\008", !1, !1, !3, null, i32 (%struct.S1*, i32)* @foo, null, null, !41} ; [ DW_TAG_subprogram ] [line 8] [def] [foo]
!1 = !{!"0x29", !42} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 2.9 (trunk 125693)\001\00\000\00\001", !42, !8, !8, !39, !40,  !44} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !42, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00foobar\00foobar\00\0015\000\001\000\006\000\001\000", !1, !1, !7, null, void ()* @foobar, null, null, null} ; [ DW_TAG_subprogram ] [line 15] [def] [scope 0] [foobar]
!7 = !{!"0x15\00\000\000\000\000\000\000", !42, !1, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null}
!9 = !{!"0x101\00sp\0016777223\000", !0, !1, !10, !32} ; [ DW_TAG_arg_variable ]
!10 = !{!"0xf\00\000\0064\0064\000\000", null, !2, !11} ; [ DW_TAG_pointer_type ]
!11 = !{!"0x16\00S1\004\000\000\000\000", !42, !2, !12} ; [ DW_TAG_typedef ]
!12 = !{!"0x13\00S1\001\00128\0064\000\000\000", !42, !2, null, !13, null, null, null} ; [ DW_TAG_structure_type ] [S1] [line 1, size 128, align 64, offset 0] [def] [from ]
!13 = !{!14, !17}
!14 = !{!"0xd\00m\002\0064\0064\000\000", !42, !1, !15} ; [ DW_TAG_member ]
!15 = !{!"0xf\00\000\0064\0064\000\000", null, !2, !16} ; [ DW_TAG_pointer_type ]
!16 = !{!"0x24\00float\000\0032\0032\000\000\004", null, !2} ; [ DW_TAG_base_type ]
!17 = !{!"0xd\00nums\003\0032\0032\0064\000", !42, !1, !5} ; [ DW_TAG_member ]
!18 = !{!"0x101\00nums\0033554439\000", !0, !1, !5, !32} ; [ DW_TAG_arg_variable ]
!19 = !{!"0x34\00p\00p\00\0014\000\001", !2, !1, !11, %struct.S1* @p, null} ; [ DW_TAG_variable ]
!20 = !MDLocation(line: 7, column: 13, scope: !0)
!21 = !MDLocation(line: 7, column: 21, scope: !0)
!22 = !MDLocation(line: 9, column: 3, scope: !23)
!23 = !{!"0xb\008\001\000", !1, !0} ; [ DW_TAG_lexical_block ]
!27 = !MDLocation(line: 10, column: 3, scope: !23)
!29 = !MDLocation(line: 11, column: 3, scope: !23)
!30 = !{%struct.S1* @p}
!31 = !MDLocation(line: 7, column: 13, scope: !0, inlinedAt: !32)
!32 = !MDLocation(line: 16, column: 3, scope: !33)
!33 = !{!"0xb\0015\0015\001", !1, !6} ; [ DW_TAG_lexical_block ]
!34 = !{i32 1}
!35 = !MDLocation(line: 7, column: 21, scope: !0, inlinedAt: !32)
!36 = !MDLocation(line: 9, column: 3, scope: !23, inlinedAt: !32)
!37 = !MDLocation(line: 10, column: 3, scope: !23, inlinedAt: !32)
!38 = !MDLocation(line: 17, column: 1, scope: !33)
!39 = !{!0, !6}
!40 = !{!19}
!41 = !{!9, !18}
!42 = !{!"nm2.c", !"/private/tmp"}
!43 = !{i32 1, !"Debug Info Version", i32 2}
!44 = !{}
