; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      long int b;
;    } Inner;
;
;    typedef struct {
;      Inner inner[2];
;    } Outer;
;
;    int foo(Outer outer) {
;      Inner i1 = outer.inner[1];
;      return i1.a;
;    }
;
;
; CHECK: DW_TAG_variable [4]
;                                                  rax, piece 0x00000004
; CHECK-NEXT: DW_AT_location [DW_FORM_block1]{{.*}}50 93 04
; CHECK-NEXT:  DW_AT_name {{.*}}"i1"
;
; ModuleID = '/Volumes/Data/llvm/test/DebugInfo/X86/sroasplit-1.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i64 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(%struct.Outer* byval align 8 %outer) #0 {
entry:
  call void @llvm.dbg.declare(metadata %struct.Outer* %outer, metadata !25, metadata !{!"0x102"}), !dbg !26
  %i1.sroa.0.0..sroa_idx = getelementptr inbounds %struct.Outer* %outer, i64 0, i32 0, i64 1, i32 0, !dbg !27
  %i1.sroa.0.0.copyload = load i32* %i1.sroa.0.0..sroa_idx, align 8, !dbg !27
  call void @llvm.dbg.value(metadata i32 %i1.sroa.0.0.copyload, i64 0, metadata !28, metadata !29), !dbg !27
  %i1.sroa.2.0..sroa_raw_cast = bitcast %struct.Outer* %outer to i8*, !dbg !27
  %i1.sroa.2.0..sroa_raw_idx = getelementptr inbounds i8* %i1.sroa.2.0..sroa_raw_cast, i64 20, !dbg !27
  ret i32 %i1.sroa.0.0.copyload, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = !{!"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/sroasplit-1.c] [DW_LANG_C99]
!1 = !{!"sroasplit-1.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\0010\000\001\000\006\00256\000\0010", !1, !5, !6, null, i32 (%struct.Outer*)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/sroasplit-1.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !9}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x16\00Outer\008\000\000\000\000", !1, null, !10} ; [ DW_TAG_typedef ] [Outer] [line 8, size 0, align 0, offset 0] [from ]
!10 = !{!"0x13\00\006\00256\0064\000\000\000", !1, null, null, !11, null, null, null} ; [ DW_TAG_structure_type ] [line 6, size 256, align 64, offset 0] [def] [from ]
!11 = !{!12}
!12 = !{!"0xd\00inner\007\00256\0064\000\000", !1, !10, !13} ; [ DW_TAG_member ] [inner] [line 7, size 256, align 64, offset 0] [from ]
!13 = !{!"0x1\00\000\00256\0064\000\000", null, null, !14, !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 256, align 64, offset 0] [from Inner]
!14 = !{!"0x16\00Inner\004\000\000\000\000", !1, null, !15} ; [ DW_TAG_typedef ] [Inner] [line 4, size 0, align 0, offset 0] [from ]
!15 = !{!"0x13\00\001\00128\0064\000\000\000", !1, null, null, !16, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 128, align 64, offset 0] [def] [from ]
!16 = !{!17, !18}
!17 = !{!"0xd\00a\002\0032\0032\000\000", !1, !15, !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!18 = !{!"0xd\00b\003\0064\0064\0064\000", !1, !15, !19} ; [ DW_TAG_member ] [b] [line 3, size 64, align 64, offset 64] [from long int]
!19 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!20 = !{!21}
!21 = !{!"0x21\000\002"}        ; [ DW_TAG_subrange_type ] [0, 1]
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 1, !"Debug Info Version", i32 2}
!24 = !{!"clang version 3.5.0 "}
!25 = !{!"0x101\00outer\0016777226\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [outer] [line 10]
!26 = !MDLocation(line: 10, scope: !4)
!27 = !MDLocation(line: 11, scope: !4)
!28 = !{!"0x100\00i1\0011\000", !4, !5, !14} ; [ DW_TAG_auto_variable ] [i1] [line 11]
!29 = !{!"0x102\00147\000\004"} ; [ DW_TAG_expression ] [DW_OP_piece 0 4] [piece, size 4, offset 0]
!31 = !{i32 3, i32 0, i32 12}
!32 = !MDLocation(line: 12, scope: !4)
