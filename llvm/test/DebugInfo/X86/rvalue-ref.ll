; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj -O0
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: DW_TAG_rvalue_reference_type

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

define void @_Z3fooOi(i32* %i) uwtable ssp {
entry:
  %i.addr = alloca i32*, align 8
  store i32* %i, i32** %i.addr, align 8
  call void @llvm.dbg.declare(metadata !{i32** %i.addr}, metadata !11), !dbg !12
  %0 = load i32** %i.addr, align 8, !dbg !13
  %1 = load i32* %0, align 4, !dbg !13
  %call = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32 %1), !dbg !13
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17}

!0 = metadata !{i32 786449, metadata !16, i32 4, metadata !"clang version 3.2 (trunk 157054) (llvm/trunk 157060)", i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1, metadata !""} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 0}
!3 = metadata !{metadata !5}
!5 = metadata !{i32 786478, metadata !16, metadata !6, metadata !"foo", metadata !"foo", metadata !"_Z3fooOi", i32 4, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32*)* @_Z3fooOi, null, null, metadata !1, i32 5} ; [ DW_TAG_subprogram ]
!6 = metadata !{i32 786473, metadata !16} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{i32 786498, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_rvalue_reference_type ]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 786689, metadata !5, metadata !"i", metadata !6, i32 16777220, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!12 = metadata !{i32 4, i32 17, metadata !5, null}
!13 = metadata !{i32 6, i32 3, metadata !14, null}
!14 = metadata !{i32 786443, metadata !16, metadata !5, i32 5, i32 1, i32 0} ; [ DW_TAG_lexical_block ]
!15 = metadata !{i32 7, i32 1, metadata !14, null}
!16 = metadata !{metadata !"foo.cpp", metadata !"/Users/echristo/tmp"}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
