; RUN: opt -S -asan %s | FileCheck %s

; The IR of this testcase is generated from the following C code:
; void bar (int);
;
; void foo() {
;   __block int x;
;   bar(x);
; }
; by compiling it with 'clang -emit-llvm -g -S' and then by manually
; adding the sanitize_address attribute to the @foo() function (so
; that ASAN accepts to instrument the function in the above opt run).

; Check that the location of the ASAN instrumented __block variable is
; correct.
; CHECK: [ DW_TAG_expression ] [DW_OP_deref] [DW_OP_plus 8] [DW_OP_deref] [DW_OP_plus 24]

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

%struct.__block_byref_x = type { i8*, %struct.__block_byref_x*, i32, i32, i32 }

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 {
entry:
  %x = alloca %struct.__block_byref_x, align 8
  call void @llvm.dbg.declare(metadata %struct.__block_byref_x* %x, metadata !12, metadata !22), !dbg !23
  %byref.isa = getelementptr inbounds %struct.__block_byref_x* %x, i32 0, i32 0, !dbg !24
  store i8* null, i8** %byref.isa, !dbg !24
  %byref.forwarding = getelementptr inbounds %struct.__block_byref_x* %x, i32 0, i32 1, !dbg !24
  store %struct.__block_byref_x* %x, %struct.__block_byref_x** %byref.forwarding, !dbg !24
  %byref.flags = getelementptr inbounds %struct.__block_byref_x* %x, i32 0, i32 2, !dbg !24
  store i32 0, i32* %byref.flags, !dbg !24
  %byref.size = getelementptr inbounds %struct.__block_byref_x* %x, i32 0, i32 3, !dbg !24
  store i32 32, i32* %byref.size, !dbg !24
  %forwarding = getelementptr inbounds %struct.__block_byref_x* %x, i32 0, i32 1, !dbg !25
  %0 = load %struct.__block_byref_x** %forwarding, !dbg !25
  %x1 = getelementptr inbounds %struct.__block_byref_x* %0, i32 0, i32 4, !dbg !25
  %1 = load i32* %x1, align 4, !dbg !25
  call void @bar(i32 %1), !dbg !25
  %2 = bitcast %struct.__block_byref_x* %x to i8*, !dbg !26
  call void @_Block_object_dispose(i8* %2, i32 8) #3, !dbg !26
  ret void, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @bar(i32) #2

declare void @_Block_object_dispose(i8*, i32)

attributes #0 = { nounwind ssp uwtable sanitize_address "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = !{!"0x11\0012\00clang version 3.6.0 (trunk 223120) (llvm/trunk 223119)\000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/tmp/block.c] [DW_LANG_C99]
!1 = !{!"block.c", !"/tmp"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\003\000\001\000\000\000\000\003", !1, !5, !6, null, void ()* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = !{!"0x29", !1}    ; [ DW_TAG_file_type ] [/tmp/block.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 2}
!9 = !{i32 2, !"Debug Info Version", i32 2}
!10 = !{i32 1, !"PIC Level", i32 2}
!11 = !{!"clang version 3.6.0 (trunk 223120) (llvm/trunk 223119)"}
!12 = !{!"0x100\00x\004\000", !4, !5, !13} ; [ DW_TAG_auto_variable ] [x] [line 4]
!13 = !{!"0x13\00\000\00224\000\000\0016\000", !1, !5, null, !14, null, null, null} ; [ DW_TAG_structure_type ] [line 0, size 224, align 0, offset 0] [def] [from ]
!14 = !{!15, !17, !18, !20, !21}
!15 = !{!"0xd\00__isa\000\0064\0064\000\000", !1, !5, !16} ; [ DW_TAG_member ] [__isa] [line 0, size 64, align 64, offset 0] [from ]
!16 = !{!"0xf\00\000\0064\0064\000\000", null, null, null} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!17 = !{!"0xd\00__forwarding\000\0064\0064\0064\000", !1, !5, !16} ; [ DW_TAG_member ] [__forwarding] [line 0, size 64, align 64, offset 64] [from ]
!18 = !{!"0xd\00__flags\000\0032\0032\00128\000", !1, !5, !19} ; [ DW_TAG_member ] [__flags] [line 0, size 32, align 32, offset 128] [from int]
!19 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = !{!"0xd\00__size\000\0032\0032\00160\000", !1, !5, !19} ; [ DW_TAG_member ] [__size] [line 0, size 32, align 32, offset 160] [from int]
!21 = !{!"0xd\00x\000\0032\0032\00192\000", !1, !5, !19} ; [ DW_TAG_member ] [x] [line 0, size 32, align 32, offset 192] [from int]
!22 = !{!"0x102\0034\008\006\0034\0024"} ; [ DW_TAG_expression ] [DW_OP_plus 8] [DW_OP_deref] [DW_OP_plus 24]
!23 = !MDLocation(line: 4, column: 15, scope: !4)
!24 = !MDLocation(line: 4, column: 3, scope: !4)
!25 = !MDLocation(line: 5, column: 3, scope: !4)
!26 = !MDLocation(line: 6, column: 1, scope: !4)
