; ModuleID = 'formal_parameter.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"
;
; From (clang -g -c -O1):
;
; int lookup(int* map);
; int verify(int val);
; void foo(int map)
; {
;   lookup(&map);
;   if (!verify(map)) {  }
; }
;
; RUN: opt %s -O2 -S -o %t
; RUN: cat %t | FileCheck --check-prefix=LOWERING %s
; RUN: llc -filetype=obj %t -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Test that we only emit only one DW_AT_formal_parameter "map" for this function.
; rdar://problem/14874886
;
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}map
; CHECK-NOT: DW_AT_name {{.*}}map

; Function Attrs: nounwind ssp uwtable
define void @foo(i32 %map) #0 {
entry:
  %map.addr = alloca i32, align 4
  store i32 %map, i32* %map.addr, align 4, !tbaa !15
  call void @llvm.dbg.declare(metadata i32* %map.addr, metadata !10, metadata !{!"0x102"}), !dbg !14
  %call = call i32 (i32*, ...)* bitcast (i32 (...)* @lookup to i32 (i32*, ...)*)(i32* %map.addr) #3, !dbg !19
  ; Ensure that all dbg intrinsics have the same scope after
  ; LowerDbgDeclare is finished with them.
  ;
  ; LOWERING: call void @llvm.dbg.value{{.*}}, !dbg ![[LOC:.*]]
  ; LOWERING: call void @llvm.dbg.value{{.*}}, !dbg ![[LOC]]
  ; LOWERING: call void @llvm.dbg.value{{.*}}, !dbg ![[LOC]]
%0 = load i32* %map.addr, align 4, !dbg !20, !tbaa !15
  %call1 = call i32 (i32, ...)* bitcast (i32 (...)* @verify to i32 (i32, ...)*)(i32 %0) #3, !dbg !20
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @lookup(...)

declare i32 @verify(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !{!"0x11\0012\00clang version 3.5.0 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [formal_parameter.c] [DW_LANG_C99]
!1 = !{!"formal_parameter.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\001\000\001\000\006\00256\001\002", !1, !5, !6, null, void (i32)* @foo, null, null, !9} ; [ DW_TAG_subprogram ] [line 1] [def] [scope 2] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [formal_parameter.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!10}
!10 = !{!"0x101\00map\0016777217\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [map] [line 1]
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 1, !"Debug Info Version", i32 2}
!13 = !{!"clang version 3.5.0 "}
!14 = !MDLocation(line: 1, scope: !4)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !MDLocation(line: 3, scope: !4)
!20 = !MDLocation(line: 4, scope: !21)
!21 = !{!"0xb\004\000\000", !1, !4} ; [ DW_TAG_lexical_block ] [formal_parameter.c]
!22 = !MDLocation(line: 5, scope: !4)
