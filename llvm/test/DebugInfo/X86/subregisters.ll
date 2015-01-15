; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t.o -filetype=obj -O0
; RUN: llvm-dwarfdump %t.o | FileCheck %s
;
; Test that on x86_64, the 32-bit subregister esi is emitted as
; DW_OP_piece 32 of the 64-bit rsi.
;
; rdar://problem/16015314
;
; CHECK:  DW_AT_location [DW_FORM_block1]       (<0x03> 54 93 04 )
; CHECK:  DW_AT_name [DW_FORM_strp]{{.*}} "a"
;
; struct bar {
;   int a;
;   int b;
; };
;
; void doSomething() __attribute__ ((noinline));
;
; void doSomething(struct bar *b)
; {
;   int a = b->a;
;   printf("%d\n", a); // set breakpoint here
; }
;
; int main()
; {
;   struct bar myBar = { 3, 4 };
;   doSomething(&myBar);
;   return 0;
; }

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.bar = type { i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@main.myBar = private unnamed_addr constant %struct.bar { i32 3, i32 4 }, align 4

; Function Attrs: noinline nounwind ssp uwtable
define void @doSomething(%struct.bar* nocapture readonly %b) #0 {
entry:
  tail call void @llvm.dbg.value(metadata %struct.bar* %b, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !25
  %a1 = getelementptr inbounds %struct.bar* %b, i64 0, i32 0, !dbg !26
  %0 = load i32* %a1, align 4, !dbg !26, !tbaa !27
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !26
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i64 0, i64 0), i32 %0) #4, !dbg !32
  ret void, !dbg !33
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #2

; Function Attrs: nounwind ssp uwtable
define i32 @main() #3 {
entry:
  %myBar = alloca i64, align 8, !dbg !34
  %tmpcast = bitcast i64* %myBar to %struct.bar*, !dbg !34
  tail call void @llvm.dbg.declare(metadata %struct.bar* %tmpcast, metadata !21, metadata !{!"0x102"}), !dbg !34
  store i64 17179869187, i64* %myBar, align 8, !dbg !34
  call void @doSomething(%struct.bar* %tmpcast), !dbg !35
  ret i32 0, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { noinline nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { nounwind ssp uwtable }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = !{!"0x11\0012\00clang version 3.5 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [subregisters.c] [DW_LANG_C99]
!1 = !{!"subregisters.c", !""}
!2 = !{}
!3 = !{!4, !17}
!4 = !{!"0x2e\00doSomething\00doSomething\00\0010\000\001\000\006\00256\001\0011", !1, !5, !6, null, void (%struct.bar*)* @doSomething, null, null, !14} ; [ DW_TAG_subprogram ] [line 10] [def] [scope 11] [doSomething]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [subregisters.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{null, !8}
!8 = !{!"0xf\00\000\0064\0064\000\000", null, null, !9} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from bar]
!9 = !{!"0x13\00bar\003\0064\0032\000\000\000", !1, null, null, !10, null, null, null} ; [ DW_TAG_structure_type ] [bar] [line 3, size 64, align 32, offset 0] [def] [from ]
!10 = !{!11, !13}
!11 = !{!"0xd\00a\004\0032\0032\000\000", !1, !9, !12} ; [ DW_TAG_member ] [a] [line 4, size 32, align 32, offset 0] [from int]
!12 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = !{!"0xd\00b\005\0032\0032\0032\000", !1, !9, !12} ; [ DW_TAG_member ] [b] [line 5, size 32, align 32, offset 32] [from int]
!14 = !{!15, !16}
!15 = !{!"0x101\00b\0016777226\000", !4, !5, !8} ; [ DW_TAG_arg_variable ] [b] [line 10]
!16 = !{!"0x100\00a\0012\000", !4, !5, !12} ; [ DW_TAG_auto_variable ] [a] [line 12]
!17 = !{!"0x2e\00main\00main\00\0016\000\001\000\006\000\001\0017", !1, !5, !18, null, i32 ()* @main, null, null, !20} ; [ DW_TAG_subprogram ] [line 16] [def] [scope 17] [main]
!18 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !19, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!19 = !{!12}
!20 = !{!21}
!21 = !{!"0x100\00myBar\0018\000", !17, !5, !9} ; [ DW_TAG_auto_variable ] [myBar] [line 18]
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 1, !"Debug Info Version", i32 2}
!24 = !{!"clang version 3.5 "}
!25 = !MDLocation(line: 10, scope: !4)
!26 = !MDLocation(line: 12, scope: !4)
!27 = !{!28, !29, i64 0}
!28 = !{!"bar", !29, i64 0, !29, i64 4}
!29 = !{!"int", !30, i64 0}
!30 = !{!"omnipotent char", !31, i64 0}
!31 = !{!"Simple C/C++ TBAA"}
!32 = !MDLocation(line: 13, scope: !4)
!33 = !MDLocation(line: 14, scope: !4)
!34 = !MDLocation(line: 18, scope: !17)
!35 = !MDLocation(line: 19, scope: !17)
!36 = !MDLocation(line: 20, scope: !17)
