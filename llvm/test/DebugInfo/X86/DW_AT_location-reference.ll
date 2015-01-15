; RUN: llc -O1 -filetype=obj -mtriple=x86_64-apple-darwin < %s > %t
; RUN: llvm-dwarfdump %t  | FileCheck %s
; RUN: llvm-objdump -r %t | FileCheck -check-prefix=DARWIN %s
; RUN: llc -O1 -filetype=obj -mtriple=x86_64-pc-linux-gnu < %s > %t
; RUN: llvm-dwarfdump %t  | FileCheck %s
; RUN: llvm-objdump -r %t | FileCheck -check-prefix=LINUX %s

; PR9493
; Adapted from the original test case in r127757.
; We use 'llc -O1' to induce variable 'x' to live in different locations.
; We don't actually care where 'x' lives, or what exact optimizations get
; used, as long as 'x' moves around we're fine.

; // The variable 'x' lives in different locations, so it needs an entry in
; // the .debug_loc table section, referenced by DW_AT_location.
; // This ref is not relocatable on Darwin, and is relocatable elsewhere.
; extern int g(int, int);
; extern int a;
; 
; void f(void) {
;   int x;
;   a = g(0, 0);
;   x = 1;
;   while (x & 1) { x *= a; }
;   a = g(x, 0);
;   x = 2;
;   while (x & 2) { x *= a; }
;   a = g(0, x);
; }

; // The 'x' variable and its symbol reference location
; CHECK: .debug_info contents:
; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_location [DW_FORM_sec_offset] (0x00000000)
; CHECK-NEXT:   DW_AT_name {{.*}} "x"
; CHECK-NEXT:   DW_AT_decl_file
; CHECK-NEXT:   DW_AT_decl_line
; CHECK-NEXT:   DW_AT_type

; Check that the location contains only 4 ranges - this verifies that the 4th
; and 5th ranges were successfully merged into a single range.
; CHECK: .debug_loc contents:
; CHECK: 0x00000000:
; CHECK: Beginning address offset:
; CHECK: Beginning address offset:
; CHECK: Beginning address offset:
; CHECK: Beginning address offset:
; CHECK-NOT: Beginning address offset:

; Check that we have no relocations in Darwin's output.
; DARWIN-NOT: X86_64_RELOC{{.*}} __debug_loc

; Check we have a relocation for the debug_loc entry in Linux output.
; LINUX: RELOCATION RECORDS FOR [.rela.debug_info]
; LINUX-NOT: RELOCATION RECORDS
; LINUX: R_X86_64{{.*}} .debug_loc+0

; ModuleID = 'simple.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"

@a = external global i32

define void @f() nounwind {
entry:
  %call = tail call i32 @g(i32 0, i32 0) nounwind, !dbg !8
  store i32 %call, i32* @a, align 4, !dbg !8
  tail call void @llvm.dbg.value(metadata i32 1, i64 0, metadata !5, metadata !{!"0x102"}), !dbg !13
  br label %while.body

while.body:                                       ; preds = %entry, %while.body
  %x.017 = phi i32 [ 1, %entry ], [ %mul, %while.body ]
  %mul = mul nsw i32 %call, %x.017, !dbg !14
  %and = and i32 %mul, 1, !dbg !14
  %tobool = icmp eq i32 %and, 0, !dbg !14
  br i1 %tobool, label %while.end, label %while.body, !dbg !14

while.end:                                        ; preds = %while.body
  tail call void @llvm.dbg.value(metadata i32 %mul, i64 0, metadata !5, metadata !{!"0x102"}), !dbg !14
  %call4 = tail call i32 @g(i32 %mul, i32 0) nounwind, !dbg !15
  store i32 %call4, i32* @a, align 4, !dbg !15
  tail call void @llvm.dbg.value(metadata i32 2, i64 0, metadata !5, metadata !{!"0x102"}), !dbg !17
  br label %while.body9

while.body9:                                      ; preds = %while.end, %while.body9
  %x.116 = phi i32 [ 2, %while.end ], [ %mul12, %while.body9 ]
  %mul12 = mul nsw i32 %call4, %x.116, !dbg !18
  %and7 = and i32 %mul12, 2, !dbg !18
  %tobool8 = icmp eq i32 %and7, 0, !dbg !18
  br i1 %tobool8, label %while.end13, label %while.body9, !dbg !18

while.end13:                                      ; preds = %while.body9
  tail call void @llvm.dbg.value(metadata i32 %mul12, i64 0, metadata !5, metadata !{!"0x102"}), !dbg !18
  %call15 = tail call i32 @g(i32 0, i32 %mul12) nounwind, !dbg !19
  store i32 %call15, i32* @a, align 4, !dbg !19
  ret void, !dbg !20
}

declare i32 @g(i32, i32)

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24}

!0 = !{!"0x2e\00f\00f\00\004\000\001\000\006\00256\001\004", !23, !1, !3, null, void ()* @f, null, null, !22} ; [ DW_TAG_subprogram ] [line 4] [def] [f]
!1 = !{!"0x29", !23} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 3.0 (trunk)\001\00\000\00\001", !23, !4, !4, !21, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !23, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{null}
!5 = !{!"0x100\00x\005\000", !6, !1, !7} ; [ DW_TAG_auto_variable ]
!6 = !{!"0xb\004\0014\000", !23, !0} ; [ DW_TAG_lexical_block ]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!8 = !MDLocation(line: 6, column: 3, scope: !6)
!12 = !{i32 1}
!13 = !MDLocation(line: 7, column: 3, scope: !6)
!14 = !MDLocation(line: 8, column: 3, scope: !6)
!15 = !MDLocation(line: 9, column: 3, scope: !6)
!16 = !{i32 2}
!17 = !MDLocation(line: 10, column: 3, scope: !6)
!18 = !MDLocation(line: 11, column: 3, scope: !6)
!19 = !MDLocation(line: 12, column: 3, scope: !6)
!20 = !MDLocation(line: 13, column: 1, scope: !6)
!21 = !{!0}
!22 = !{!5}
!23 = !{!"simple.c", !"/home/rengol01/temp/tests/dwarf/relocation"}
!24 = !{i32 1, !"Debug Info Version", i32 2}
