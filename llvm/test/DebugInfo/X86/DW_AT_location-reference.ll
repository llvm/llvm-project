; RUN: llc -O1 -filetype=obj -mtriple=x86_64-apple-darwin < %s > %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s --check-prefixes=CHECK,DWARFv4
; RUN: llvm-objdump -r %t | FileCheck --check-prefix=DARWIN %s

; RUN: llc -O1 -filetype=obj -mtriple=x86_64-pc-linux-gnu < %s > %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s --check-prefixes=CHECK,DWARFv4
; RUN: llvm-objdump -r %t | FileCheck --check-prefix=LINUX %s

; RUN: llc -dwarf-version=3 -O1 -filetype=obj -mtriple=x86_64-pc-linux-gnu < %s > %t
; RUN: llvm-dwarfdump -debug-info -v %t | FileCheck %s --check-prefixes=CHECK,DWARF32v3
; RUN: llvm-objdump -r %t | FileCheck --check-prefix=LINUX %s

; RUN: llc -dwarf64 -dwarf-version=3 -O1 -filetype=obj -mtriple=x86_64-pc-linux-gnu < %s > %t
; RUN: llvm-dwarfdump -debug-info -v %t | FileCheck %s --check-prefixes=CHECK,DWARF64v3
; RUN: llvm-objdump -r %t | FileCheck --check-prefix=LINUX %s

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
; DWARF32v3-NEXT: DW_AT_location [DW_FORM_data4] (0x00000000
; DWARF64v3-NEXT: DW_AT_location [DW_FORM_data8] (0x00000000
; DWARFv4-NEXT:   DW_AT_location [DW_FORM_sec_offset] (0x00000000
; Check that the location contains only 4 ranges.
; CHECK-NEXT:   [0x{{[0-9a-f]*}}, 0x{{[0-9a-f]*}})
; CHECK-NEXT:   [0x{{[0-9a-f]*}}, 0x{{[0-9a-f]*}})
; CHECK-NEXT:   [0x{{[0-9a-f]*}}, 0x{{[0-9a-f]*}})
; CHECK-NEXT:   [0x{{[0-9a-f]*}}, 0x{{[0-9a-f]*}}){{.*}})
; CHECK-NEXT:   DW_AT_name {{.*}} "x"
; CHECK-NEXT:   DW_AT_decl_file
; CHECK-NEXT:   DW_AT_decl_line
; CHECK-NEXT:   DW_AT_type

; Check that we have no relocations in Darwin's output.
; DARWIN-NOT: X86_64_RELOC{{.*}} __debug_loc

; Check we have a relocation for the debug_loc entry in Linux output.
; LINUX: RELOCATION RECORDS FOR [.debug_info]
; LINUX-NOT: RELOCATION RECORDS
; LINUX: R_X86_64{{.*}} .debug_loc

; ModuleID = 'simple.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32"

@a = external global i32

define void @f() nounwind !dbg !0 {
entry:
  %call = tail call i32 @g(i32 0, i32 0) nounwind, !dbg !8
  store i32 %call, ptr @a, align 4, !dbg !8
  tail call void @llvm.dbg.value(metadata i32 1, metadata !5, metadata !DIExpression()), !dbg !13
  br label %while.body

while.body:                                       ; preds = %entry, %while.body
  %x.017 = phi i32 [ 1, %entry ], [ %mul, %while.body ]
  %mul = mul nsw i32 %call, %x.017, !dbg !14
  %and = and i32 %mul, 1, !dbg !14
  %tobool = icmp eq i32 %and, 0, !dbg !14
  br i1 %tobool, label %while.end, label %while.body, !dbg !14

while.end:                                        ; preds = %while.body
  tail call void @llvm.dbg.value(metadata i32 %mul, metadata !5, metadata !DIExpression()), !dbg !14
  %call4 = tail call i32 @g(i32 %mul, i32 0) nounwind, !dbg !15
  store i32 %call4, ptr @a, align 4, !dbg !15
  tail call void @llvm.dbg.value(metadata i32 2, metadata !5, metadata !DIExpression()), !dbg !17
  br label %while.body9

while.body9:                                      ; preds = %while.end, %while.body9
  %x.116 = phi i32 [ 2, %while.end ], [ %mul12, %while.body9 ]
  %mul12 = mul nsw i32 %call4, %x.116, !dbg !18
  %and7 = and i32 %mul12, 2, !dbg !18
  %tobool8 = icmp eq i32 %and7, 0, !dbg !18
  br i1 %tobool8, label %while.end13, label %while.body9, !dbg !18

while.end13:                                      ; preds = %while.body9
  tail call void @llvm.dbg.value(metadata i32 %mul12, metadata !5, metadata !DIExpression()), !dbg !18
  %call15 = tail call i32 @g(i32 0, i32 %mul12) nounwind, !dbg !19
  store i32 %call15, ptr @a, align 4, !dbg !19
  ret void, !dbg !20
}

declare i32 @g(i32, i32)

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24}

!0 = distinct !DISubprogram(name: "f", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 4, file: !23, scope: !1, type: !3, retainedNodes: !22)
!1 = !DIFile(filename: "simple.c", directory: "/home/rengol01/temp/tests/dwarf/relocation")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk)", isOptimized: true, emissionKind: FullDebug, file: !23, enums: !{}, retainedTypes: !{}, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DILocalVariable(name: "x", line: 5, scope: !6, file: !1, type: !7)
!6 = distinct !DILexicalBlock(line: 4, column: 14, file: !23, scope: !0)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 6, column: 3, scope: !6)
!12 = !{i32 1}
!13 = !DILocation(line: 7, column: 3, scope: !6)
!14 = !DILocation(line: 8, column: 3, scope: !6)
!15 = !DILocation(line: 9, column: 3, scope: !6)
!16 = !{i32 2}
!17 = !DILocation(line: 10, column: 3, scope: !6)
!18 = !DILocation(line: 11, column: 3, scope: !6)
!19 = !DILocation(line: 12, column: 3, scope: !6)
!20 = !DILocation(line: 13, column: 1, scope: !6)
!22 = !{!5}
!23 = !DIFile(filename: "simple.c", directory: "/home/rengol01/temp/tests/dwarf/relocation")
!24 = !{i32 1, !"Debug Info Version", i32 3}
