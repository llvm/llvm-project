; Test dwarf codegen of DW_OP_minus.
; RUN: llc -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s
; RUN: llc -dwarf-version=2 -filetype=obj < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=DWARF2
; RUN: llc -dwarf-version=3 -filetype=obj < %s | llvm-dwarfdump - \
; RUN:   | FileCheck %s --check-prefix=DWARF2

; This was derived manually from:
; int inc(int i) {
;  return i+1;
; }

; DWARF2: .debug_info
; DWARF2: DW_TAG_formal_parameter
; DWARF2-NEXT: DW_AT_name {{.*}}"i"
; DWARF2-NOT:  DW_AT_location

; CHECK: Beginning address offset: 0x0000000000000000
; CHECK:    Ending address offset: 0x0000000000000004
; CHECK:     Location description: 70 00 10 ff ff ff ff 0f 1a 10 01 1c 9f
;        rax+0, constu 0xffffffff, and, constu 0x00000001, minus, stack-value
source_filename = "minus.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define i32 @inc(i32 %i) local_unnamed_addr #1 !dbg !7 {
entry:
  %add = add nsw i32 %i, 1, !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !12, metadata !13), !dbg !14
  ret i32 %add, !dbg !16
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 286322) (llvm/trunk 286305)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "minus.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 4.0.0 (trunk 286322) (llvm/trunk 286305)"}
!7 = distinct !DISubprogram(name: "inc", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "i", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)
!14 = !DILocation(line: 1, column: 13, scope: !7)
!15 = !DILocation(line: 2, column: 11, scope: !7)
!16 = !DILocation(line: 2, column: 3, scope: !7)
