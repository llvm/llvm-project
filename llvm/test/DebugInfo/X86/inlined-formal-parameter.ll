; RUN: llc -filetype=obj -o %t.o %s
; RUN: llvm-dwarfdump -v %t.o | FileCheck %s

; Testcase generated using 'clang -g -O2 -S -emit-llvm' from the following:
;; void sink(void);
;; static __attribute__((always_inline)) void bar(int a) { sink(); }
;; void foo(void) {
;;   bar(0);
;;   bar(0);
;; }

; Check that we have formal parameters for 'a' in both inlined subroutines.
; CHECK: .debug_info
; CHECK:       DW_TAG_inlined_subroutine
; CHECK-NEXT:    DW_AT_abstract_origin {{.*}} "bar"
; CHECK:         DW_TAG_formal_parameter
; CHECK-NEXT:      DW_AT_const_value
; CHECK-NEXT:      DW_AT_abstract_origin {{.*}} "a"
; CHECK:       DW_TAG_inlined_subroutine
; CHECK-NEXT:    DW_AT_abstract_origin {{.*}} "bar"
; CHECK:         DW_TAG_formal_parameter
; CHECK-NEXT:      DW_AT_location [DW_FORM_data4]	(
; CHECK-NEXT:        [{{.*}}, {{.*}}): DW_OP_consts +0)
; CHECK-NEXT:      DW_AT_abstract_origin {{.*}} "a"

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin"

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !17) #3, !dbg !18
  tail call void @sink() #3, !dbg !20
  tail call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !17) #3, !dbg !21
  tail call void @sink() #3, !dbg !23
  ret void, !dbg !24
}

declare void @sink()

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable  }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "t.c", directory: "/path/to/dir")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !8, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)"}
!17 = !DIExpression()
!18 = !DILocation(line: 2, column: 52, scope: !7, inlinedAt: !19)
!19 = distinct !DILocation(line: 4, column: 3, scope: !4)
!20 = !DILocation(line: 2, column: 57, scope: !7, inlinedAt: !19)
!21 = !DILocation(line: 2, column: 52, scope: !7, inlinedAt: !22)
!22 = distinct !DILocation(line: 5, column: 3, scope: !4)
!23 = !DILocation(line: 2, column: 57, scope: !7, inlinedAt: !22)
!24 = !DILocation(line: 6, column: 1, scope: !4)
