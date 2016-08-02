; RUN: llc -O1 -filetype=obj -o - %s | llvm-dwarfdump -debug-dump=all - | FileCheck %s
; Generated with -O1 from:
; int f1();
; void f2(int*);
; int f3(int);
;
; int foo() {
;   int i = 3;
;   f3(i);
;   i = 7;
;   i = f1();
;   f2(&i);
;   return 0;
; }
;
; Test that we generate valid debug info for optimized code,
; particularly variables that are described as constants and passed
; by reference.
; rdar://problem/14874886
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_location [DW_FORM_data4]	([[LOC:.*]])
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"i"
; CHECK: .debug_loc contents:
; CHECK: [[LOC]]:
;        consts 0x00000003
; CHECK: Beginning address offset: 0x0000000000000{{.*}}
; CHECK:    Ending address offset: [[C1:.*]]
; CHECK:     Location description: 11 03
;        consts 0x00000007
; CHECK: Beginning address offset: [[C1]]
; CHECK:    Ending address offset: [[C2:.*]]
; CHECK:     Location description: 11 07
;        rax, piece 0x00000004
; CHECK: Beginning address offset: [[C2]]
; CHECK:    Ending address offset: [[R1:.*]]
; CHECK:     Location description: 50 93 04
;         rdi+0
; CHECK: Beginning address offset: [[R1]]
; CHECK:    Ending address offset: [[R2:.*]]
; CHECK:     Location description: 75 00
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo() #0 !dbg !4 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 3, i64 0, metadata !10, metadata !DIExpression()), !dbg !15
  %call = call i32 @f3(i32 3) #3, !dbg !16
  call void @llvm.dbg.value(metadata i32 7, i64 0, metadata !10, metadata !DIExpression()), !dbg !18
  %call1 = call i32 (...) @f1() #3, !dbg !19
  call void @llvm.dbg.value(metadata i32 %call1, i64 0, metadata !10, metadata !DIExpression()), !dbg !19
  store i32 %call1, i32* %i, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata i32* %i, i64 0, metadata !10, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @f2(i32* %i) #3, !dbg !24
  ret i32 0, !dbg !25
}

declare i32 @f3(i32)

declare i32 @f1(...)

declare void @f2(i32*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "dbg-value-const-byref.c", directory: "")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, scopeLine: 5, file: !1, scope: !5, type: !6, variables: !9)
!5 = !DIFile(filename: "dbg-value-const-byref.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "i", line: 6, scope: !4, file: !5, type: !8)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.5.0 "}
!14 = !{i32 3}
!15 = !DILocation(line: 6, scope: !4)
!16 = !DILocation(line: 7, scope: !4)
!17 = !{i32 7}
!18 = !DILocation(line: 8, scope: !4)
!19 = !DILocation(line: 9, scope: !4)
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
!24 = !DILocation(line: 10, scope: !4)
!25 = !DILocation(line: 11, scope: !4)
