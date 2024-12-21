; RUN: llc -mtriple=nvptx64-nvidia-cuda -mattr=+ptx70 < %s | FileCheck %s
; RUN: %if ptxas %{ llc -mtriple=nvptx64-nvidia-cuda -mattr=+ptx70 < %s | %ptxas-verify %}

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
;
; CHECK: DEBUG_VALUE: foo:i <- 3
; CHECK: DEBUG_VALUE: foo:i <- 7
; CHECK: DEBUG_VALUE: foo:i <- %
; CHECK: DEBUG_VALUE: foo:i <- [DW_OP_deref] $vrdepot

; Function Attrs: nounwind ssp uwtable
define i32 @foo() #0 !dbg !4 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.value(metadata i32 3, metadata !10, metadata !DIExpression()), !dbg !15
  %call = call i32 @f3(i32 3) #3, !dbg !16
  call void @llvm.dbg.value(metadata i32 7, metadata !10, metadata !DIExpression()), !dbg !18
  %call1 = call i32 @f1() #3, !dbg !19
  call void @llvm.dbg.value(metadata i32 %call1, metadata !10, metadata !DIExpression()), !dbg !19
  store i32 %call1, ptr %i, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata ptr %i, metadata !10, metadata !DIExpression(DW_OP_deref)), !dbg !24
  call void @f2(ptr %i) #3, !dbg !24
  ret i32 0, !dbg !25
}

declare i32 @f3(i32)

declare i32 @f1()

declare void @f2(ptr)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2, nameTableKind: None)
!1 = !DIFile(filename: "dbg-value-const-byref.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6, retainedNodes: !9)
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
