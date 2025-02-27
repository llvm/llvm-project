; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/thinlto_funcimport_debug.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; If we import func1 and not func2 we should only link DISubprogram for func1
; RUN: llvm-link %t2.bc -summary-index=%t3.thinlto.bc -import=func1:%t.bc -S | FileCheck %s

;; Repeat runlines using "new" debuginfo iterators mode.
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc --try-experimental-debuginfo-iterators
; RUN: llvm-link %t2.bc -summary-index=%t3.thinlto.bc -import=func1:%t.bc -S --try-experimental-debuginfo-iterators | FileCheck %s

; CHECK: declare i32 @func2
; CHECK: define available_externally i32 @func1

; Ensure that each subprogram points to the correct CU.
; CHECK: ![[CU1:[0-9]+]] = distinct !DICompileUnit(
; CHECK: ![[CU2:[0-9]+]] = distinct !DICompileUnit(

; CHECK: distinct !DISubprogram(name: "main", {{.*}}, unit: ![[CU1]]
; CHECK: distinct !DISubprogram(name: "func1", {{.*}}, unit: ![[CU2]]
; CHECK-NOT: distinct !DISubprogram(name: "func2"
; CHECK: distinct !DISubprogram(name: "func3", {{.*}}, unit: ![[CU2]]
; CHECK: distinct !DISubprogram(name: "func4", {{.*}}, unit: ![[CU2]]


; ModuleID = 'dbg.o'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define i32 @func1(i32 %n) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %n, i64 0, metadata !9, metadata !17), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 5, i64 0, metadata !10, metadata !17), !dbg !19
  %cmp = icmp sgt i32 %n, 10, !dbg !20
  %. = select i1 %cmp, i32 10, i32 5, !dbg !22
  tail call void @llvm.dbg.value(metadata i32 %., i64 0, metadata !10, metadata !17), !dbg !19
  ret i32 %., !dbg !23
}

; Function Attrs: nounwind readnone uwtable
define i32 @func2(i32 %n) #0 !dbg !11 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %n, i64 0, metadata !13, metadata !17), !dbg !24
  ret i32 %n, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "dbg.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "func1", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10}
!9 = !DILocalVariable(name: "n", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!10 = !DILocalVariable(name: "x", scope: !4, file: !1, line: 2, type: !7)
!11 = distinct !DISubprogram(name: "func2", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!12 = !{!13}
!13 = !DILocalVariable(name: "n", arg: 1, scope: !11, file: !1, line: 8, type: !7)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"clang version 3.8.0 (trunk 251407) (llvm/trunk 251401)"}
!17 = !DIExpression()
!18 = !DILocation(line: 1, column: 15, scope: !4)
!19 = !DILocation(line: 2, column: 7, scope: !4)
!20 = !DILocation(line: 3, column: 9, scope: !21, inlinedAt: !26)
!21 = distinct !DILexicalBlock(scope: !27, file: !1, line: 3, column: 7)
!22 = !DILocation(line: 3, column: 7, scope: !4)
!23 = !DILocation(line: 5, column: 3, scope: !4)
!24 = !DILocation(line: 8, column: 15, scope: !11)
!25 = !DILocation(line: 9, column: 3, scope: !11)
!26 = !DILocation(line: 9, column: 3, scope: !4)
!27 = distinct !DISubprogram(name: "func3", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !28)
!28 = !{!29}
!29 = !DILocalVariable(name: "n", arg: 1, scope: !30, file: !1, line: 8, type: !7)
!30 = distinct !DISubprogram(name: "func4", scope: !1, file: !1, line: 8, type: !5, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !31)
!31 = !{!32}
!32 = !DILocalVariable(name: "n", arg: 1, scope: !30, file: !1, line: 8, type: !7)

