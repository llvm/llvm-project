; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

declare void @raf(i32, ...)

; CHECK-LABEL: test3
; CHECK-DAG:   movl    $7, %edi
; CHECK-DAG:   xorl    %esi, %esi
; CHECK-DAG:   xorl    %edx, %edx
; CHECK-DAG:   xorl    %ecx, %ecx
; CHECK-DAG:   xorl    %r8d, %r8d
; CHECK-DAG:   xorl    %r9d, %r9d
; CHECK-DAG:   xorl    %eax, %eax
; CHECK:       pushq   %rbx
; CHECK:       pushq   $0
; CHECK:       callq   raf


; Function Attrs: nounwind uwtable
define void @test3() {
entry:
  br label %for.body

for.body:
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.dbg.value(metadata i32 %i.04, i64 0, metadata !10, metadata !12), !dbg !6
  tail call void (i32, ...) @raf(i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 %i.04)
  %inc = add nuw nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 21
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void  
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) 
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (trunk 288844)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "pr31242.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 4.0.0 (trunk 288844)"}
!6 = !DILocation(line: 2, column: 16, scope: !7)
!7 = distinct !DISubprogram(name: "test3", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: true, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "value", arg: 1, scope: !7, file: !1, line: 2, type: !11)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIExpression()
