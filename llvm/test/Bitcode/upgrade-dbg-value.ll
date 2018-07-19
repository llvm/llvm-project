; Test upgrade of dbg.dvalue intrinsics with offsets.
;
; RUN: llvm-dis < %s.bc | FileCheck %s
; RUN: verify-uselistorder < %s.bc

define void @f() !dbg !3 {
entry:
  ; CHECK-NOT: call void @llvm.dbg.value
  ; CHECK: call void @llvm.dbg.value(metadata i32 42, metadata !8, metadata !DIExpression())
  call void @llvm.dbg.value(metadata i32 42, i64 0, metadata !8, metadata !9), !dbg !10
  ; CHECK-NOT: call void @llvm.dbg.value
  call void @llvm.dbg.value(metadata i32 0, i64 1, metadata !8, metadata !9), !dbg !10
  ret void
}

; CHECK: declare void @llvm.dbg.value(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llc r309174", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 1, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, isOptimized: false, unit: !0, variables: !7)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{!8}
!8 = !DILocalVariable(name: "i", scope: !3, file: !1, line: 2, type: !6)
!9 = !DIExpression()
!10 = !DILocation(line: 2, scope: !3)
