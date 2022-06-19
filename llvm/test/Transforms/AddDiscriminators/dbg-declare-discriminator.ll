; RUN: opt -S -passes=add-discriminators < %s | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata)

; This checks whether the add-discriminators pass producess valid metadata on
; llvm.dbg.declare instructions
;
; CHECK-LABEL: @test_valid_metadata
define void @test_valid_metadata() {
  %a = alloca i8
  call void @llvm.dbg.declare(metadata i8* %a, metadata !2, metadata !5), !dbg !6
  %b = alloca i8
  call void @llvm.dbg.declare(metadata i8* %b, metadata !9, metadata !5), !dbg !11
  ret void
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!12}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DILocalVariable(scope: !3)
!3 = distinct !DISubprogram(scope: null, file: !4, isLocal: false, isDefinition: true, isOptimized: false, unit: !12)
!4 = !DIFile(filename: "a.cpp", directory: "/tmp")
!5 = !DIExpression()
!6 = !DILocation(line: 0, scope: !3, inlinedAt: !7)
!7 = distinct !DILocation(line: 0, scope: !8)
!8 = distinct !DISubprogram(linkageName: "test_valid_metadata", scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !12)
!9 = !DILocalVariable(scope: !10)
!10 = distinct !DISubprogram(scope: null, file: !4, isLocal: false, isDefinition: true, isOptimized: false, unit: !12)
!11 = !DILocation(line: 0, scope: !10)
!12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !4)
