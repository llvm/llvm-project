; RUN: llc -mtriple=armv7-none-linux-gnueabihf < %s -o - | FileCheck %s

; Function Attrs: nounwind
define void @need_cfi_def_cfa_offset() #0 !dbg !3 {
; CHECK-LABEL: need_cfi_def_cfa_offset:
; CHECK: sub	sp, sp, #4
; CHECK: .cfi_def_cfa_offset 4
entry:
  %Depth = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %Depth, metadata !9, metadata !10), !dbg !11
  store i32 2, i32* %Depth, align 4, !dbg !11
  ret void, !dbg !12
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "", isOptimized: false, subprograms: !{!3})
!1 = !DIFile(filename: "file.c", directory: "/dir")
!2 = !{}
!3 = distinct !DISubprogram(name: "need_cfi_def_cfa_offset", scope: !1, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, variables: !2)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocalVariable(name: "Depth", scope: !3, file: !1, line: 3, type: !6)
!10 = !DIExpression()
!11 = !DILocation(line: 3, column: 9, scope: !3)
!12 = !DILocation(line: 7, column: 5, scope: !3)
