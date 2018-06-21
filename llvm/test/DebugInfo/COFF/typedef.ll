; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s

; CHECK: CodeViewDebugInfo [
; CHECK:   Subsection [
; CHECK:     LocalSym {
; CHECK:       Type: wchar_t (0x71)
; CHECK:       Flags [ (0x0)
; CHECK:       ]
; CHECK:       VarName: foo
; CHECK:     }
; CHECK:   Subsection [
; CHECK:     SubSectionType: Symbols (0xF1)
; CHECK:     UDTSym {
; CHECK:       Type: wchar_t (0x71)
; CHECK:       UDTName: XYZ
; CHECK:     }
; CHECK:   ]

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define void @test1() !dbg !5 {
entry:
  %foo = alloca i16, align 2
  call void @llvm.dbg.declare(metadata i16* %foo, metadata !8, metadata !11), !dbg !12
  ret void, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "-", directory: "/usr/local/google/home/majnemer/llvm/src")
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: !6, file: !6, type: !7, unit: !0, variables: !{})
!6 = !DIFile(filename: "<stdin>", directory: ".")
!7 = !DISubroutineType(types: !{})
!8 = !DILocalVariable(name: "foo", scope: !5, file: !6, line: 3, type: !9)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "XYZ", file: !6, line: 2, baseType: !10)
!10 = !DIBasicType(name: "wchar_t", size: 16, align: 16, encoding: DW_ATE_unsigned)
!11 = !DIExpression()
!12 = !DILocation(line: 3, column: 16, scope: !5)
