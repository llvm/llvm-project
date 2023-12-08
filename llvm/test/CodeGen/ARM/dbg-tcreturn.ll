; RUN: llc %s -o - -stop-after=finalize-isel -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-Fi8-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7-apple-ios7.0.0"

; CHECK-LABEL: name: test
; CHECK:       body:
; CHECK-NEXT:   bb.0.entry:
; CHECK-NEXT:     liveins: $r0, $r1
; CHECK:          %1:gpr = COPY $r1
; CHECK-NEXT:     %0:gpr = COPY $r0
; CHECK-NEXT:     $r0 = COPY %0
; CHECK-NEXT:     $r1 = COPY %1
; CHECK-NEXT:     DBG_VALUE $noreg, $noreg, !13, !DIExpression(), debug-location !16
; CHECK-NEXT:     TCRETURNdi &__divsi3, 0, csr_ios, implicit $sp, implicit $r0, implicit $r1

define i32 @test(i32 %a1, i32 %a2) !dbg !5 {
entry:
  %res = sdiv i32 %a1, %a2
  call void @llvm.dbg.value(metadata i32 %res, metadata !13, metadata !DIExpression()), !dbg !16
  ret i32 %res
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, producer: "Swift", isOptimized: true, runtimeVersion: 5, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.swift", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"Swift Minor Version", i8 3}
!5 = distinct !DISubprogram(name: "n0", linkageName: "n1", scope: !7, file: !6, line: 86, type: !8, scopeLine: 86, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DIFile(filename: "bar.swift", directory: "")
!7 = !DIModule(scope: null, name: "Swift")
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !12}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Int", scope: !7, file: !11, size: 32, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "$i1")
!11 = !DIFile(filename: "f1.swift", directory: "")
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "n2", scope: !7, file: !6, size: 32, elements: !2, runtimeLang: DW_LANG_Swift, identifier: "n3")
!13 = !DILocalVariable(name: "n4", scope: !14, file: !1, line: 89, type: !15)
!14 = distinct !DILexicalBlock(scope: !5, file: !6, line: 86, column: 34)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !10)
!16 = !DILocation(line: 89, column: 9, scope: !14)
