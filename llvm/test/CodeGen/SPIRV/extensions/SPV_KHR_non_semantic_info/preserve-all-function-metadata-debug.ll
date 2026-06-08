; Adapted from the SPIRV-LLVM-Translator --spirv-preserve-auxdata forward-path
; tests. Debug info (!dbg) must never be emitted as NonSemantic.AuxData. The
; "keep-me" attribute is a positive control: it confirms the feature is active,
; so the absence of a debug record is meaningful rather than the feature being
; off.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info -spirv-preserve-auxdata \
; RUN:   %s -o - | FileCheck %s

; CHECK: %[[#Import:]] = OpExtInstImport "NonSemantic.AuxData"
; CHECK-DAG: %[[#Attr:]] = OpString "keep-me"
; CHECK-DAG: OpName %[[#Fcn:]] "foo"
; CHECK-DAG: %[[#VoidT:]] = OpTypeVoid

; CHECK: %[[#]] = OpExtInst %[[#VoidT]] %[[#Import]] {{.+}} %[[#Fcn]] %[[#Attr]]
; CHECK-NOT: OpExtInst %[[#VoidT]] %[[#Import]]

target triple = "spir64-unknown-unknown"

define spir_func void @foo() #0 !dbg !4 {
  ret void
}

attributes #0 = { "keep-me" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.c", directory: "./")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
