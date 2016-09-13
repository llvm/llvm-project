; Use llvm-as to verify each module
; RUN: llvm-as %s -o %t1.bc
; RUN: llvm-as %p/Inputs/odr.ll -o %t2.bc
; Check that we can link it
; RUN: llvm-link %t1.bc %t2.bc -o %t
@bar = global i64 0, align 8, !dbg !6

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, retainedTypes: !2, globals: !5)
!1 = !DIFile(filename: "a", directory: "")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !4, file: !1, identifier: "zed")
!4 = distinct !DISubprogram(name: "b", scope: null, isLocal: false, isDefinition: true, isOptimized: false, unit: !0)
!5 = !{!6}
!6 = distinct !DIGlobalVariable(name: "c", scope: null, isLocal: false, isDefinition: true)
!7 = !{i32 2, !"Debug Info Version", i32 3}
