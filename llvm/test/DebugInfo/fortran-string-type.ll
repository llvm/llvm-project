;; Test for !DIStringType. This DI is used to construct a Fortran CHARACTER
;; intrinsic type, either with a compile-time constant LEN type parameter or
;; when LEN is a dynamic parameter as in a deferred-length CHARACTER.  (See
;; section 7.4.4 of the Fortran 2018 standard.)

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; CHECK: !DIStringType(name: "character(*)", stringLength: !{{[0-9]+}}, stringLengthExpression: !DIExpression(), size: 32)
; CHECK: !DIStringType(name: "character(10)", size: 80, align: 8)
; CHECK: !DIBasicType(tag: DW_TAG_string_type

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: "Flang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortran-string-type.f", directory: "/")
!4 = !{}
!5 = !{!6, !9, !12, !13}
!6 = !DIStringType(name: "character(*)", stringLength: !7, stringLengthExpression: !DIExpression(), size: 32)
!7 = !DILocalVariable(arg: 2, scope: !8, file: !3, line: 256, type: !11, flags: DIFlagArtificial)
!8 = distinct !DISubprogram(name: "subprgm", scope: !2, file: !3, line: 256, type: !9, isLocal: false, isDefinition: true, scopeLine: 256, isOptimized: false, unit: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !6, !11}
!11 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!12 = !DIStringType(name: "character(10)", size: 80, align: 8)
!13 = !DIBasicType(tag: DW_TAG_string_type, name: "character")
