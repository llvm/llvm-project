; RUN: opt -annotation-remarks -pass-remarks-missed='annotation-remarks' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --input-file=%t.opt.yaml %s
; RUN: opt -passes='annotation-remarks' -pass-remarks-missed='annotation-remarks' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --input-file=%t.opt.yaml %s

; Make sure a suitable location is used for the function start when emitting
; the annotation summary remarks.

; CHECK:      --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: DebugLoc: { File: test.c, Line: 10, Column: 0 }
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '4'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            _remarks1
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: DebugLoc: { File: test.c, Line: 10, Column: 0 }
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '3'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            _remarks2
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: DebugLoc: { File: test.c, Line: 20, Column: 0 }
; CHECK-NEXT: Function:        test2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '2'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            _remarks1
; CHECK-NEXT: ...

define void @test1(float* %a) !dbg !7 {
entry:
  %a.addr = alloca float*, align 8, !dbg !16, !annotation !5
  store float* null, float** %a.addr, align 8, !annotation !6
  store float* %a, float** %a.addr, align 8, !annotation !5
  ret void, !annotation !5
}

define void @test2(float* %a) !dbg !17 {
entry:
  %a.addr = alloca float*, align 8, !annotation !6
  ret void, !dbg !18, !annotation !6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"_remarks1", !"_remarks2"}
!6 = !{!"_remarks1"}
!7 = distinct !DISubprogram(name: "test1", scope: !1, file: !1, line: 11, type: !8, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10, !10, !13}
!10 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 32, align: 32)
!12 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!13 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!16 = !DILocation(line: 400, column: 3, scope: !7)
!17 = distinct !DISubprogram(name: "test2", scope: !1, file: !1, line: 21, type: !8, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!18 = !DILocation(line: 200, column: 3, scope: !17)
