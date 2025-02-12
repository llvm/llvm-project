; RUN: opt -passes='annotation-remarks' -pass-remarks-missed='annotation-remarks' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --input-file=%t.opt.yaml %s

define void @test1(ptr %a) !dbg !5 {
entry:
  %a.addr = alloca ptr, align 8, !dbg !11
  call void @llvm.dbg.value(metadata ptr %a.addr, metadata !9, metadata !DIExpression()), !dbg !11
  store ptr null, ptr %a.addr, align 8, !dbg !12, !annotation !13
  store ptr %a, ptr %a.addr, align 8, !dbg !14, !annotation !13
  ret void, !dbg !15
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/llvm-project/llvm/test/Transforms/Util/bounds-safety-annotation-remarks.ll", directory: "/")
!2 = !{i32 4}
!3 = !{i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DILocation(line: 2, column: 1, scope: !5)
!13 = !{!"auto-init"}
!14 = !DILocation(line: 3, column: 1, scope: !5)
!15 = !DILocation(line: 4, column: 1, scope: !5)



; CHECK: --- !Analysis
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AnnotationSummary
; CHECK-NEXT: DebugLoc:        { File: 
; CHECK-NEXT:                    Line: 1, Column: 0 }
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'Annotated '
; CHECK-NEXT:   - count:           '2'
; CHECK-NEXT:   - String:          ' instructions with '
; CHECK-NEXT:   - type:            auto-init
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AutoInitStore
; CHECK-NEXT: DebugLoc:        { File: 
; CHECK-NEXT:                    Line: 2, Column: 1 }
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT:   - String:          "\nStore size: "
; CHECK-NEXT:   - StoreSize:       '8'
; CHECK-NEXT:   - String:          ' bytes.'
; CHECK-NEXT:   - String:          "\n Written Variables: "
; CHECK-NEXT:   - WVarName:        a.addr
; CHECK-NEXT:   - String:          ' ('
; CHECK-NEXT:   - WVarSize:        '8'
; CHECK-NEXT:   - String:          ' bytes)'
; CHECK-NEXT:   - String:          .
; CHECK-NEXT:   - String:          ' Volatile: '
; CHECK-NEXT:   - StoreVolatile:   'false'
; CHECK-NEXT:   - String:          .
; CHECK-NEXT:   - String:          ' Atomic: '
; CHECK-NEXT:   - StoreAtomic:     'false'
; CHECK-NEXT:   - String:          .
; CHECK-NEXT: ...
; CHECK-NEXT: --- !Missed
; CHECK-NEXT: Pass:            annotation-remarks
; CHECK-NEXT: Name:            AutoInitStore
; CHECK-NEXT: DebugLoc:        { File: 
; CHECK-NEXT:                    Line: 3, Column: 1 }
; CHECK-NEXT: Function:        test1
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          Store inserted by -ftrivial-auto-var-init.
; CHECK-NEXT:   - String:          "\nStore size: "
; CHECK-NEXT:   - StoreSize:       '8'
; CHECK-NEXT:   - String:          ' bytes.'
; CHECK-NEXT:   - String:          "\n Written Variables: "
; CHECK-NEXT:   - WVarName:        a.addr
; CHECK-NEXT:   - String:          ' ('
; CHECK-NEXT:   - WVarSize:        '8'
; CHECK-NEXT:   - String:          ' bytes)'
; CHECK-NEXT:   - String:          .
; CHECK-NEXT:   - String:          ' Volatile: '
; CHECK-NEXT:   - StoreVolatile:   'false'
; CHECK-NEXT:   - String:          .
; CHECK-NEXT:   - String:          ' Atomic: '
; CHECK-NEXT:   - StoreAtomic:     'false'
; CHECK-NEXT:   - String:          .
; CHECK-NEXT: ...
