; RUN: llc %s --filetype=obj -o %t.bc
; RUN: llvm-objcopy --dump-section=DXIL=%t.dxil %t.bc
; RUN: llvm-dis %t.dxil -o - | FileCheck %s --check-prefix=DXIL-DIS

;; Check that all the debug info is properly stripped from DXIL part
; DXIL-DIS: define void @foo
; DXIL-DIS-NOT: #dbg
; DXIL-DIS-NOT: !llvm.dbg.cu
; DXIL-DIS-NOT: !dx.source
; DXIL-DIS-NOT: !DICompileUnit
; DXIL-DIS-NOT: !DIFile
; DXIL-DIS-NOT: !"Dwarf Version"
; DXIL-DIS-NOT: !"Debug Info Version"
; DXIL-DIS-NOT: "foo.hlsl"
; DXIL-DIS-NOT: !DISubprogram
; DXIL-DIS-NOT: !DISubroutineType
; DXIL-DIS-NOT: !DILabel
; DXIL-DIS-NOT: !DIBasicType
; DXIL-DIS-NOT: !DILocation
; DXIL-DIS-NOT: !DILocalVariable
; DXIL-DIS-NOT: !DIGlobalVariable

target triple = "dxil-pc-shadermodel6.3-library"

@g = global i32 0, align 4, !dbg !21

define void @foo(i32 %i) !dbg !4 {
entry:
  %a = alloca i32, align 4
    #dbg_declare(ptr %a, !17, !DIExpression(), !18)
    #dbg_assign(ptr %a, !17, !DIExpression(), !19, ptr poison, !DIExpression(), !18)

    #dbg_value(i32 %i, !9, !DIExpression(), !11)
  br label %label

label:
    #dbg_label(!8, !12)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!dx.source.contents = !{!13}
!dx.source.defines = !{!14}
!dx.source.mainFileName = !{!15}
!dx.source.args = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !20)
!1 = !DIFile(filename: "foo.hlsl", directory: "")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = !DILabel(scope: !4, name: "label", file: !1, line: 2, column: 1)
!9 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 4, type: !10)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 1, scope: !4)
!12 = !DILocation(line: 33, column: 1, scope: !4)
!13 = !{!"foo.hlsl", !"static int g; export void foo(int i) { int a; return; }"}
!14 = !{}
!15 = !{!"foo.hlsl"}
!16 = !{!"--driver-mode=dxc", !"-g", !"-T", !"lib_6_3", !"foo.hlsl"}
!17 = !DILocalVariable(name: "a", arg: 2, scope: !4, file: !1, line: 1, type: !10)
!18 = !DILocation(line: 1, column: 30, scope: !4)
!19 = distinct !DIAssignID()
!20 = !{!21}
!21 = !DIGlobalVariableExpression(var: !22, expr: !DIExpression())
!22 = distinct !DIGlobalVariable(name: "g", scope: !0, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true)
