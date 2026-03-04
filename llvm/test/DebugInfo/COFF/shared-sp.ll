; RUN: llc -filetype=obj < %s -o - 2>&1 | llvm-readobj - --codeview | FileCheck %s

; Check that when DISubprogram is attached to two functions, CodeView is
; produced correctly.

; CHECK:  Subsection [
; CHECK:    GlobalProcIdSym {
; CHECK:      Kind: S_GPROC32_ID (0x1147)
; CHECK:      FunctionType: foo (0x1002)
; CHECK:      CodeOffset: foo+0x0
; CHECK:      Flags [ (0x80)
; CHECK:        HasOptimizedDebugInfo (0x80)
; CHECK:      ]
; CHECK:      DisplayName: foo
; CHECK:      LinkageName: foo
; CHECK:    }
; CHECK:    LocalSym {
; CHECK:      Kind: S_LOCAL (0x113E)
; CHECK:      Type: int (0x74)
; CHECK:      Flags [ (0x0)
; CHECK:      ]
; CHECK:      VarName: a
; CHECK:    }
; CHECK:    LocalSym {
; CHECK:      Kind: S_LOCAL (0x113E)
; CHECK:      Type: foo::bar (0x1005)
; CHECK:      Flags [ (0x0)
; CHECK:      ]
; CHECK:      VarName: c
; CHECK:    }
; CHECK:    UDTSym {
; CHECK:      Kind: S_UDT (0x1108)
; CHECK:      Type: foo::bar (0x1005)
; CHECK:      UDTName: foo::bar
; CHECK:    }
; CHECK:  ]
; CHECK:  Subsection [
; CHECK:    GlobalProcIdSym {
; CHECK:      Kind: S_GPROC32_ID (0x1147)
; CHECK:      FunctionType: foo (0x1002)
; CHECK:      CodeOffset: foo_clone+0x0
; CHECK:      Flags [ (0x80)
; CHECK:        HasOptimizedDebugInfo (0x80)
; CHECK:      ]
; CHECK:      DisplayName: foo
; CHECK:      LinkageName: foo_clone
; CHECK:    }
; CHECK:    LocalSym {
; CHECK:      Kind: S_LOCAL (0x113E)
; CHECK:      Type: int (0x74)
; CHECK:      Flags [ (0x0)
; CHECK:      ]
; CHECK:      VarName: a
; CHECK:    }
; CHECK:    LocalSym {
; CHECK:      Kind: S_LOCAL (0x113E)
; CHECK:      Type: foo::bar (0x1005)
; CHECK:      Flags [ (0x0)
; CHECK:      ]
; CHECK:      VarName: c
; CHECK:    }
; CHECK:    UDTSym {
; CHECK:      Kind: S_UDT (0x1108)
; CHECK:      Type: foo::bar (0x1005)
; CHECK:      UDTName: foo::bar
; CHECK:    }
; CHECK:  ]

; ModuleID = 'shared-sp.ll'
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.29.30133"

!0 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !{})
!1 = !DIFile(filename: "example.c", directory: "/")
!2 = !DISubroutineType(types: !3)
!3 = !{!5}
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; Local variable.
!10 = !DILocalVariable(name: "a", scope: !0, file: !1, line: 2, type: !5)

; DICompositeType local to foo.
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", scope: !0, file: !1, line: 2, size: 32, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !11, file: !1, line: 2, baseType: !5, size: 32)

; Local variable of type struct bar, local to foo.
!14 = !DILocalVariable(name: "c", scope: !0, file: !1, line: 2, type: !11)

!101 = !DILocation(line: 2, column: 5, scope: !0)
!102 = !DILocation(line: 3, column: 1, scope: !0)
!103 = !DILocation(line: 2, column: 12, scope: !0)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!29, !30}

!29 = !{i32 2, !"CodeView", i32 1}
!30 = !{i32 2, !"Debug Info Version", i32 3}

define i32 @foo() !dbg !0 {
entry:
  ; Local variable 'a' debug info.
  %a.addr = alloca i32, align 4, !dbg !101
    #dbg_declare(ptr %a.addr, !10, !DIExpression(), !101)
  store i32 42, ptr %a.addr, align 4, !dbg !101

  ; Local variable 'c' (struct bar) debug info.
  %c.addr = alloca %struct.bar, align 4, !dbg !103
    #dbg_declare(ptr %c.addr, !14, !DIExpression(), !103)

  ret i32 42, !dbg !102
}

define i32 @foo_clone() !dbg !0 {
entry:
  ; Local variable 'a' debug info.
  %a.addr = alloca i32, align 4, !dbg !101
    #dbg_declare(ptr %a.addr, !10, !DIExpression(), !101)
  store i32 42, ptr %a.addr, align 4, !dbg !101

  ; Local variable 'c' (struct bar) debug info.
  %c.addr = alloca %struct.bar, align 4, !dbg !103
    #dbg_declare(ptr %c.addr, !14, !DIExpression(), !103)

  ret i32 42, !dbg !102
}

%struct.bar = type { i32 }
