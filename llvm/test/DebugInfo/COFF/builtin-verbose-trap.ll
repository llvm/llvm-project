; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s

; Original C++ source:
; int done() {
;         __builtin_verbose_trap("STL_FOO", "broken stuff");
; }

; Make sure the call to clang-trap is annotated, even though the associated
; debug location as a line number of 0.

; CHECK:        InlineesSym {
; CHECK-NEXT:     Kind: S_INLINEES (0x1168)
; CHECK-NEXT:     Inlinees [
; CHECK-NEXT:       FuncID: __clang_trap_msg$STL_FOO$broken stuff (0x1002)
; CHECK-NEXT:     ]
; CHECK-NEXT:   }
; CHECK-NEXT:   InlineSiteSym {
; CHECK-NEXT:     Kind: S_INLINESITE (0x114D)
; CHECK-NEXT:     PtrParent: 0x0
; CHECK-NEXT:     PtrEnd: 0x0
; CHECK-NEXT:     Inlinee: __clang_trap_msg$STL_FOO$broken stuff (0x1002)
; CHECK-NEXT:     BinaryAnnotations [
; CHECK-NEXT:       ChangeCodeOffsetAndLineOffset: {CodeOffset: 0x1, LineOffset: 0}
; CHECK-NEXT:       ChangeCodeLength: 0x7
; CHECK-NEXT:     ]
; CHECK-NEXT:   }


target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

define dso_local i32 @done() !dbg !10 {
  %1 = alloca i32
  call void @llvm.trap(), !dbg !14
  %2 = load i32, ptr %1, align 4
  ret i32 %2
}

declare void @llvm.trap()


!llvm.dbg.cu = !{!0}
!llvm.linker.options = !{!2, !3}
!llvm.module.flags = !{!4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.1.8 (taskcluster-cBNpsXlaTmqEnu_aiQ0ElA)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/Users/jrmuizel/bugs/inline", checksumkind: CSK_MD5, checksum: "ebf5ba4ead45cede7df7db37b24dc58d")
!2 = !{!"/DEFAULTLIB:libcmt.lib"}
!3 = !{!"/DEFAULTLIB:oldnames.lib"}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{!"clang version 20.1.8 (taskcluster-cBNpsXlaTmqEnu_aiQ0ElA)"}
!10 = distinct !DISubprogram(name: "done", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 0, scope: !15, inlinedAt: !17)
!15 = distinct !DISubprogram(name: "__clang_trap_msg$STL_FOO$broken stuff", scope: !1, file: !1, type: !16, flags: DIFlagArtificial, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DISubroutineType(types: null)
!17 = !DILocation(line: 2, scope: !10)
