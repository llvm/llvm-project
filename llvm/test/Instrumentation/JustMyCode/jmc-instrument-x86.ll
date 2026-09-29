; RUN: opt -jmc-instrumenter -S < %s | FileCheck %s
; RUN: opt -passes=jmc-instrumenter -S < %s | FileCheck %s

; CHECK: $_JustMyCode_Default = comdat any

; CHECK: @"_A8764FDD_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !0
; CHECK: @llvm.used = appending global [1 x ptr] [ptr @_JustMyCode_Default], section "llvm.metadata"

; CHECK: define void @w1() #0 !dbg !10 {
; CHECK:   call x86_fastcallcc void @__CheckForDebuggerJustMyCode(ptr inreg noundef @"_A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @_JustMyCode_Default(ptr inreg noundef %0) unnamed_addr comdat {
; CHECK:   ret void
; CHECK: }

; CHECK: declare x86_fastcallcc void @__CheckForDebuggerJustMyCode(ptr inreg noundef) unnamed_addr

; CHECK: !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
; CHECK: !1 = distinct !DIGlobalVariable(name: "_A8764FDD_x@c", scope: !2, file: !3, type: !5, isLocal: true, isDefinition: true)
; CHECK: !2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
; CHECK: !3 = !DIFile(filename: "./b/./../b/x.c", directory: "C:\\\\a\\\\")
; CHECK: !4 = !{!0}
; CHECK: !5 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char, flags: DIFlagArtificial)
; CHECK: !6 = !{i32 2, !"CodeView", i32 1}
; CHECK: !7 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: !8 = !{!"clang"}
; CHECK: !9 = !{!"/alternatename:__CheckForDebuggerJustMyCode=_JustMyCode_Default"}
; CHECK: !10 = distinct !DISubprogram(name: "w1", scope: !3, file: !3, line: 1, type: !11, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
; CHECK: !11 = !DISubroutineType(types: !12)

target datalayout = "e-m:x-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc"

define void @w1() #0 !dbg !10 {
  ret void
}

attributes #0 = { "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "./b/./../b/x.c", directory: "C:\\\\a\\\\")
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "w1", scope: !1, file: !1, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{null}
!33 = !{}
