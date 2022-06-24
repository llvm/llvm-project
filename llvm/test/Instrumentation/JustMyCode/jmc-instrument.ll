; RUN: opt -jmc-instrument -mtriple=x86_64-pc-windows-msvc  -S < %s | FileCheck %s
; RUN: opt -jmc-instrument -mtriple=aarch64-pc-windows-msvc -S < %s | FileCheck %s
; RUN: opt -jmc-instrument -mtriple=arm-pc-windows-msvc     -S < %s | FileCheck %s

; CHECK: $__JustMyCode_Default = comdat any

; CHECK: @"__7DF23CF5_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !0
; CHECK: @llvm.used = appending global [1 x i8*] [i8* bitcast (void (i8*)* @__JustMyCode_Default to i8*)], section "llvm.metadata"
; CHECK: @"__A8764FDD_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !5
; CHECK: @"__0C712A50_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !9
; CHECK: @"__A3605329_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !12

; CHECK: define void @l1() !dbg !19 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__7DF23CF5_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @l2() !dbg !23 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__7DF23CF5_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w1() !dbg !25 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w2() !dbg !26 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w3() !dbg !28 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w4() !dbg !30 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w5() !dbg !32 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__0C712A50_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w6() !dbg !33 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A3605329_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w7() !dbg !34 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__0C712A50_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @__JustMyCode_Default(i8* noundef %0) unnamed_addr comdat {
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @__CheckForDebuggerJustMyCode(i8* noundef) unnamed_addr

; CHECK: !llvm.linker.options = !{!18}

; CHECK: !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
; CHECK: !1 = distinct !DIGlobalVariable(name: "__7DF23CF5_x@c", scope: !2, file: !3, type: !8, isLocal: true, isDefinition: true)
; CHECK: !2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
; CHECK: !3 = !DIFile(filename: "a/x.c", directory: "/tmp")
; CHECK: !4 = !{!0, !5, !9, !12}
; CHECK: !5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
; CHECK: !6 = distinct !DIGlobalVariable(name: "__A8764FDD_x@c", scope: !2, file: !7, type: !8, isLocal: true, isDefinition: true)
; CHECK: !7 = !DIFile(filename: "./x.c", directory: "C:\\\\a\\\\b\\\\")
; CHECK: !8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char, flags: DIFlagArtificial)
; CHECK: !9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
; CHECK: !10 = distinct !DIGlobalVariable(name: "__0C712A50_x@c", scope: !2, file: !11, type: !8, isLocal: true, isDefinition: true)
; CHECK: !11 = !DIFile(filename: "./b/.\\../b/x.c", directory: "a\\d/p")
; CHECK: !12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
; CHECK: !13 = distinct !DIGlobalVariable(name: "__A3605329_x@c", scope: !2, file: !14, type: !8, isLocal: true, isDefinition: true)
; CHECK: !14 = !DIFile(filename: "./b/./../b/x.c", directory: "a/d/p")
; CHECK: !15 = !{i32 2, !"CodeView", i32 1}
; CHECK: !16 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: !17 = !{!"clang"}
; CHECK: !18 = !{!"/alternatename:__CheckForDebuggerJustMyCode=__JustMyCode_Default"}
; CHECK: !19 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !20 = !DISubroutineType(types: !21)
; CHECK: !21 = !{null}
; CHECK: !22 = !{}
; CHECK: !23 = distinct !DISubprogram(name: "f", scope: !24, file: !24, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !24 = !DIFile(filename: "x.c", directory: "/tmp/a")
; CHECK: !25 = distinct !DISubprogram(name: "f", scope: !7, file: !7, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !26 = distinct !DISubprogram(name: "f", scope: !27, file: !27, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !27 = !DIFile(filename: "./b\\x.c", directory: "C:\\\\a\\\\")
; CHECK: !28 = distinct !DISubprogram(name: "f", scope: !29, file: !29, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !29 = !DIFile(filename: "./b/x.c", directory: "C:\\\\a\\\\")
; CHECK: !30 = distinct !DISubprogram(name: "f", scope: !31, file: !31, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !31 = !DIFile(filename: "./b/./../b/x.c", directory: "C:/a")
; CHECK: !32 = distinct !DISubprogram(name: "f", scope: !11, file: !11, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !33 = distinct !DISubprogram(name: "f", scope: !14, file: !14, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !34 = distinct !DISubprogram(name: "f", scope: !35, file: !35, line: 1, type: !20, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
; CHECK: !35 = !DIFile(filename: "./b/./../b\\x.c", directory: "a/d/p")

; All use the same flag
define void @l1() !dbg !20 {
  ret void
}
define void @l2() !dbg !21 {
  ret void
}

; All use the same flag
define void @w1() !dbg !22 {
  ret void
}
define void @w2() !dbg !23 {
  ret void
}
define void @w3() !dbg !24 {
  ret void
}
define void @w4() !dbg !25 {
  ret void
}

; Test relative paths
define void @w5() !dbg !26 {
  ret void
}
define void @w6() !dbg !27 {
  ret void
}
define void @w7() !dbg !28 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a/x.c", directory: "/tmp")
!2 = !DIFile(filename: "x.c", directory: "/tmp/a")
!3 = !DIFile(filename: "./x.c", directory: "C:\\\\a\\\\b\\\\")
!4 = !DIFile(filename: "./b\\x.c", directory: "C:\\\\a\\\\")
!5 = !DIFile(filename: "./b/x.c", directory: "C:\\\\a\\\\")
!6 = !DIFile(filename: "./b/./../b/x.c", directory: "C:/a")
!7 = !DIFile(filename: "./b/.\\../b/x.c", directory: "a\\d/p")
!8 = !DIFile(filename: "./b/./../b/x.c", directory: "a/d/p")
!9 = !DIFile(filename: "./b/./../b\\x.c", directory: "a/d/p")
!17 = !{i32 2, !"CodeView", i32 1}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{!"clang"}
!20 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!21 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!22 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!23 = distinct !DISubprogram(name: "f", scope: !4, file: !4, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!24 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!25 = distinct !DISubprogram(name: "f", scope: !6, file: !6, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!26 = distinct !DISubprogram(name: "f", scope: !7, file: !7, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!27 = distinct !DISubprogram(name: "f", scope: !8, file: !8, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!28 = distinct !DISubprogram(name: "f", scope: !9, file: !9, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{null}
!33 = !{}
