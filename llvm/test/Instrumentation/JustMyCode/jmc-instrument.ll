; RUN: opt -jmc-instrument -mtriple=x86_64-pc-windows-msvc  -S < %s | FileCheck %s
; RUN: opt -jmc-instrument -mtriple=aarch64-pc-windows-msvc -S < %s | FileCheck %s
; RUN: opt -jmc-instrument -mtriple=arm-pc-windows-msvc     -S < %s | FileCheck %s

; CHECK: $__JustMyCode_Default = comdat any

; CHECK: @"__7DF23CF5_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !0
; CHECK: @"__A85D9D03_x@c" = internal unnamed_addr global i8 1, section ".msvcjmc", align 1, !dbg !5
; CHECK: @llvm.used = appending global [1 x i8*] [i8* bitcast (void (i8*)* @__JustMyCode_Default to i8*)], section "llvm.metadata"

; CHECK: define void @l1() !dbg !13 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__7DF23CF5_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @l2() !dbg !17 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__7DF23CF5_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w1() !dbg !19 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A85D9D03_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w2() !dbg !20 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A85D9D03_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w3() !dbg !22 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A85D9D03_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w4() !dbg !24 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(i8* noundef @"__A85D9D03_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @__CheckForDebuggerJustMyCode(i8* noundef) unnamed_addr

; CHECK: define void @__JustMyCode_Default(i8* noundef %0) unnamed_addr comdat {
; CHECK:   ret void
; CHECK: }

; CHECK: !llvm.linker.options = !{!12}

; CHECK: !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
; CHECK: !1 = distinct !DIGlobalVariable(name: "__7DF23CF5_x@c", scope: !2, file: !3, type: !8, isLocal: true, isDefinition: true)
; CHECK: !2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
; CHECK: !3 = !DIFile(filename: "a/x.c", directory: "/tmp")
; CHECK: !4 = !{!0, !5}
; CHECK: !5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
; CHECK: !6 = distinct !DIGlobalVariable(name: "__A85D9D03_x@c", scope: !2, file: !7, type: !8, isLocal: true, isDefinition: true)
; CHECK: !7 = !DIFile(filename: "./x.c", directory: "C:\\\\a\\\\b\\\\")
; CHECK: !8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char, flags: DIFlagArtificial)
; CHECK: !9 = !{i32 2, !"CodeView", i32 1}
; CHECK: !10 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: !11 = !{!"clang"}
; CHECK: !12 = !{!"/alternatename:__CheckForDebuggerJustMyCode=__JustMyCode_Default"}
; CHECK: !13 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
; CHECK: !14 = !DISubroutineType(types: !15)
; CHECK: !15 = !{null}
; CHECK: !16 = !{}
; CHECK: !17 = distinct !DISubprogram(name: "f", scope: !18, file: !18, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
; CHECK: !18 = !DIFile(filename: "x.c", directory: "/tmp/a")
; CHECK: !19 = distinct !DISubprogram(name: "f", scope: !7, file: !7, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
; CHECK: !20 = distinct !DISubprogram(name: "f", scope: !21, file: !21, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
; CHECK: !21 = !DIFile(filename: "./b\\x.c", directory: "C:\\\\a\\\\")
; CHECK: !22 = distinct !DISubprogram(name: "f", scope: !23, file: !23, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
; CHECK: !23 = !DIFile(filename: "./b/x.c", directory: "C:\\\\a\\\\")
; CHECK: !24 = distinct !DISubprogram(name: "f", scope: !25, file: !25, line: 1, type: !14, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !16)
; CHECK: !25 = !DIFile(filename: "./b/./../b/x.c", directory: "C:\\\\a")

; All use the same flag
define void @l1() !dbg !10 {
  ret void
}
define void @l2() !dbg !11 {
  ret void
}

; All use the same flag
define void @w1() !dbg !12 {
  ret void
}
define void @w2() !dbg !13 {
  ret void
}
define void @w3() !dbg !14 {
  ret void
}
define void @w4() !dbg !15 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a/x.c", directory: "/tmp")
!2 = !DIFile(filename: "x.c", directory: "/tmp/a")
!3 = !DIFile(filename: "./x.c", directory: "C:\\\\a\\\\b\\\\")
!4 = !DIFile(filename: "./b\\x.c", directory: "C:\\\\a\\\\")
!5 = !DIFile(filename: "./b/x.c", directory: "C:\\\\a\\\\")
!6 = !DIFile(filename: "./b/./../b/x.c", directory: "C:\\\\a")
!7 = !{i32 2, !"CodeView", i32 1}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!11 = distinct !DISubprogram(name: "f", scope: !2, file: !2, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!12 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!13 = distinct !DISubprogram(name: "f", scope: !4, file: !4, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!14 = distinct !DISubprogram(name: "f", scope: !5, file: !5, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!15 = distinct !DISubprogram(name: "f", scope: !6, file: !6, line: 1, type: !31, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
!31 = !DISubroutineType(types: !32)
!32 = !{null}
!33 = !{}
