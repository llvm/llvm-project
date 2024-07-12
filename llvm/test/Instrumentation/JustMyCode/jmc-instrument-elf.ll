; RUN: opt -jmc-instrumenter -mtriple=x86_64-unknown-linux-gnu  -S < %s | FileCheck %s
; RUN: opt -passes=jmc-instrumenter -mtriple=x86_64-unknown-linux-gnu  -S < %s | FileCheck %s

; CHECK: @"__7DF23CF5_x@c" = internal unnamed_addr global i8 1, section ".data.just.my.code", align 1, !dbg !0
; CHECK: @"__A8764FDD_x@c" = internal unnamed_addr global i8 1, section ".data.just.my.code", align 1, !dbg !5

; CHECK: define void @l1() !dbg !12 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(ptr noundef @"__7DF23CF5_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @l2() !dbg !16 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(ptr noundef @"__7DF23CF5_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w1() !dbg !18 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(ptr noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w2() !dbg !19 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(ptr noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w3() !dbg !21 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(ptr noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define void @w4() !dbg !23 {
; CHECK:   call void @__CheckForDebuggerJustMyCode(ptr noundef @"__A8764FDD_x@c")
; CHECK:   ret void
; CHECK: }

; CHECK: define weak void @__CheckForDebuggerJustMyCode(ptr noundef %0) unnamed_addr {
; CHECK:   ret void
; CHECK: }

; CHECK: !llvm.dbg.cu = !{!2}
; CHECK: !llvm.module.flags = !{!9, !10}
; CHECK: !llvm.ident = !{!11}

; CHECK: !0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
; CHECK: !1 = distinct !DIGlobalVariable(name: "__7DF23CF5_x@c", scope: !2, file: !3, type: !8, isLocal: true, isDefinition: true)
; CHECK: !2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
; CHECK: !3 = !DIFile(filename: "a/x.c", directory: "/tmp")
; CHECK: !4 = !{!0, !5}
; CHECK: !5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
; CHECK: !6 = distinct !DIGlobalVariable(name: "__A8764FDD_x@c", scope: !2, file: !7, type: !8, isLocal: true, isDefinition: true)
; CHECK: !7 = !DIFile(filename: "./x.c", directory: "C:\\\\a\\\\b\\\\")
; CHECK: !8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char, flags: DIFlagArtificial)
; CHECK: !9 = !{i32 2, !"CodeView", i32 1}
; CHECK: !10 = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: !11 = !{!"clang"}
; CHECK: !12 = distinct !DISubprogram(name: "f", scope: !3, file: !3, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
; CHECK: !13 = !DISubroutineType(types: !14)
; CHECK: !14 = !{null}
; CHECK: !15 = !{}
; CHECK: !16 = distinct !DISubprogram(name: "f", scope: !17, file: !17, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
; CHECK: !17 = !DIFile(filename: "x.c", directory: "/tmp/a")
; CHECK: !18 = distinct !DISubprogram(name: "f", scope: !7, file: !7, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
; CHECK: !19 = distinct !DISubprogram(name: "f", scope: !20, file: !20, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
; CHECK: !20 = !DIFile(filename: "./b\\x.c", directory: "C:\\\\a\\\\")
; CHECK: !21 = distinct !DISubprogram(name: "f", scope: !22, file: !22, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
; CHECK: !22 = !DIFile(filename: "./b/x.c", directory: "C:\\\\a\\\\")
; CHECK: !23 = distinct !DISubprogram(name: "f", scope: !24, file: !24, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
; CHECK: !24 = !DIFile(filename: "./b/./../b/x.c", directory: "C:\\\\a")

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
