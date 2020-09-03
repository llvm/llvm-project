; RUN: opt < %s -codegenprepare -S -mtriple=x86_64-unknown-unknown | FileCheck %s --match-full-lines
  
; Make sure the promoted zext doesn't get a debug location associated.
; CHECK: %promoted = zext i8 %t to i64

define void @patatino(i8* %p, i64* %q, i32 %b, i32* %addr) !dbg !6 {
entry:
  %t = load i8, i8* %p, align 1, !dbg !8
  %zextt = zext i8 %t to i32, !dbg !9
  %add = add nuw i32 %zextt, %b, !dbg !10
  %add2 = add nuw i32 %zextt, 12, !dbg !11
  store i32 %add, i32* %addr, align 4, !dbg !12
  %s = zext i32 %add2 to i64, !dbg !13
  store i64 %s, i64* %q, align 4, !dbg !14
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a2.ll", directory: "/")
!2 = !{}
!3 = !{i32 8}
!4 = !{i32 0}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "patatino", linkageName: "patatino", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 1, column: 1, scope: !6)
!9 = !DILocation(line: 2, column: 1, scope: !6)
!10 = !DILocation(line: 3, column: 1, scope: !6)
!11 = !DILocation(line: 4, column: 1, scope: !6)
!12 = !DILocation(line: 5, column: 1, scope: !6)
!13 = !DILocation(line: 6, column: 1, scope: !6)
!14 = !DILocation(line: 7, column: 1, scope: !6)
!15 = !DILocation(line: 8, column: 1, scope: !6)
