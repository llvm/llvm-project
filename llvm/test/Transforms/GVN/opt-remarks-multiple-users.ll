; RUN: opt < %s -passes=gvn -o /dev/null  -pass-remarks-output=%t -S
; RUN: cat %t | FileCheck %s

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: DebugLoc: { File: '/tmp/s.c', Line: 2, Column: 2 }
; CHECK-NEXT: Function:        multipleUsers
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     store
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 1, Column: 1 }
; CHECK-NEXT: ...
; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: DebugLoc: { File: '/tmp/s.c', Line: 4, Column: 4 }
; CHECK-NEXT: Function:        multipleUsers
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     load
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 2, Column: 2 }
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 3, Column: 3 }
; CHECK-NEXT: ...

; ModuleID = 'bugpoint-reduced-simplified.bc'
source_filename = "gvn-test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Tests that a last clobbering use can be determined even in the presence of
; multiple users, given that one of them lies on a path between every other
; potentially clobbering use and the load.

define dso_local void @multipleUsers(ptr %a, i32 %b) local_unnamed_addr #0 {
entry:
  store i32 %b, ptr %a, align 4
  tail call void @clobberingFunc() #1, !dbg !10
  %0 = load i32, ptr %a, align 4, !dbg !11
  tail call void @clobberingFunc() #1, !dbg !12
  %1 = load i32, ptr %a, align 4, !dbg !13
  %add2 = add nsw i32 %1, %0
  ret void
}

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: DebugLoc: { File: '/tmp/s.c', Line: 2, Column: 2 }
; CHECK-NEXT: Function:        multipleUsers2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     store
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 1, Column: 1 }
; CHECK-NEXT: ...
; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            gvn
; CHECK-NEXT: Name:            LoadClobbered
; CHECK-NEXT: DebugLoc: { File: '/tmp/s.c', Line: 4, Column: 4 }
; CHECK-NEXT: Function:        multipleUsers2
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'load of type '
; CHECK-NEXT:   - Type:            i32
; CHECK-NEXT:   - String:          ' not eliminated'
; CHECK-NEXT:   - String:          ' in favor of '
; CHECK-NEXT:   - OtherAccess:     load
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 2, Column: 2 }
; CHECK-NEXT:   - String:          ' because it is clobbered by '
; CHECK-NEXT:   - ClobberedBy:     call
; CHECK-NEXT:     DebugLoc: { File: '/tmp/s.c', Line: 3, Column: 3 }
; CHECK-NEXT: ...

; Ignore uses in other functions

define dso_local void @multipleUsers2(i32 %b) local_unnamed_addr #0 {
entry:
  store i32 %b, ptr @g, align 4
  tail call void @clobberingFunc() #1, !dbg !15
  %0 = load i32, ptr @g, align 4, !dbg !16
  tail call void @clobberingFunc() #1, !dbg !17
  %1 = load i32, ptr @g, align 4, !dbg !18
  %add3 = add nsw i32 %1, %0
  ret void
}

declare dso_local void @clobberingFunc() local_unnamed_addr #0

@g = external global i32

define dso_local void @globalUser(i32 %b) local_unnamed_addr #0 {
entry:
  store i32 %b, ptr @g, align 4
  ret void
}


attributes #0 = { "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!4, !5, !6}
!llvm.ident = !{!0}

!0 = !{!"clang version 10.0.0 (git@github.com:llvm/llvm-project.git a2f6ae9abffcba260c22bb235879f0576bf3b783)"}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0 (trunk 282540) (llvm/trunk 282542)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !3)
!2 = !DIFile(filename: "/tmp/s.c", directory: "/tmp")
!3 = !{}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "multipleUsers", scope: !2, file: !2, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !1, retainedNodes: !3)
!9 = !DISubroutineType(types: !3)
!10 = !DILocation(line: 1, column: 1, scope: !8)
!11 = !DILocation(line: 2, column: 2, scope: !8)
!12 = !DILocation(line: 3, column: 3, scope: !8)
!13 = !DILocation(line: 4, column: 4, scope: !8)
!14 = distinct !DISubprogram(name: "multipleUsers2", scope: !2, file: !2, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !1, retainedNodes: !3)
!15 = !DILocation(line: 1, column: 1, scope: !14)
!16 = !DILocation(line: 2, column: 2, scope: !14)
!17 = !DILocation(line: 3, column: 3, scope: !14)
!18 = !DILocation(line: 4, column: 4, scope: !14)
