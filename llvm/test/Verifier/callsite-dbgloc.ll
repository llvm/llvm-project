; RUN: llvm-as %s -o %t 2>&1 | FileCheck %s
; Created and then edited from
;   extern void i();
;   void h() { i(); }
;   void g() { h(); }
;   void f() { g(); }
;
; Compiling this with inlining runs into the
; "!dbg attachment points at wrong subprogram for function"
; assertion.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; Function Attrs: nounwind ssp uwtable
define void @h() #0 !dbg !7 {
entry:
  call void (...) @i(), !dbg !9
  ret void, !dbg !10
}

; Function Attrs: nounwind ssp uwtable
define weak void @j() #0 !dbg !17 {
entry:
  call void (...) @i(), !dbg !18
  ret void, !dbg !19
}

; Function Attrs: nounwind ssp uwtable
define linkonce_odr void @k() #0 !dbg !20 {
entry:
  call void (...) @i(), !dbg !21
  ret void, !dbg !22
}

declare !dbg !23 void @i(...) #1

; Function Attrs: nounwind ssp uwtable
define void @g() #0 !dbg !11 {
entry:
; Manually removed !dbg.
; CHECK: inlinable function call in a function with debug info must have a !dbg location
; CHECK: @h()
  call void @h()
; CHECK-NOT: inlinable function call in a function with debug info must have a !dbg location
; CHECK-NOT: @j()
  call void @j()
; CHECK: inlinable function call in a function with debug info must have a !dbg location
; CHECK: @k()
  call void @k()
; CHECK-NOT: inlinable function call in a function with debug info must have a !dbg location
; CHECK-NOT: @i()
  call void (...) @i()
  ret void, !dbg !13
}

; Function Attrs: nounwind ssp uwtable
define void @f() #0 !dbg !14 {
entry:
  call void @g(), !dbg !15
  ret void, !dbg !16
}

attributes #0 = { nounwind ssp uwtable }

; CHECK: warning: ignoring invalid debug info

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 267186)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 (trunk 267186)"}
!7 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 12, scope: !7)
!10 = !DILocation(line: 2, column: 17, scope: !7)
!11 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: false, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 3, column: 12, scope: !11)
!13 = !DILocation(line: 3, column: 17, scope: !11)
!14 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: false, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 4, column: 12, scope: !14)
!16 = !DILocation(line: 4, column: 17, scope: !14)
!17 = distinct !DISubprogram(name: "j", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 4, column: 12, scope: !17)
!19 = !DILocation(line: 4, column: 17, scope: !17)
!20 = distinct !DISubprogram(name: "k", scope: !1, file: !1, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: false, unit: !0, retainedNodes: !2)
!21 = !DILocation(line: 4, column: 12, scope: !20)
!22 = !DILocation(line: 4, column: 17, scope: !20)
!23 = !DISubprogram(name: "i", scope: !1, file: !1, line: 1, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!24 = !DISubroutineType(types: !25)
!25 = !{null}
