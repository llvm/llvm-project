; RUN: opt < %s -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S -hoist-common-insts=true | FileCheck %s

; TODO: Track the acutal DebugLoc of the hoisted instruction when no-line
; DebugLoc is supported (https://reviews.llvm.org/D24180)

; Checks if the debug info for hoisted "x = i" is removed and
; the debug info for hoisted "bar()" is set as line 0
; int x;
; void bar();
; void baz();
;
; void foo(int i) {
;   if (i == 0) {
;     x = i;
;     bar();
;   } else {
;     x = i;
;     bar();
;     baz();
;   }
; }

target triple = "x86_64-unknown-linux-gnu"

@x = global i32 0, align 4

; Function Attrs: uwtable
define void @_Z3fooi(i32) #0 !dbg !6 {
; CHECK: load i32, i32* %2, align 4, !dbg ![[LOAD:[0-9]+]], !tbaa
; CHECK: store i32 %5, i32* @x, align 4, !dbg ![[BAR:[0-9]+]], !tbaa
; CHECK: call void @_Z3barv(), !dbg ![[BAR]]
; CHECK: call void @_Z3bazv(), !dbg ![[BAZ:[0-9]+]]
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4, !tbaa !8
  %3 = load i32, i32* %2, align 4, !dbg !12, !tbaa !8
  %4 = icmp eq i32 %3, 0, !dbg !13
  br i1 %4, label %5, label %7, !dbg !12

; <label>:5:
  %6 = load i32, i32* %2, align 4, !dbg !14, !tbaa !8
  store i32 %6, i32* @x, align 4, !dbg !15, !tbaa !8
  call void @_Z3barv(), !dbg !16
  br label %9, !dbg !17

; <label>:7:
  %8 = load i32, i32* %2, align 4, !dbg !18, !tbaa !8
  store i32 %8, i32* @x, align 4, !dbg !19, !tbaa !8
  call void @_Z3barv(), !dbg !20
  call void @_Z3bazv(), !dbg !21
  br label %9

; <label>:9:
  ret void, !dbg !21
}

declare void @_Z3barv() #1

declare void @_Z3bazv() #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

; CHECK: ![[LOAD]] = !DILocation(line: 6, column: 7
; CHECK: ![[BAR]] = !DILocation(line: 0
; CHECK: ![[BAZ]] = !DILocation(line: 12, column: 5
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "a", directory: "b/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{}
!6 = distinct !DISubprogram(unit: !0)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !DILocation(line: 6, column: 7, scope: !6)
!13 = !DILocation(line: 6, column: 9, scope: !6)
!14 = !DILocation(line: 7, column: 9, scope: !6)
!15 = !DILocation(line: 7, column: 7, scope: !6)
!16 = !DILocation(line: 8, column: 5, scope: !6)
!17 = !DILocation(line: 9, column: 3, scope: !6)
!18 = !DILocation(line: 10, column: 9, scope: !6)
!19 = !DILocation(line: 10, column: 7, scope: !6)
!20 = !DILocation(line: 11, column: 5, scope: !6)
!21 = !DILocation(line: 12, column: 5, scope: !6)
!22 = !DILocation(line: 14, column: 1, scope: !6)
