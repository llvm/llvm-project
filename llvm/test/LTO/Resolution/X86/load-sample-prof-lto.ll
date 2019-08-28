; Test that LTO pipeline loads profile.
;
; RUN: opt < %s -o %t.bc

; Run the old pm LTO pipeline.
; RUN: llvm-lto2 run -o %t.out %t.bc -save-temps \
; RUN:   -r %t.bc,foo,px -r %t.bc,bar,x \
; RUN:   -lto-sample-profile-file=%S/Inputs/load-sample-prof.prof
; RUN: llvm-dis %t.out.0.4.opt.bc -o - | FileCheck %s

; Run the new pm LTO pipeline.
; RUN: llvm-lto2 run -o %t.out %t.bc -save-temps -use-new-pm \
; RUN:   -r %t.bc,foo,px -r %t.bc,bar,x \
; RUN:   -lto-sample-profile-file=%S/Inputs/load-sample-prof.prof
; RUN: llvm-dis %t.out.0.4.opt.bc -o - | FileCheck %s

; Make sure profile information is attached.
; CHECK: !prof

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() local_unnamed_addr !dbg !7 {
entry:
  tail call void @bar(), !dbg !10
  ret void, !dbg !11
}

declare void @bar() local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 "}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 4, column: 5, scope: !7)
!11 = !DILocation(line: 5, column: 1, scope: !7)
