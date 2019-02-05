; RUN: opt -run-twice -verify -S -o - %s | FileCheck %s

; If a module contains a DISubprogram referenced only indirectly from
; instruction-level debug info metadata, but not attached to any Function
; defined within the module, cloning such a module with CloneModule was
; causing a DICompileUnit duplication: it would be moved in indirecty via a
; DISubprogram by DebugInfoFinder (making sure DISubprogram's don't get
; duplicated) first without being explicitly self-mapped within the ValueMap
; shared among CloneFunctionInto calls, and then it would get copied during
; named metadata cloning.
;
; This is to make sure we don't regress on that.

; Derived from the following C-snippet
;
; static int eliminated(int j);
; __attribute__((nodebug)) int nodebug(int k) { return eliminated(k); }
; __attribute__((always_inline)) static int eliminated(int j) { return j * 2; }
;
; compiled with `clang -O1 -g1 -emit-llvm -S`

; CHECK:     DICompileUnit
; CHECK-NOT: DICompileUnit

source_filename = "clone-module.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; Function Attrs: norecurse nounwind readnone ssp uwtable
define i32 @nodebug(i32 %k) local_unnamed_addr #0 {
entry:
  %mul.i = shl nsw i32 %k, 1, !dbg !8
  ret i32 %mul.i
}

attributes #0 = { norecurse nounwind readnone ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm/build/obj")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 7.0.0"}
!8 = !DILocation(line: 3, column: 72, scope: !9)
!9 = distinct !DISubprogram(name: "eliminated", scope: !1, file: !1, line: 3, type: !10, isLocal: true, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !2)
