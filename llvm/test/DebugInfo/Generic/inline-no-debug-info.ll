; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

; This was generated from the following source:
; int a, b;
; __attribute__((__always_inline__)) static void callee2() { b = 2; }
; __attribute__((__nodebug__)) void callee() { a = 1; callee2(); }
; void caller() { callee(); }
; by running
;   clang -S test.c -emit-llvm -O1 -gline-tables-only -fno-strict-aliasing

; CHECK-LABEL: @caller(

; This instruction did not have a !dbg metadata in the callee.
; CHECK: store i32 1, {{.*}}, !dbg [[A:!.*]]

; This instruction came from callee with a !dbg metadata.
; CHECK: store i32 2, {{.*}}, !dbg [[B:!.*]]

; The remaining instruction from the caller.
; CHECK: ret void, !dbg [[A]]

; Debug location of the code in caller() and of the inlined code that did not
; have any debug location before.
; CHECK-DAG: [[A]] = !DILocation(line: 4, scope: !{{[0-9]+}})

; Debug location of the inlined code.
; CHECK-DAG: [[B]] = !DILocation(line: 2, scope: !{{[0-9]+}}, inlinedAt: [[A_INL:![0-9]*]])
; CHECK-DAG: [[A_INL]] = distinct !DILocation(line: 4, scope: !{{[0-9]+}})


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i32 0, align 4
@b = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @callee() #0 {
entry:
  store i32 1, i32* @a, align 4
  store i32 2, i32* @b, align 4, !dbg !11
  ret void
}

; Function Attrs: nounwind uwtable
define void @caller() #0 !dbg !4 {
entry:
  tail call void @callee(), !dbg !12
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (210174)", isOptimized: true, emissionKind: LineTablesOnly, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "/code/llvm/build0")
!2 = !{}
!4 = distinct !DISubprogram(name: "caller", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "test.c", directory: "/code/llvm/build0")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "callee2", line: 2, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5.0 (210174)"}
!11 = !DILocation(line: 2, scope: !7)
!12 = !DILocation(line: 4, scope: !4)
