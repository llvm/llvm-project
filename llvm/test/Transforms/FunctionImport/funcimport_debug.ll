; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_debug.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now and confirm that metadata is linked for imported function.
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -S | FileCheck %s

; CHECK: define available_externally void @func()

; Check that we have exactly two subprograms (that func's subprogram wasn't
; linked more than once for example), and that they are connected to
; the correct compile unit.
; CHECK: ![[CU1:[0-9]+]] = distinct !DICompileUnit(
; CHECK: ![[CU2:[0-9]+]] = distinct !DICompileUnit(
; CHECK: distinct !DISubprogram(name: "main"
; CHECK-SAME:                   unit: ![[CU1]]
; CHECK: distinct !DISubprogram(name: "func"
; CHECK-SAME:                   unit: ![[CU2]]
; CHECK-NOT: distinct !DISubprogram

; ModuleID = 'funcimport_debug.o'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !4 {
entry:
  call void (...) @func(), !dbg !11
  ret i32 0, !dbg !12
}

declare void @func(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 255685) (llvm/trunk 255682)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "funcimport_debug.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 255685) (llvm/trunk 255682)"}
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DILocation(line: 4, column: 1, scope: !4)
