; Do setup work for all below tests: generate bitcode and combined index
; RUN: llvm-as -function-summary %s -o %t.bc
; RUN: llvm-as -function-summary %p/Inputs/funcimport_debug.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now and confirm that metadata is linked for imported function.
; RUN: opt -function-import -summary-file %t3.thinlto.bc %s -S | FileCheck %s

; CHECK: define available_externally void @func()

; Check that we have exactly two subprograms (that func's subprogram wasn't
; linked more than once for example), and that they are connected to
; the subprogram list on a compute unit.
; CHECK: !{{[0-9]+}} = distinct !DICompileUnit({{.*}} subprograms: ![[SPs1:[0-9]+]]
; CHECK: ![[SPs1]] = !{![[MAINSP:[0-9]+]]}
; CHECK: ![[MAINSP]] = distinct !DISubprogram(name: "main"
; CHECK: !{{[0-9]+}} = distinct !DICompileUnit({{.*}} subprograms: ![[SPs2:[0-9]+]]
; CHECK-NOT: ![[SPs2]] = !{{{.*}}null{{.*}}}
; CHECK: ![[SPs2]] = !{![[FUNCSP:[0-9]+]]}
; CHECK: ![[FUNCSP]] = distinct !DISubprogram(name: "func"
; CHECK-NOT: distinct !DISubprogram

; ModuleID = 'funcimport_debug.o'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !4 {
entry:
  call void (...) @func(), !dbg !11
  ret i32 0, !dbg !12
}

declare void @func(...) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 255685) (llvm/trunk 255682)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "funcimport_debug.c", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 255685) (llvm/trunk 255682)"}
!11 = !DILocation(line: 3, column: 3, scope: !4)
!12 = !DILocation(line: 4, column: 1, scope: !4)
