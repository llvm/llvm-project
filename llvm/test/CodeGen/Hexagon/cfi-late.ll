; RUN: llc -march=hexagon -enable-misched=false < %s | FileCheck %s
; This testcase causes the scheduler to crash for some reason. Disable
; it for now.

target datalayout = "e-m:e-p:32:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon-unknown--elf"

; Check that allocframe was packetized with the two adds.
; CHECK: foo:
; CHECK: {
; CHECK-DAG: allocframe
; CHECK-DAG: add
; CHECK-DAG: add
; CHECK: }
; CHECK: dealloc_return
; CHECK: }

; Function Attrs: nounwind
define i32 @foo(i32 %x, i32 %y) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !9, metadata !14), !dbg !15
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !10, metadata !14), !dbg !16
  %add = add nsw i32 %x, 1, !dbg !17
  %add1 = add nsw i32 %y, 1, !dbg !18
  %call = tail call i32 @bar(i32 %add, i32 %add1) #3, !dbg !19
  %add2 = add nsw i32 %call, 1, !dbg !20
  ret i32 %add2, !dbg !21
}

declare i32 @bar(i32, i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv4" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv4" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (http://llvm.org/git/clang.git 15506a21305e212c406f980ed9b6b1bac785df56)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "cfi-late.c", directory: "/test")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10}
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 3, type: !7)
!10 = !DILocalVariable(name: "y", arg: 2, scope: !4, file: !1, line: 3, type: !7)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.8.0 (http://llvm.org/git/clang.git 15506a21305e212c406f980ed9b6b1bac785df56)"}
!14 = !DIExpression()
!15 = !DILocation(line: 3, column: 13, scope: !4)
!16 = !DILocation(line: 3, column: 20, scope: !4)
!17 = !DILocation(line: 4, column: 15, scope: !4)
!18 = !DILocation(line: 4, column: 20, scope: !4)
!19 = !DILocation(line: 4, column: 10, scope: !4)
!20 = !DILocation(line: 4, column: 24, scope: !4)
!21 = !DILocation(line: 4, column: 3, scope: !4)
