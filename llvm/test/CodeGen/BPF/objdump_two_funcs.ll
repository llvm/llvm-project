; RUN: llc -mtriple=bpfel -filetype=obj -o - %s | llvm-objdump -S - | FileCheck %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o - %s | llvm-objdump -S - | FileCheck %s
;
; Source code:
;   __attribute__((section("s1")))
;   int func1(int a) {
;       return a * a;
;   }
;   __attribute__((section("s2")))
;   int func2(int a) {
;       return a * a * a;
;   }
; Compiler flag to generate IR:
;   clang -target bpf -S -gdwarf-5 -gembed-source -emit-llvm -g -O2 bug.c

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @func1(i32 %a) local_unnamed_addr #0 section "s1" !dbg !7 {
entry:
; CHECK: <func1>:
  call void @llvm.dbg.value(metadata i32 %a, metadata !12, metadata !DIExpression()), !dbg !13
  %mul = mul nsw i32 %a, %a, !dbg !14
  ret i32 %mul, !dbg !15
; CHECK: ; return a * a;
}

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @func2(i32 %a) local_unnamed_addr #0 section "s2" !dbg !16 {
entry:
; CHECK: <func2>:
  call void @llvm.dbg.value(metadata i32 %a, metadata !18, metadata !DIExpression()), !dbg !19
  %mul = mul nsw i32 %a, %a, !dbg !20
  %mul1 = mul nsw i32 %mul, %a, !dbg !21
  ret i32 %mul1, !dbg !22
; CHECK: ; return a * a * a;
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 366422) (llvm/trunk 366423)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "bug.c", directory: "/tmp/home/yhs/work/tests/llvm/reloc", checksumkind: CSK_MD5, checksum: "c7c9938d4e6989ca33db748213aab194", source: "__attribute__((section(\22s1\22)))\0Aint func1(int a) {\0A    return a * a;\0A}\0A__attribute__((section(\22s2\22)))\0Aint func2(int a) {\0A    return a * a * a;\0A}\0A")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 366422) (llvm/trunk 366423)"}
!7 = distinct !DISubprogram(name: "func1", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocation(line: 3, column: 14, scope: !7)
!15 = !DILocation(line: 3, column: 5, scope: !7)
!16 = distinct !DISubprogram(name: "func2", scope: !1, file: !1, line: 6, type: !8, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !17)
!17 = !{!18}
!18 = !DILocalVariable(name: "a", arg: 1, scope: !16, file: !1, line: 6, type: !10)
!19 = !DILocation(line: 0, scope: !16)
!20 = !DILocation(line: 7, column: 14, scope: !16)
!21 = !DILocation(line: 7, column: 18, scope: !16)
!22 = !DILocation(line: 7, column: 5, scope: !16)
