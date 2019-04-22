; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s

; Check that the "inlinedAt" attribute of a DILocation references another
; DILocation that is marked as distinct.  Note that the checks for distinct
; DILocations do not include the column number as SPIR-V does not allow for
; representing this info.

; Built with clang -O -g from the source:
; bool f();
; inline __attribute__((always_inline)) int f1() {
;   if (bool b = f())
;     return 1;
;   return 0;
; }
;
; inline __attribute__((always_inline)) int f2() {
;   if (int i = f1())
;     return 3;
;   return 4;
; }
;
; int main() {
;   f2();
; }

source_filename = "test/DebugInfo/Generic/inlined-locations.ll"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: norecurse
define dso_local i32 @main() local_unnamed_addr #0 !dbg !7 {
  %1 = tail call spir_func zeroext i1 @_Z1fv(), !dbg !11
  call void @llvm.dbg.value(metadata i8 undef, metadata !16, metadata !DIExpression()), !dbg !24
  ret i32 0, !dbg !25
}

declare dso_local spir_func zeroext i1 @_Z1fv() local_unnamed_addr #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "inlined-locations.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 "}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 14, type: !8, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 3, column: 16, scope: !12, inlinedAt: !18)
; CHECK: !{{.*}} = !DILocation(line: 3, column: 16, scope: !{{.*}}, inlinedAt: ![[loc1:[0-9]+]])
!12 = distinct !DILexicalBlock(scope: !13, file: !1, line: 3, column: 12)
!13 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 2, type: !14, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!14 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !9)
!15 = !{!16}
!16 = !DILocalVariable(name: "b", scope: !12, file: !1, line: 3, type: !17)
!17 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!18 = distinct !DILocation(line: 9, column: 15, scope: !19, inlinedAt: !23)
; CHECK: ![[loc1]] = distinct !DILocation(line: 9, scope: !{{.*}}, inlinedAt: ![[loc2:[0-9]+]])
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 9, column: 11)
!20 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 8, type: !14, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!21 = !{!22}
!22 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 9, type: !10)
!23 = distinct !DILocation(line: 15, column: 3, scope: !7)
; CHECK: ![[loc2]] = distinct !DILocation(line: 15, scope: !{{.*}})
!24 = !DILocation(line: 3, column: 12, scope: !12, inlinedAt: !18)
; CHECK: !{{.*}} = !DILocation(line: 3, column: 12, scope: !{{.*}}, inlinedAt: ![[loc1]])
!25 = !DILocation(line: 16, column: 1, scope: !7)
