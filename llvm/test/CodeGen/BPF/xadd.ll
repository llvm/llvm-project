; RUN: not llc -march=bpfel < %s 2>&1 | FileCheck %s
; RUN: not llc -march=bpfeb < %s 2>&1 | FileCheck %s
; RUN: not llc -march=bpfel -mattr=+alu32 < %s 2>&1 | FileCheck %s
; RUN: not llc -march=bpfeb -mattr=+alu32 < %s 2>&1 | FileCheck %s

; This file is generated with the source command and source
; $ clang -target bpf -O2 -g -S -emit-llvm t.c
; $ cat t.c
; int test(int *ptr) {
;    int r;
;    __sync_fetch_and_add(ptr, 4);
;    r = __sync_fetch_and_add(ptr, 6);
;    return r;
; }

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "bpf"

; Function Attrs: nounwind
define dso_local i32 @test(i32* nocapture %ptr) local_unnamed_addr #0 !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32* %ptr, metadata !13, metadata !DIExpression()), !dbg !15
  %0 = atomicrmw add i32* %ptr, i32 4 seq_cst, !dbg !16
  %1 = atomicrmw add i32* %ptr, i32 6 seq_cst, !dbg !17
; CHECK: line 4: Invalid usage of the XADD return value
  call void @llvm.dbg.value(metadata i32 %1, metadata !14, metadata !DIExpression()), !dbg !18
  ret i32 %1, !dbg !19
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 342605) (llvm/trunk 342612)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/home/yhs/work/tests/llvm/sync/test1")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 342605) (llvm/trunk 342612)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "ptr", arg: 1, scope: !7, file: !1, line: 1, type: !11)
!14 = !DILocalVariable(name: "r", scope: !7, file: !1, line: 2, type: !10)
!15 = !DILocation(line: 1, column: 15, scope: !7)
!16 = !DILocation(line: 3, column: 4, scope: !7)
!17 = !DILocation(line: 4, column: 8, scope: !7)
!18 = !DILocation(line: 2, column: 8, scope: !7)
!19 = !DILocation(line: 5, column: 4, scope: !7)
