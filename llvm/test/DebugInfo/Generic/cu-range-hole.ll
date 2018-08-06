; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that we emit ranges for this CU since we have a function with and
; without debug info.
; Note: This depends upon the order of output in the .o file. Currently it's
; in order of the output to make sure that the CU has multiple ranges since
; there's a function in the middle. If they were together then it would have
; a single range and no DW_AT_ranges.
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_ranges
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram

; Function Attrs: nounwind uwtable
define i32 @b(i32 %c) #0 !dbg !5 {
entry:
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load i32, i32* %c.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

; Function Attrs: nounwind uwtable
define i32 @a(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32, i32* %b.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @d(i32 %e) #0 !dbg !10 {
entry:
  %e.addr = alloca i32, align 4
  store i32 %e, i32* %e.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %e.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, i32* %e.addr, align 4, !dbg !16
  %add = add nsw i32 %0, 1, !dbg !16
  ret i32 %add, !dbg !16
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0, !0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!11, !12}

!0 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)", isOptimized: false, emissionKind: FullDebug, file: !2, enums: !3, retainedTypes: !3, globals: !3, imports: !3)
!2 = !DIFile(filename: "b.c", directory: "/usr/local/google/home/echristo")
!3 = !{}
!5 = distinct !DISubprogram(name: "b", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !1, scopeLine: 1, file: !2, scope: !6, type: !7, retainedNodes: !3)
!6 = !DIFile(filename: "b.c", directory: "/usr/local/google/home/echristo")
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "d", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !1, scopeLine: 3, file: !2, scope: !6, type: !7, retainedNodes: !3)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !DILocalVariable(name: "c", line: 1, arg: 1, scope: !5, file: !6, type: !9)
!14 = !DILocation(line: 1, scope: !5)
!15 = !DILocalVariable(name: "e", line: 3, arg: 1, scope: !10, file: !6, type: !9)
!16 = !DILocation(line: 3, scope: !10)
