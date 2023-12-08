; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu < %s -o - -dwarf-version=2 -no-dwarf-ranges-section | FileCheck %s --check-prefix=DISABLED
; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu < %s -o - -dwarf-version=2 | FileCheck %s

; DISABLED-NOT:  {{DW_AT_ranges|.debug_ranges}}
; DISABLED:      .section .debug_info
; DISABLED-NOT:  {{DW_AT_ranges|.section}}
; DISABLED:      .quad .Lfunc_begin0 # DW_AT_low_pc
; DISABLED-NEXT: .quad .Lfunc_end1   # DW_AT_high_pc
; DISABLED-NOT:  {{DW_AT_ranges|.debug_ranges}}

; .debug_ranges section must be emitted by default
; CHECK: .section .debug_info
; CHECK: quad 0 # DW_AT_low_pc
; CHECK-NEXT: long [[RANGE0:[.]Ldebug_ranges[0-9]+]] # DW_AT_ranges
; CHECK: .debug_ranges
; CHECK-NEXT: [[RANGE0]]:
; CHECK-NEXT: .quad .Lfunc_begin0
; CHECK-NEXT: .quad .Lfunc_end0
; CHECK-NEXT: .quad .Lfunc_begin1
; CHECK-NEXT: .quad .Lfunc_end1
; CHECK-NEXT: .quad 0
; CHECK-NEXT: .quad 0

; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f1v() #0 section "a" !dbg !7 {
entry:
  ret void, !dbg !10
}

; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f2v() #0 section "b" !dbg !11 {
entry:
  ret void, !dbg !12
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 (trunk 309523) (llvm/trunk 309526)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "funcs.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 6.0.0 (trunk 309523) (llvm/trunk 309526)"}
!7 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 1, column: 42, scope: !7)
!11 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!12 = !DILocation(line: 2, column: 42, scope: !11)
