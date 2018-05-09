; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - -use-dwarf-ranges-base-address-specifier | FileCheck --check-prefix=COMMON --check-prefix=BASE %s
; RUN: llc -filetype=asm -mtriple=x86_64-pc-linux-gnu %s -o - | FileCheck --check-prefix=COMMON --check-prefix=NOBASE %s

; Group ranges in a range list that apply to the same section and use a base
; address selection entry to reduce the number of relocations to one reloc per
; section per range list. DWARF5 debug_rnglist will be more efficient than this
; in terms of relocations, but it's still better than one reloc per entry in a
; range list.

; This is an object/executable size tradeoff - shrinking objects, but growing
; the linked executable. In one large binary tested, total object size (not just
; debug info) shrank by 16%, entirely relocation entries. Linked executable
; grew by 4%. This was with compressed debug info in the objects, uncompressed
; in the linked executable. Without compression in the objects, the win would be
; smaller (the growth of debug_ranges itself would be more significant).

; COMMON: {{^.Ldebug_ranges0}}
; COMMON-NEXT:   .quad   .Lfunc_begin0
; COMMON-NEXT:   .quad   .Lfunc_end0
; BASE-NEXT:   .quad   -1
; BASE-NEXT:   .quad   .Lfunc_begin1
; BASE-NEXT:   .quad   .Lfunc_begin1-.Lfunc_begin1
; BASE-NEXT:   .quad   .Lfunc_end1-.Lfunc_begin1
; BASE-NEXT:   .quad   .Lfunc_begin3-.Lfunc_begin1
; BASE-NEXT:   .quad   .Lfunc_end3-.Lfunc_begin1
; BASE-NEXT:   .quad   -1
; BASE-NEXT:   .quad   0
; NOBASE-NEXT:   .quad   .Lfunc_begin1
; NOBASE-NEXT:   .quad   .Lfunc_end1
; NOBASE-NEXT:   .quad   .Lfunc_begin3
; NOBASE-NEXT:   .quad   .Lfunc_end3
; COMMON-NEXT:   .quad   .Lfunc_begin4
; COMMON-NEXT:   .quad   .Lfunc_end4
; COMMON-NEXT:   .quad   .Lfunc_begin5
; COMMON-NEXT:   .quad   .Lfunc_end5
; COMMON-NEXT:   .quad   0
; COMMON-NEXT:   .quad   0

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

; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f3v() #0 section "b" {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f4v() #0 section "b" !dbg !13 {
entry:
  ret void, !dbg !14
}

; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f5v() #0 section "e" !dbg !15 {
entry:
  ret void, !dbg !16
}

; Function Attrs: noinline nounwind optnone uwtable
define void @_Z2f6v() #0 section "f" !dbg !17 {
entry:
  ret void, !dbg !18
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

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
!13 = distinct !DISubprogram(name: "f4", linkageName: "_Z2f4v", scope: !1, file: !1, line: 4, type: !8, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!14 = !DILocation(line: 4, column: 42, scope: !13)
!15 = distinct !DISubprogram(name: "f5", linkageName: "_Z2f5v", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 5, column: 42, scope: !15)
!17 = distinct !DISubprogram(name: "f6", linkageName: "_Z2f6v", scope: !1, file: !1, line: 6, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DILocation(line: 6, column: 42, scope: !17)
