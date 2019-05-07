; RUN: llc -march=bpfel -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @test() local_unnamed_addr #0 !dbg !7 {
entry:
  ret i32 0, !dbg !11
}

; CHECK-RELOC: file format ELF64-BPF
; CHECK-RELOC: RELOCATION RECORDS FOR [.debug_info]:
; CHECK-RELOC: R_BPF_64_32 .debug_abbrev
; CHECK-RELOC: R_BPF_64_64 .text
; CHECK-RELOC: RELOCATION RECORDS FOR [.BTF.ext]:
; CHECK-RELOC: R_BPF_NONE .text

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "tt.c", directory: "/home/yhs/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 350573) (llvm/trunk 350569)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 14, scope: !7)
