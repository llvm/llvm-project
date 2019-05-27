; RUN: llc -march=bpfel -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s
; RUN: llc -march=bpfeb -filetype=obj < %s | llvm-objdump -r - | FileCheck --check-prefix=CHECK-RELOC %s

; source code:
;   int g __attribute__((section("ids"))) = 4;
;   static volatile int s = 0;
;   int test() {
;     return g + s;
;   }
; compilation flag:
;   clang -target bpf -g -O2 -emit-llvm -S test.c

@g = dso_local local_unnamed_addr global i32 4, section "ids", align 4, !dbg !0
@s = internal global i32 0, align 4, !dbg !6

; Function Attrs: norecurse nounwind
define dso_local i32 @test() local_unnamed_addr #0 !dbg !14 {
  %1 = load i32, i32* @g, align 4, !dbg !17, !tbaa !18
  %2 = load volatile i32, i32* @s, align 4, !dbg !22, !tbaa !18
  %3 = add nsw i32 %2, %1, !dbg !23
  ret i32 %3, !dbg !24
}

; CHECK-RELOC: file format ELF64-BPF
; CHECK-RELOC: RELOCATION RECORDS FOR [.BTF]:
; CHECK-RELOC: R_BPF_NONE .bss
; CHECK-RELOC: R_BPF_NONE g
; CHECK-RELOC: RELOCATION RECORDS FOR [.BTF.ext]:

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/relocation")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "s", scope: !2, file: !3, line: 2, type: !8, isLocal: true, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 9.0.0 (trunk 360739) (llvm/trunk 360747)"}
!14 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 3, type: !15, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{!9}
!17 = !DILocation(line: 4, column: 10, scope: !14)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 4, column: 14, scope: !14)
!23 = !DILocation(line: 4, column: 12, scope: !14)
!24 = !DILocation(line: 4, column: 3, scope: !14)
