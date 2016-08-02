; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Given the following source, ensure that the discriminator is emitted for
; the inlined callsite.

;void xyz();
;static void __attribute__((always_inline)) bar() { xyz(); }
;void foo() {
;  bar(); bar();
;}

;CHECK: DW_TAG_inlined_subroutine
;CHECK-NOT: DW_AT_GNU_discriminator
;CHECK: DW_TAG_inlined_subroutine
;CHECK-NOT: {{DW_TAG|NULL}}
;CHECK: DW_AT_GNU_discriminator{{.*}}0x01

; Function Attrs: uwtable
define void @_Z3foov() #0 !dbg !4 {
  tail call void @_Z3xyzv(), !dbg !11
  tail call void @_Z3xyzv(), !dbg !13
  ret void, !dbg !16
}

declare void @_Z3xyzv() #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 252497)", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "a.cc", directory: "/tmp")
!2 = !{}
!3 = !{!4, !7}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "bar", linkageName: "_ZL3barv", scope: !1, file: !1, line: 2, type: !5, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, variables: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 252497)"}
!11 = !DILocation(line: 2, column: 52, scope: !7, inlinedAt: !12)
!12 = distinct !DILocation(line: 4, column: 3, scope: !4)
!13 = !DILocation(line: 2, column: 52, scope: !7, inlinedAt: !14)
!14 = distinct !DILocation(line: 4, column: 10, scope: !15)
!15 = !DILexicalBlockFile(scope: !4, file: !1, discriminator: 1)
!16 = !DILocation(line: 5, column: 1, scope: !4)
