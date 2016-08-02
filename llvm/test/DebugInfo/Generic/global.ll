; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Also test that the null streamer doesn't crash with debug info.
; RUN: %llc_dwarf -O0 -filetype=null < %s

; generated from the following source compiled to bitcode with clang -g -O1
; static int i;
; int main() {
;   (void)&i;
; }

; CHECK: debug_info contents
; CHECK: DW_TAG_variable

; Function Attrs: nounwind readnone uwtable
define i32 @main() #0 !dbg !4 {
entry:
  ret i32 0, !dbg !12
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: true, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !9, imports: !2)
!1 = !DIFile(filename: "global.cpp", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "main", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 2, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "global.cpp", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DIGlobalVariable(name: "i", linkageName: "_ZL1i", line: 1, isLocal: true, isDefinition: true, scope: null, file: !5, type: !8)
!11 = !{i32 2, !"Dwarf Version", i32 3}
!12 = !DILocation(line: 4, scope: !4)
!13 = !{i32 1, !"Debug Info Version", i32 3}
