; RUN: llc %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name{{.*}}"f"
; CHECK-NOT: DW_TAG_compile_unit
;
; created from
;   void f() {} // compile with -g
;   void g() {} // compile with -Rpass=inline
; and llvm-linking the result.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: nounwind ssp uwtable
define void @f() #0 !dbg !4 {
entry:
  ret void, !dbg !15
}

; Function Attrs: nounwind ssp uwtable
define void @g() #0 !dbg !9 {
entry:
  ret void, !dbg !16
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0, !7}
!llvm.ident = !{!11, !11}
!llvm.module.flags = !{!12, !13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 265328) (llvm/trunk 265330)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 265328) (llvm/trunk 265330)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!9 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !7, variables: !2)
!10 = !DISubroutineType(types: !2)
!11 = !{!"clang version 3.9.0 (trunk 265328) (llvm/trunk 265330)"}
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !DILocation(line: 1, column: 12, scope: !4)
!16 = !DILocation(line: 1, column: 12, scope: !9)
