; RUN: %llc_dwarf -filetype=obj %s -o - | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; Generated at -O2 from:
;   void f();
;   __attribute__((always_inline)) void g() {
;     f();
;   }
;   void h() {
;     g();
;   };
; CHECK: DW_TAG_subprogram
; CHECK:  DW_AT_abstract_origin {{.*}}"g"
; CHECK-NOT:  DW_AT_abstract_origin {{.*}}"g"
; CHECK: DW_TAG
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: alwaysinline nounwind ssp uwtable
define void @g() #0 !dbg !7 {
entry:
  tail call void (...) @f() #3, !dbg !10
  ret void, !dbg !11
}

declare void @f(...)

; Function Attrs: nounwind ssp uwtable
define void @h() #2 !dbg !12 {
entry:
  tail call void (...) @f() #3, !dbg !13
  ret void, !dbg !15
}

attributes #0 = { alwaysinline nounwind ssp uwtable }
attributes #2 = { nounwind ssp uwtable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/llvm")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 2}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 3.9.0 "}
!7 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 3, column: 3, scope: !7)
!11 = !DILocation(line: 4, column: 1, scope: !7)
!12 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 5, type: !8, isLocal: false, isDefinition: true, scopeLine: 5, isOptimized: true, unit: !0, variables: !2)
!13 = !DILocation(line: 3, column: 3, scope: !7, inlinedAt: !14)
!14 = distinct !DILocation(line: 6, column: 3, scope: !12)
!15 = !DILocation(line: 7, column: 1, scope: !12)
