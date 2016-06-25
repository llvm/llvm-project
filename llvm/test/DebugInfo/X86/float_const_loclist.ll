; RUN: llc %s -stop-after=livedebugvalues -o %t 2>&1 | FileCheck --check-prefix=SANITY %s
; RUN: llc < %s -filetype=obj | llvm-dwarfdump - | FileCheck %s
; Test debug_loc support for floating point constants.
;
; Created from clang -O1:
;   void barrier();
;   void foo() {
;     float f;
;     long double ld;
;     barrier();
;     f = 3.14;
;     ld = 3.14;
;     barrier();
;   }
;
; SANITY: CALL{{.*}} @barrier
; SANITY: DBG_VALUE x86_fp80 0xK4000C8F5C28F5C28F800
; SANITY: DBG_VALUE float 0x40091EB860000000
; SANITY: TAILJMP{{.*}} @barrier
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NEXT:  DW_AT_location {{.*}} (0x[[LD:.*]])
; CHECK-NEXT:  DW_AT_name {{.*}}"ld"
; CHECK: DW_TAG_variable
; CHECK-NEXT:  DW_AT_location {{.*}} (0x[[F:.*]])
; CHECK-NEXT:  DW_AT_name {{.*}}"f"
;
; CHECK: .debug_loc contents:
; CHECK: [[LD]]: Beginning address offset: [[START:.*]]
; CHECK:            Ending address offset: [[END:.*]]
; CHECK: Location description: 10 80 f0 a3 e1 f5 d1 f0 fa c8 01 93 08 10 80 80 01 9d 10 40
;                   constu 0xc8f5c28f5c28f800, piece 8, constu 0x00004000, bit-piece 16 64
; CHECK: [[F]]: Beginning address offset: [[START]]
; CHECK:           Ending address offset: [[END]]
; CHECK:            Location description: 10 c3 eb a3 82 04
;                                         constu ...
source_filename = "test.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 !dbg !4 {
entry:
  tail call void (...) @barrier() #3, !dbg !16
  tail call void @llvm.dbg.value(metadata float 0x40091EB860000000, i64 0, metadata !8, metadata !17), !dbg !18
  tail call void @llvm.dbg.value(metadata x86_fp80 0xK4000C8F5C28F5C28F800, i64 0, metadata !10, metadata !17), !dbg !19
  tail call void (...) @barrier() #3, !dbg !20
  ret void, !dbg !21
}

declare void @barrier(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.9.0 (trunk 265328) (llvm/trunk 265330)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/Volumes/Data/radar/25448338")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 4, isOptimized: true, unit: !0, variables: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8, !10}
!8 = !DILocalVariable(name: "f", scope: !4, file: !1, line: 5, type: !9)
!9 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!10 = !DILocalVariable(name: "ld", scope: !4, file: !1, line: 6, type: !11)
!11 = !DIBasicType(name: "long double", size: 128, align: 128, encoding: DW_ATE_float)
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !{!"clang version 3.9.0 (trunk 265328) (llvm/trunk 265330)"}
!16 = !DILocation(line: 7, column: 3, scope: !4)
!17 = !DIExpression()
!18 = !DILocation(line: 5, column: 9, scope: !4)
!19 = !DILocation(line: 6, column: 15, scope: !4)
!20 = !DILocation(line: 10, column: 3, scope: !4)
!21 = !DILocation(line: 11, column: 1, scope: !4)
