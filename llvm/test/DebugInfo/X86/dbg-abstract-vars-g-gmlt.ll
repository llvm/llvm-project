; RUN: llc < %s -filetype=obj | llvm-dwarfdump -v - --debug-info | FileCheck %s
;
; IR module created as follows:
;   clang -emit-llvm -S db-abs-1.cpp -o db-abs-1.ll -g
;   clang -emit-llvm -S db-abs-2.cpp -o db-abs-2.ll -gmlt
;   llvm-link db-abs-1.ll db-abs-2.ll -S -o db-abs-3.ll
; --- db-abs-1.cpp ---
; void f1();
; inline __attribute__((always_inline)) void f2(int) {
;   f1();
; }
; void f3() {
;   f2(0);
; }
; --- db-abs-2.cpp ---
; void f() {
; }
; ---
; The point is that f3() is compiled -g and we get an abstract variable for the
; unnamed parameter to f2(); then f() is compiled -gmlt and it's okay to have
; the abstract variable still there.
; PR31437.
;
; (The 'always_inline' attribute forces f2() to be inlined even at -O0, the 
; 'inline' keyword means the non-inlined definition of f2() can be omitted from
; the IR.  These are just tactics to simplify the generated test case.)
;
; Verify we see the formal parameter in the first compile-unit, and nothing in
; the second compile-unit.
;
; CHECK:      DW_TAG_compile_unit
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_name {{.*}} "db-abs-1.cpp"
; CHECK-NOT:  NULL
; CHECK:      DW_TAG_subprogram
; CHECK-NEXT: DW_AT_linkage_name {{.*}} "_Z2f2i"
; CHECK-NOT:  {{DW_TAG|NULL}}
; CHECK:      DW_TAG_formal_parameter
; CHECK-NOT:  DW_AT_name
; CHECK:      {{DW_TAG|NULL}}
; CHECK:      DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}} "_Z2f2i"

; CHECK:      DW_TAG_compile_unit
; CHECK-NOT:  DW_TAG

; ModuleID = 'llvm-link'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline uwtable
define void @_Z2f3v() #0 !dbg !8 {
entry:
  %.addr.i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata ptr %.addr.i, metadata !11, metadata !16), !dbg !17
  store i32 0, ptr %.addr.i, align 4
  call void @_Z2f1v(), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare void @_Z2f1v() #2

; Function Attrs: noinline nounwind uwtable
define void @_Z1fv() #3 !dbg !21 {
entry:
  ret void, !dbg !23
}

attributes #0 = { noinline uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
attributes #3 = { noinline nounwind uwtable }

!llvm.dbg.cu = !{!0, !3}
!llvm.ident = !{!5, !5}
!llvm.module.flags = !{!6, !7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 293745)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "db-abs-1.cpp", directory: "/home/probinson/projects/scratch/pr31437")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "clang version 5.0.0 (trunk 293745)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!4 = !DIFile(filename: "db-abs-2.cpp", directory: "/home/probinson/projects/scratch/pr31437")
!5 = !{!"clang version 5.0.0 (trunk 293745)"}
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 5, type: !9, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocalVariable(arg: 1, scope: !12, file: !1, line: 2, type: !15)
!12 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2i", scope: !1, file: !1, line: 2, type: !13, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DIExpression()
!17 = !DILocation(line: 2, column: 50, scope: !12, inlinedAt: !18)
!18 = distinct !DILocation(line: 6, column: 3, scope: !8)
!19 = !DILocation(line: 3, column: 3, scope: !12, inlinedAt: !18)
!20 = !DILocation(line: 7, column: 1, scope: !8)
!21 = distinct !DISubprogram(name: "f", scope: !4, file: !4, line: 1, type: !22, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !3, retainedNodes: !2)
!22 = !DISubroutineType(types: !2)
!23 = !DILocation(line: 2, column: 1, scope: !21)
