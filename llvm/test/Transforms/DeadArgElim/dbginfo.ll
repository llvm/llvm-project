; RUN: opt -passes=deadargelim -S < %s | FileCheck %s
; PR14016

; Built with clang (then manually running -mem2reg with opt) from the following source:
; static void f1(int, ...) {
; }
;
; void f2() {
;   f1(1);
; }

; Test both varargs removal and removal of a traditional dead arg together, to
; test both the basic functionality, and a particular wrinkle involving updating
; the function->debug info mapping on update to ensure it's accurate when used
; again for the next removal.

; CHECK: define internal void @_ZL2f1iz({{.*}} !dbg [[SP:![0-9]+]]
; CHECK: [[SP]] = distinct !DISubprogram(name: "f1"

; Check that debug info metadata for subprograms stores pointers to
; updated LLVM functions.

; Function Attrs: uwtable
define void @_Z2f2v() #0 !dbg !4 {
entry:
  call void (i32, ...) @_ZL2f1iz(i32 1), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind uwtable
define internal void @_ZL2f1iz(i32, ...) #1 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %0, metadata !17, metadata !18), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}
!llvm.ident = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "dbg.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", line: 4, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "dbg.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "f1", linkageName: "_ZL2f1iz", line: 1, isLocal: true, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !9, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, null}
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{!"clang version 3.6.0 "}
!15 = !DILocation(line: 5, column: 3, scope: !4)
!16 = !DILocation(line: 6, column: 1, scope: !4)
!17 = !DILocalVariable(name: "", line: 1, arg: 1, scope: !8, file: !5, type: !11)
!18 = !DIExpression()
!19 = !DILocation(line: 1, column: 19, scope: !8)
!20 = !DILocation(line: 2, column: 1, scope: !8)
