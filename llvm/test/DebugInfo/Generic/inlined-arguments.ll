; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from clang -O -g with the following source
;
; void f1(int x, int y);
; void f3(int line);
; void f2() {
;   f1(1, 2);
; }
; void f1(int x, int y) {
;   f3(y);
; }

; CHECK: DW_AT_name{{.*}}"f1"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"x"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"y"

; Function Attrs: uwtable
define void @_Z2f2v() #0 !dbg !4 {
  tail call void @llvm.dbg.value(metadata i32 undef, i64 0, metadata !16, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i32 2, i64 0, metadata !20, metadata !DIExpression()), !dbg !18
  tail call void @_Z2f3i(i32 2), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: uwtable
define void @_Z2f1ii(i32 %x, i32 %y) #0 !dbg !8 {
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !13, metadata !DIExpression()), !dbg !23
  tail call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !14, metadata !DIExpression()), !dbg !23
  tail call void @_Z2f3i(i32 %y), !dbg !24
  ret void, !dbg !25
}

declare void @_Z2f3i(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 ", isOptimized: true, emissionKind: 1, file: !1, enums: !2, retainedTypes: !2, subprograms: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "exp.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{!4, !8}
!4 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 3, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "exp.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1ii", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 6, file: !1, scope: !5, type: !9, variables: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11, !11}
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13, !14}
!13 = !DILocalVariable(name: "x", line: 6, arg: 1, scope: !8, file: !5, type: !11)
!14 = !DILocalVariable(name: "y", line: 6, arg: 2, scope: !8, file: !5, type: !11)
!15 = !{i32 undef}
!16 = !DILocalVariable(name: "x", line: 6, arg: 1, scope: !8, file: !5, type: !11)
!17 = !DILocation(line: 4, scope: !4)
!18 = !DILocation(line: 6, scope: !8, inlinedAt: !17)
!19 = !{i32 2}
!20 = !DILocalVariable(name: "y", line: 6, arg: 2, scope: !8, file: !5, type: !11)
!21 = !DILocation(line: 7, scope: !8, inlinedAt: !17)
!22 = !DILocation(line: 5, scope: !4)
!23 = !DILocation(line: 6, scope: !8)
!24 = !DILocation(line: 7, scope: !8)
!25 = !DILocation(line: 8, scope: !8)
!26 = !{i32 1, !"Debug Info Version", i32 3}
