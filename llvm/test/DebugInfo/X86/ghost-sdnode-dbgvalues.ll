; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-apple-macosx10.10.0 -o %t %s

; Testcase generated from:
; #include <stdint.h>
; int foo(int a) {
;     int b = (int16_t)a + 8;
;     int c = (int16_t)b + 8;
;     int d = (int16_t)c + 8;
;     int e = (int16_t)d + 8;
;     int f = (int16_t)e + 8;
;     return f;
; }
; by emitting the IR and then manually applying mem2reg to it.

; This testcase would trigger the assert commited along with it if the
; fix of r221709 isn't applied. There is no other check except the successful
; run of llc.
; What happened before r221709, is that SDDbgInfo (the data structure helping
; SelectionDAG to keep track of dbg.values) kept a map keyed by SDNode pointers.
; This map was never purged when the SDNodes were deallocated and thus if a new
; SDNode was allocated in the same memory, it would have an entry in the SDDbgInfo
; map upon creation (Reallocation in the same memory can happen easily as
; SelectionDAG uses a Recycling allocator). This behavior could turn into a
; pathological memory consumption explosion if the DAG combiner hit the 'right'
; allocation patterns as could be seen in PR20893.
; By nature, this test could bitrot quite easily. If it doesn't trigger an assert
; when run with r221709 reverted, then it really doesn't test anything anymore.

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i32 %a) #0 !dbg !8 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, i64 0, metadata !16, metadata !17), !dbg !18
  %conv = trunc i32 %a to i16, !dbg !19
  %conv1 = sext i16 %conv to i32, !dbg !19
  %add = add nsw i32 %conv1, 8, !dbg !19
  call void @llvm.dbg.value(metadata i32 %add, i64 0, metadata !20, metadata !17), !dbg !21
  %conv2 = trunc i32 %add to i16, !dbg !22
  %conv3 = sext i16 %conv2 to i32, !dbg !22
  %add4 = add nsw i32 %conv3, 8, !dbg !22
  call void @llvm.dbg.value(metadata i32 %add4, i64 0, metadata !23, metadata !17), !dbg !24
  %conv5 = trunc i32 %add4 to i16, !dbg !25
  %conv6 = sext i16 %conv5 to i32, !dbg !25
  %add7 = add nsw i32 %conv6, 8, !dbg !25
  call void @llvm.dbg.value(metadata i32 %add7, i64 0, metadata !26, metadata !17), !dbg !27
  %conv8 = trunc i32 %add7 to i16, !dbg !28
  %conv9 = sext i16 %conv8 to i32, !dbg !28
  %add10 = add nsw i32 %conv9, 8, !dbg !28
  call void @llvm.dbg.value(metadata i32 %add10, i64 0, metadata !29, metadata !17), !dbg !30
  %conv11 = trunc i32 %add10 to i16, !dbg !31
  %conv12 = sext i16 %conv11 to i32, !dbg !31
  %add13 = add nsw i32 %conv12, 8, !dbg !31
  call void @llvm.dbg.value(metadata i32 %add13, i64 0, metadata !32, metadata !17), !dbg !33
  ret i32 %add13, !dbg !34
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.6.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !3, globals: !2, imports: !2)
!1 = !DIFile(filename: "ghost-sdnode-dbgvalues.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_typedef, name: "int16_t", line: 30, file: !5, baseType: !6)
!5 = !DIFile(filename: "/usr/include/sys/_types/_int16_t.h", directory: "/tmp")
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "foo", line: 3, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !9, type: !10, variables: !2)
!9 = !DIFile(filename: "ghost-sdnode-dbgvalues.c", directory: "/tmp")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.6.0 "}
!16 = !DILocalVariable(name: "a", line: 3, arg: 1, scope: !8, file: !9, type: !12)
!17 = !DIExpression()
!18 = !DILocation(line: 3, column: 13, scope: !8)
!19 = !DILocation(line: 4, column: 5, scope: !8)
!20 = !DILocalVariable(name: "b", line: 4, scope: !8, file: !9, type: !12)
!21 = !DILocation(line: 4, column: 9, scope: !8)
!22 = !DILocation(line: 5, column: 5, scope: !8)
!23 = !DILocalVariable(name: "c", line: 5, scope: !8, file: !9, type: !12)
!24 = !DILocation(line: 5, column: 9, scope: !8)
!25 = !DILocation(line: 6, column: 5, scope: !8)
!26 = !DILocalVariable(name: "d", line: 6, scope: !8, file: !9, type: !12)
!27 = !DILocation(line: 6, column: 9, scope: !8)
!28 = !DILocation(line: 7, column: 5, scope: !8)
!29 = !DILocalVariable(name: "e", line: 7, scope: !8, file: !9, type: !12)
!30 = !DILocation(line: 7, column: 9, scope: !8)
!31 = !DILocation(line: 8, column: 5, scope: !8)
!32 = !DILocalVariable(name: "f", line: 8, scope: !8, file: !9, type: !12)
!33 = !DILocation(line: 8, column: 9, scope: !8)
!34 = !DILocation(line: 9, column: 5, scope: !8)
