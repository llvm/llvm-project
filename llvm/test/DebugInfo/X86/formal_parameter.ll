; ModuleID = 'formal_parameter.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"
;
; From (clang -g -c -O1):
;
; int lookup(int* map);
; int verify(int val);
; void foo(int map)
; {
;   lookup(&map);
;   if (!verify(map)) {  }
; }
;
; RUN: opt %s -O2 -S -o %t
; RUN: cat %t | FileCheck --check-prefix=LOWERING %s
; RUN: llc -filetype=obj %t -o - | llvm-dwarfdump -debug-info - | FileCheck %s
; Test that we only emit only one DW_AT_formal_parameter "map" for this function.
; rdar://problem/14874886
;
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name {{.*}}map
; CHECK-NOT: DW_AT_name {{.*}}map

; Function Attrs: nounwind ssp uwtable
define void @foo(i32 %map) #0 !dbg !4 {
entry:
  %map.addr = alloca i32, align 4
  store i32 %map, ptr %map.addr, align 4, !tbaa !15
  call void @llvm.dbg.declare(metadata ptr %map.addr, metadata !10, metadata !DIExpression()), !dbg !14
  %call = call i32 (ptr, ...) @lookup(ptr %map.addr) #3, !dbg !19
  ; Ensure that all dbg intrinsics have the same scope after
  ; LowerDbgDeclare is finished with them.
  ;
  ; LOWERING: call void @llvm.dbg.value{{.*}}, !dbg ![[LOC:.*]]
  ; LOWERING: call void @llvm.dbg.value{{.*}}, !dbg ![[LOC]]
  ; LOWERING: call void @llvm.dbg.value{{.*}}, !dbg ![[LOC]]
%0 = load i32, ptr %map.addr, align 4, !dbg !20, !tbaa !15
  %call1 = call i32 (i32, ...) @verify(i32 %0) #3, !dbg !20
  ret void, !dbg !22
}

; LOWERING: ![[SCOPE:[0-9]+]] = distinct !DISubprogram(name: "foo",
; LOWERING: ![[LOC]] = !DILocation(line: 0, scope: ![[SCOPE]]

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @lookup(...)

declare i32 @verify(...)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "formal_parameter.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !9)
!5 = !DIFile(filename: "formal_parameter.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DILocalVariable(name: "map", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.5.0 "}
!14 = !DILocation(line: 1, scope: !4)
!15 = !{!16, !16, i64 0}
!16 = !{!"int", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C/C++ TBAA"}
!19 = !DILocation(line: 3, scope: !4)
!20 = !DILocation(line: 4, scope: !21)
!21 = distinct !DILexicalBlock(line: 4, column: 0, file: !1, scope: !4)
!22 = !DILocation(line: 5, scope: !4)
