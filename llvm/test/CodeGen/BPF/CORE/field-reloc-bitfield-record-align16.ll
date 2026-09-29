; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; Source code:
;   struct t2 {
;       int a[8];
;       long: 64;
;       long: 64;
;       long: 64;
;       long: 64;
;   };
;
;   struct t1 {
;       int f1: 1;
;       int f2: 2;
;       long: 61;
;       long: 64;
;       long: 64;
;       long: 64;
;       long: 64;
;       long: 64;
;       long: 64;
;       long: 64;
;       struct t2 f3;
;   } __attribute__((preserve_access_index));
;
;   struct t1 g;
;   int foo() {
;     return g.f1;
;   }
; Compilation flag:
;   clang -target bpfel -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

%struct.t1 = type { i512, %struct.t2 }
%struct.t2 = type { [8 x i32], i256 }

@g = dso_local global %struct.t1 zeroinitializer, align 4, !dbg !0

; Function Attrs: nounwind
define dso_local i32 @foo() #0 !dbg !22 {
entry:
  %0 = call ptr @llvm.preserve.struct.access.index.p0.p0.t1s(ptr elementtype(%struct.t1) @g, i32 0, i32 0), !dbg !26, !llvm.preserve.access.index !5
  %bf.load = load i512, ptr %0, align 4, !dbg !26
  %bf.shl = shl i512 %bf.load, 511, !dbg !26
  %bf.ashr = ashr i512 %bf.shl, 511, !dbg !26
  %bf.cast = trunc i512 %bf.ashr to i32, !dbg !26
  ret i32 %bf.cast, !dbg !27
}

; CHECK:        .long   68                              # BTF_KIND_STRUCT(id = 4)

; CHECK:        .ascii  ".text"                         # string offset=9
; CHECK:        .ascii  "t1"                            # string offset=68
; CHECK:        .ascii  "0:0"                           # string offset=105

; CHECK:        .long   16                              # FieldReloc
; CHECK:        .long   9                               # Field reloc section string offset=9
; CHECK:        .long   1
; CHECK:        .long   .Ltmp[[#]]
; CHECK:        .long   4
; CHECK:        .long   105
; CHECK:        .long   0

; Function Attrs: nofree nosync nounwind readnone willreturn
declare ptr @llvm.preserve.struct.access.index.p0.p0.t1s(ptr, i32 immarg, i32 immarg) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nofree nosync nounwind readnone willreturn }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20}
!llvm.ident = !{!21}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 23, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 468279d2d249e44ffa3535a613245b4ceb81a908)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/bitfield/simple", checksumkind: CSK_MD5, checksum: "b9c80125731b87136772eec36d0b48a3")
!4 = !{!0}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 9, size: 1024, elements: !6)
!6 = !{!7, !9, !10}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "f1", scope: !5, file: !3, line: 10, baseType: !8, size: 1, flags: DIFlagBitField, extraData: i64 0)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "f2", scope: !5, file: !3, line: 11, baseType: !8, size: 2, offset: 1, flags: DIFlagBitField, extraData: i64 0)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "f3", scope: !5, file: !3, line: 20, baseType: !11, size: 512, offset: 512)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !3, line: 1, size: 512, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !3, line: 2, baseType: !14, size: 256)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 256, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 8)
!17 = !{i32 7, !"Dwarf Version", i32 5}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 7, !"frame-pointer", i32 2}
!21 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 468279d2d249e44ffa3535a613245b4ceb81a908)"}
!22 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 24, type: !23, scopeLine: 24, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !25)
!23 = !DISubroutineType(types: !24)
!24 = !{!8}
!25 = !{}
!26 = !DILocation(line: 25, column: 12, scope: !22)
!27 = !DILocation(line: 25, column: 3, scope: !22)
