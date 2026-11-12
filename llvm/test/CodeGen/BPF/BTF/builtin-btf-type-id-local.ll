; RUN: opt -O2 -mtriple=bpf-pc-linux -S -o %t1 %s
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   struct s {
;     int a;
;   };
;   int test(void) {
;     return __builtin_btf_type_id(*(const struct s *)0, 0);
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

; Function Attrs: nounwind
define dso_local i32 @test() #0 !dbg !7 {
entry:
  %0 = call i64 @llvm.bpf.btf.type.id(i32 0, i64 1), !dbg !11, !llvm.preserve.access.index !12
  %conv = trunc i64 %0 to i32, !dbg !11
  ret i32 %conv, !dbg !16
}

; CHECK:             .long   1                               # BTF_KIND_INT(id = 2)
; CHECK-NEXT:        .long   16777216                        # 0x1000000
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   16777248                        # 0x1000020

; CHECK:             .long   16                              # BTF_KIND_STRUCT(id = 4)
; CHECK-NEXT:        .long   67108865                        # 0x4000001
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   18
; CHECK-NEXT:        .long   2

; CHECK:             .ascii  "int"                           # string offset=1
; CHECK:             .ascii  ".text"                         # string offset=10
; CHECK:             .byte   115                             # string offset=16
; CHECK:             .byte   97                              # string offset=18
; CHECK:             .byte   48                              # string offset=20

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   10                              # Field reloc section string offset=10
; CHECK-NEXT:        .long   1
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   20
; CHECK-NEXT:        .long   7

; Function Attrs: nounwind readnone
declare i64 @llvm.bpf.btf.type.id(i32, i64) #1

attributes #0 = { nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 9783e2098800b954c55ae598a1ce5c4b93444fc0)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/bpf/test")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 9783e2098800b954c55ae598a1ce5c4b93444fc0)"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 5, column: 10, scope: !7)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s", file: !1, line: 1, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !1, line: 2, baseType: !10, size: 32)
!16 = !DILocation(line: 5, column: 3, scope: !7)
