; RUN: llc -mtriple=bpfel -mcpu=v4 -mattr=+arena -filetype=asm < %s | FileCheck %s
; RUN: llc -mtriple=bpfel -mcpu=v4 -filetype=asm < %s | FileCheck --check-prefix=OFF %s

; The kernel verifies a function that is not static from its type in BTF.
;   int *f(int *p, void *c __attribute__((btf_decl_tag("arg:ctx"))));
; The result and p are addresses in arena. The source says what c is.

; OFF-NOT: BTF_KIND_TYPE_TAG

; 'void *' of c
; CHECK:      .long 0 # BTF_KIND_PTR(id = 3)
; CHECK-NEXT: .long 33554432
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 0 # BTF_KIND_FUNC_PROTO(id = 4)
; CHECK-NEXT: .long 218103810
; CHECK-NEXT: .long 6
; CHECK-NEXT: .long {{[0-9]+}}
; CHECK-NEXT: .long 6
; CHECK-NEXT: .long {{[0-9]+}}
; CHECK-NEXT: .long 3
; CHECK-NEXT: .long [[NAME:[0-9]+]] # BTF_KIND_TYPE_TAG(id = 5)
; CHECK-NEXT: .long 301989888
; CHECK-NEXT: .long 2
; CHECK-NEXT: .long 0 # BTF_KIND_PTR(id = 6)
; CHECK-NEXT: .long 33554432
; CHECK-NEXT: .long 5
; CHECK:      .ascii "arena" # string offset=[[NAME]]

define ptr @f(ptr %p, ptr %c) !dbg !7 {
  ret ptr %p, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10, !12}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!13 = !{!14, !15}
!14 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!15 = !DILocalVariable(name: "c", arg: 2, scope: !7, file: !1, line: 1, type: !12, annotations: !16)
!16 = !{!17}
!17 = !{!"btf_decl_tag", !"arg:ctx"}
!18 = !DILocation(line: 1, column: 1, scope: !7)
