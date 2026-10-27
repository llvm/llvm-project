; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mtriple=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   void *a;
;   typedef void alias;
;   alias *b;
;   void (*cb)(void);
; Compilation flag:
;   clang -target bpf -O2 -g -fcommon -S -emit-llvm t.c
; Change cb's DISubroutineType list from !{null} to !{} to model an
; unavailable prototype.

@a = common dso_local local_unnamed_addr global ptr null, align 8, !dbg !0
@b = common dso_local local_unnamed_addr global ptr null, align 8, !dbg !11
@cb = common dso_local local_unnamed_addr global ptr null, align 8, !dbg !15

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 1)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 2)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   1                       # BTF_KIND_TYPEDEF(id = 3)
; CHECK-NEXT:        .long   134217728               # 0x8000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 4)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .byte   0                       # string offset=0

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0, !11, !15}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 8.0.0 (trunk 345296) (llvm/trunk 345297)"}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "alias", file: !3, line: 2, baseType: null)
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression())
!16 = distinct !DIGlobalVariable(name: "cb", scope: !2, file: !3, line: 4, type: !17, isLocal: false, isDefinition: true)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !18, size: 64)
!18 = !DISubroutineType(types: !{})
