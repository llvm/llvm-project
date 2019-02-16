; RUN: llc -march=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s
; RUN: llc -march=bpfeb -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   struct t1;
;   struct t2 {struct t1 *p;} a;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t.c

%struct.t2 = type { %struct.t1* }
%struct.t1 = type opaque

@a = common dso_local local_unnamed_addr global %struct.t2 zeroinitializer, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

; CHECK:             .section        .BTF,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   48
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   1                       # BTF_KIND_STRUCT(id = 1)
; CHECK-NEXT:        .long   67108865                # 0x4000001
; CHECK-NEXT:        .long   8
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   2
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   0                       # BTF_KIND_PTR(id = 2)
; CHECK-NEXT:        .long   33554432                # 0x2000000
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   6                       # BTF_KIND_FWD(id = 3)
; CHECK-NEXT:        .long   117440512               # 0x7000000
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .byte   0                       # string offset=0
; CHECK-NEXT:        .ascii  "t2"                    # string offset=1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .byte   112                     # string offset=4
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .ascii  "t1"                    # string offset=6
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .section        .BTF.ext,"",@progbits
; CHECK-NEXT:        .short  60319                   # 0xeb9f
; CHECK-NEXT:        .byte   1
; CHECK-NEXT:        .byte   0
; CHECK-NEXT:        .long   24
; CHECK-NEXT:        .long   0
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   8                       # FuncInfo
; CHECK-NEXT:        .long   16                      # LineInfo

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t2", file: !3, line: 2, size: 64, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "p", scope: !6, file: !3, line: 2, baseType: !9, size: 64)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 1, flags: DIFlagFwdDecl)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 8.0.0 (trunk 344789) (llvm/trunk 344782)"}
