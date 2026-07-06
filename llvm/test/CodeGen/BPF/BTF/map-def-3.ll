; RUN: llc -mtriple=bpfel -mcpu=v3 -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
;
; Source code:
;   struct key_type {
;     int a1;
;   };
;   const struct key_type __attribute__((section(".maps"))) hash_map;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm t3.c

%struct.key_type = type { i32 }

@hash_map = dso_local local_unnamed_addr constant %struct.key_type zeroinitializer, section ".maps", align 4, !dbg !0

; CHECK-BTF: [1] INT 'int' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [2] STRUCT 'key_type' size=4 vlen=1
; CHECK-BTF-NEXT:         'a1' type_id=1 bits_offset=0
; CHECK-BTF-NEXT: [3] CONST '(anon)' type_id=2
; CHECK-BTF-NEXT: [4] VAR 'hash_map' type_id=3, linkage=global
; CHECK-BTF-NEXT: [5] DATASEC '.maps' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=4 offset=0 size=4

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "hash_map", scope: !2, file: !3, line: 4, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5bd074629f00d4798674b411cf00216f38016483)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t3.c", directory: "/tmp/home/yhs/tmp1")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "key_type", file: !3, line: 1, size: 32, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "a1", scope: !7, file: !3, line: 2, baseType: !10, size: 32)
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 7, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 5bd074629f00d4798674b411cf00216f38016483)"}
