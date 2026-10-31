; RUN: llc -mtriple=bpfel -filetype=obj %s -o %t.o
; RUN: llvm-objcopy --dump-section=.BTF=%t.btf %t.o
; RUN: %python %p/print_btf.py %t.btf | FileCheck %s --implicit-check-not=" ARRAY " --implicit-check-not="VAR 'samples'"

; Source code:
;   _Complex float samples[2];
;   int keep;
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm array.c

; CHECK: [[INT:[0-9]+]]] INT 'int'
; CHECK: [[KEEP:[0-9]+]]] VAR 'keep' type_id=[[INT]],
; CHECK: DATASEC '.bss' size=0 vlen=1
; CHECK-NEXT: type_id=[[KEEP]] offset={{[0-9]+}} size=4

@samples = global [2 x { float, float }] zeroinitializer, !dbg !2
@keep = global i32 0, !dbg !6

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug, globals: !{!2, !6})
!1 = !DIFile(filename: "array.c", directory: "/")
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = distinct !DIGlobalVariable(name: "samples", scope: !0, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true)
!4 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 128, elements: !{!DISubrange(count: 2)})
!5 = !DIBasicType(name: "complex", size: 64, encoding: DW_ATE_complex_float)
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "keep", scope: !0, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Debug Info Version", i32 3}
