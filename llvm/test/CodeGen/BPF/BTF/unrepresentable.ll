; RUN: not llc -filetype=obj %s -o %t.o 2>&1 | FileCheck %s

; Source code:
;   struct entry { _Complex float value; } nested;
;   int type_id(void) { return __builtin_btf_type_id(nested.value, 1); }
;   _Complex float bad_map __attribute__((section(".maps")));
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm entry.c

; CHECK-DAG: error: CO-RE relocation has unrepresentable BTF root type 'complex'
; CHECK-DAG: error: BTF record references unrepresentable type 'complex'
; CHECK-DAG: error: BTF map 'bad_map' has unrepresentable type 'complex'

target triple = "bpf"

@nested = global { { float, float } } zeroinitializer, !dbg !2
@bad_map = global { float, float } zeroinitializer, section ".maps", !dbg !12
@"llvm.btf_type_id.0$7" = external global i64, !llvm.preserve.access.index !6 #0

define i32 @type_id() !dbg !7 {
  %id = load i64, ptr @"llvm.btf_type_id.0$7"
  %value = call i64 @llvm.bpf.passthrough.i64.i64(i32 0, i64 %id)
  %narrow = trunc i64 %value to i32, !dbg !10
  ret i32 %narrow, !dbg !10
}

declare i64 @llvm.bpf.passthrough.i64.i64(i32, i64)
attributes #0 = { "btf_type_id" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug, globals: !{!2, !12})
!1 = !DIFile(filename: "entry.c", directory: "/")
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = distinct !DIGlobalVariable(name: "nested", scope: !0, file: !1, line: 1, type: !4, isLocal: false, isDefinition: true)
!4 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "entry", size: 64, elements: !{!5})
!5 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !4, baseType: !6, size: 64)
!6 = !DIBasicType(name: "complex", size: 64, encoding: DW_ATE_complex_float)
!7 = distinct !DISubprogram(name: "type_id", scope: !1, file: !1, line: 2, type: !8, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !{!9})
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 2, column: 21, scope: !7)
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "bad_map", scope: !0, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true)
