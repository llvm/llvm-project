; RUN: opt -O2 -S %s -o %t.ll
; RUN: FileCheck %s --check-prefix=IR < %t.ll
; RUN: llc -filetype=obj %t.ll -o %t.o
; RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -dr %t.o | FileCheck %s

; A pointer access and a 32-bit offset query on the same field need different
; load types. The global and function-local structs named S have different
; members at the same offset.
; IR-DAG: @[[OFFSET:[^ ]+]] = external global i32, !llvm.preserve.access.index ![[ROOT:[0-9]+]]
; IR-DAG: @{{[^ ]+}} = external global i64, !llvm.preserve.access.index ![[ROOT]]
; IR-LABEL: define {{.*}}@member_offset(
; IR: load i32, ptr @[[OFFSET]]

; CHECK-LABEL: <read_member>:
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 4)
; CHECK-NEXT: CO-RE <byte_off> {{\[}}[[TYPE_ID:[0-9]+]]{{\]}} struct S::b (0:1)
; CHECK-LABEL: <member_offset>:
; CHECK: {{[rw]}}0 = 4{{$}}
; CHECK-NEXT: CO-RE <byte_off> {{\[}}[[TYPE_ID]]{{\]}} struct S::b (0:1)
; CHECK-LABEL: <read_other_member>:
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 4)
; CHECK-NEXT: CO-RE <byte_off> {{\[[0-9]+\]}} struct S::c (0:1)

target triple = "bpfeb"

%struct.S = type { i32, i32 }

define i32 @read_member(ptr %p) !dbg !7 {
  %field = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 1, i32 1), !llvm.preserve.access.index !3, !dbg !9
  %value = load i32, ptr %field, align 4, !dbg !9
  ret i32 %value, !dbg !9
}

define i32 @member_offset(ptr %p) !dbg !14 {
  %field = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 1, i32 1), !llvm.preserve.access.index !3, !dbg !15
  %offset = call i32 @llvm.bpf.preserve.field.info.p0(ptr %field, i64 0), !dbg !15
  ret i32 %offset, !dbg !15
}

define i32 @read_other_member(ptr %p) !dbg !20 {
  %field = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 1, i32 1), !llvm.preserve.access.index !16, !dbg !22
  %value = load i32, ptr %field, align 4, !dbg !22
  ret i32 %value, !dbg !22
}

declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, size: 64, elements: !4)
!4 = !{!5, !6}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, baseType: !2, size: 32)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !1, baseType: !2, size: 32, offset: 32)
!7 = distinct !DISubprogram(name: "read_member", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DISubroutineType(types: !{!2, !13})
!9 = !DILocation(line: 1, scope: !7)
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!14 = distinct !DISubprogram(name: "member_offset", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 3, scope: !14)
!16 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", scope: !20, file: !1, size: 64, elements: !17)
!17 = !{!18, !19}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !16, file: !1, baseType: !2, size: 32)
!19 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !16, file: !1, baseType: !2, size: 32, offset: 32)
!20 = distinct !DISubprogram(name: "read_other_member", scope: !1, file: !1, type: !21, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DISubroutineType(types: !{!2, !23})
!22 = !DILocation(line: 4, scope: !20)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
