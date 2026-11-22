; RUN: opt -O2 %s | llc -filetype=obj -o %t.o
; RUN: llvm-objdump --no-print-imm-hex -dr --no-show-raw-insn %t.o | FileCheck %s

; A global C++ template's qualified argument contains punctuation also used
; in relocation global names. Both consumers must recover the access path
; and local offset from the suffix.

; CHECK-LABEL: <read_member>:
; CHECK: {{[rw]}}0 = *(u32 *)(r1 + 4)
; CHECK-NEXT: CO-RE <byte_off> {{\[[0-9]+\]}} struct S<ns::Arg$>::b (0:1)

target triple = "bpfeb"

%struct.S = type { i32, i32 }

define i32 @read_member(ptr %p) !dbg !7 {
  %field = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 1, i32 1), !llvm.preserve.access.index !3, !dbg !9
  %value = load i32, ptr %field, align 4, !dbg !9
  ret i32 %value, !dbg !9
}

declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<ns::Arg$>", file: !1, size: 64, elements: !4, templateParams: !11)
!4 = !{!5, !6}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, baseType: !2, size: 32)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !1, baseType: !2, size: 32, offset: 32)
!7 = distinct !DISubprogram(name: "read_member", scope: !1, file: !1, type: !8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DISubroutineType(types: !{!2, !15})
!9 = !DILocation(line: 1, scope: !7)
!11 = !{!DITemplateTypeParameter(type: !12)}
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Arg$", scope: !13, file: !1, size: 8, elements: !{})
!13 = !DINamespace(name: "ns", scope: null)
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
