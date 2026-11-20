; RUN: opt -O2 %s | llc -filetype=obj -o - | \
; RUN:   llvm-objdump --no-addresses --no-show-raw-insn --no-print-imm-hex -dr - | \
; RUN:   FileCheck %s --implicit-check-not=CO-RE

target triple = "bpfel"

%S = type { i64 }

; &(*a)[1] for int (*a)[2][3] advances by three ints with no CO-RE relocation.
; CHECK-LABEL: <scalar>:
; CHECK:       += 12
define ptr @scalar(ptr %a) !dbg !6 {
  %row = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([2 x [3 x i32]]) %a, i32 1, i32 1), !llvm.preserve.access.index !10
  ret ptr %row
}

; &p[1] for struct S (*p)[3] advances by three records in the target layout.
; CHECK-LABEL: <record>:
; CHECK:       = 24
; CHECK-NEXT:  CO-RE <byte_off> [{{[0-9]+}}] struct S::[3] (3)
define ptr @record(ptr %p) !dbg !7 {
  %row = call ptr @llvm.preserve.array.access.index.p0.p0(ptr elementtype([3 x %S]) %p, i32 0, i32 1), !llvm.preserve.access.index !13
  ret ptr %row
}

declare ptr @llvm.preserve.array.access.index.p0.p0(ptr, i32 immarg, i32 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "subarray.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{!20, !21})
!4 = !DISubroutineType(types: !{!13, !13})
!6 = distinct !DISubprogram(name: "scalar", scope: !1, file: !1, type: !3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !{!23})
!7 = distinct !DISubprogram(name: "record", scope: !1, file: !1, type: !4, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !{!24})
!8 = !DISubrange(count: 2)
!9 = !DISubrange(count: 3)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 192, elements: !12)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!8, !9}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 192, elements: !16)
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", size: 64, elements: !17)
!16 = !{!9}
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !15, baseType: !19, size: 64)
!19 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!22 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 96, elements: !16)
!23 = !DILocalVariable(name: "a", arg: 1, scope: !6, file: !1, type: !21)
!24 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, type: !13)
