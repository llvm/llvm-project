; RUN: %llc_dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump %t | FileCheck %s

; Make sure DIFlagArtificial on a global will emit DW_AT_artificial.
; (This happens through its type, because DIGlobalVariable doesn't take flags.)

;CHECK: DW_TAG_variable
;CHECK-NEXT: DW_AT_name      ("<i32 as _>::{vtable}")
;CHECK-NEXT: DW_AT_type      (0x{{[0-9a-f]+}} "<i32 as _>::{vtable_type}")
;CHECK-NEXT: DW_AT_artificial        (true)
;CHECK-NEXT: DW_AT_location  (DW_OP_addr 0x0)

@vtable.0 = hidden constant <{ ptr, [16 x i8] }> <{ ptr @"_ZN4core3ptr24drop_in_place$LT$i32$GT$17hb6decb5d90d320b6E", [16 x i8] c"\04\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00" }>, align 8, !dbg !0

; Function Attrs: inlinehint nonlazybind uwtable
declare hidden void @"_ZN4core3ptr24drop_in_place$LT$i32$GT$17hb6decb5d90d320b6E"(ptr align 4) unnamed_addr #0

attributes #0 = { inlinehint nonlazybind uwtable "probe-stack"="inline-asm" "target-cpu"="x86-64" }

!llvm.module.flags = !{!13, !14, !15, !16}
!llvm.dbg.cu = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "<i32 as _>::{vtable}", scope: null, file: !2, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<unknown>", directory: "")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "<i32 as _>::{vtable_type}", file: !2, size: 192, align: 64, flags: DIFlagArtificial, elements: !4, vtableHolder: !11, templateParams: !12, identifier: "c43795a3e66ad62fad3e5bb73c25fc9")
!4 = !{!5, !8, !10}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "drop_in_place", scope: !3, file: !2, baseType: !6, size: 64, align: 64)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()", baseType: !7, size: 64, align: 64, dwarfAddressSpace: 0)
!7 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "size", scope: !3, file: !2, baseType: !9, size: 64, align: 64, offset: 64)
!9 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "align", scope: !3, file: !2, baseType: !9, size: 64, align: 64, offset: 128)
!11 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!12 = !{}
!13 = !{i32 8, !"PIC Level", i32 2}
!14 = !{i32 2, !"RtLibUseGOT", i32 1}
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !2, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !18)
!18 = !{!0}
