; RUN: llvm-as < %s | llvm-dis | llc -mtriple=x86_64 -O0 -filetype=obj -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s

; A test to verify the use of a DIDerivedType as a bound of a
; DISubrangeType.

; CHECK: DW_TAG_array_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] {{.*}}
; CHECK: DW_TAG_subrange_type {{.*}}
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4] {{.*}}
; CHECK-NEXT: DW_AT_upper_bound [DW_FORM_ref4] {{.*}}

; ModuleID = 'vla.ads'
source_filename = "vla.ads"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%vla__array_type_FP = type { ptr, ptr }
%vla__record_type_I = type <{ i32 }>

@vla_E = dso_local global i16 0, align 2, !dbg !0

; Function Attrs: inlinehint
define dso_local void @vla__array_typeIP(%vla__array_type_FP %_init) #0 !dbg !10 {
entry:
  %0 = extractvalue %vla__array_type_FP %_init, 1, !dbg !15
  %1 = call ptr @llvm.invariant.start.p0(i64 8, ptr %0), !dbg !15
  ret void, !dbg !15
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare ptr @llvm.invariant.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: inlinehint
define dso_local void @vla__record_typeIP(ptr noalias nocapture nonnull %_init, i32 %l1) #0 !dbg !16 {
entry:
    #dbg_declare(ptr %_init, !31, !DIExpression(), !32)
    #dbg_value(i32 %l1, !33, !DIExpression(), !34)
  %0 = getelementptr inbounds %vla__record_type_I, ptr %_init, i32 0, i32 0, !dbg !32
  store i32 %l1, ptr %0, align 4, !dbg !32, !tbaa !35
  ret void, !dbg !32
}

attributes #0 = { inlinehint }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!6, !7, !8, !9}
!llvm.dbg.cu = !{!2}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "vla_E", scope: !2, file: !3, line: 16, type: !5, isLocal: false, isDefinition: true, align: 16)
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, producer: "GNAT/LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false)
!3 = !DIFile(filename: "vla.ads", directory: "")
!4 = !{!0}
!5 = !DIBasicType(name: "short_integer", size: 16, encoding: DW_ATE_signed)
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"PIE Level", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = distinct !DISubprogram(name: "vla__array_typeIP", scope: !3, file: !3, line: 17, type: !11, scopeLine: 17, spFlags: DISPFlagDefinition, unit: !2)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, align: 64, dwarfAddressSpace: 0)
!14 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "vla__array_type")
!15 = !DILocation(line: 17, column: 9, scope: !10)
!16 = distinct !DISubprogram(name: "vla__record_typeIP", scope: !3, file: !3, line: 18, type: !17, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !30)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !23}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 64, align: 64, dwarfAddressSpace: 0)
!20 = !DICompositeType(tag: DW_TAG_structure_type, name: "vla__record_type", file: !3, line: 18, size: !DIExpression(DW_OP_push_object_address, DW_OP_deref_size, 4, DW_OP_constu, 32, DW_OP_mul, DW_OP_constu, 32, DW_OP_plus), align: 32, elements: !21, identifier: "vla__record_type")
!21 = !{!22, !26}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "l1", file: !3, line: 18, baseType: !23)
!23 = !DISubrangeType(name: "natural", file: !24, line: 1, size: 32, align: 32, baseType: !25, lowerBound: i64 0, upperBound: i64 2147483647)
!24 = !DIFile(filename: "system.ads", directory: "/home/tromey/AdaCore/gnat-llvm/llvm-interface//lib/gnat-llvm/x86_64-unknown-linux-gnu/rts-native/adainclude/")
!25 = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "a1", file: !3, line: 19, baseType: !27, offset: 32)
!27 = !DICompositeType(tag: DW_TAG_array_type, scope: !20, file: !3, line: 19, baseType: !25, align: 32, elements: !28)
!28 = !{!29}
!29 = !DISubrangeType(baseType: !25, lowerBound: i64 1, upperBound: !22)
!30 = !{}
!31 = !DILocalVariable(name: "_init", arg: 1, scope: !16, file: !3, line: 18, type: !20, flags: DIFlagArtificial)
!32 = !DILocation(line: 18, column: 9, scope: !16)
!33 = !DILocalVariable(name: "l1", arg: 2, scope: !16, file: !3, line: 18, type: !23, flags: DIFlagArtificial)
!34 = !DILocation(line: 18, column: 22, scope: !16)
!35 = !{!36, !36, i64 0, i64 4}
!36 = !{!37, i64 4, !"natural#T5"}
!37 = !{!38, i64 4, !"natural#TN"}
!38 = !{!39, i64 4, !"integerB#TN"}
!39 = !{!"Ada Root"}
