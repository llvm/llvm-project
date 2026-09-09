; RUN: llc < %s -mtriple=nvptx64 -O0 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; At -O0, keep the intermediate (base + constant) node when it has an
; associated debug value, so we do not fold the immediate into [reg+imm].
define ptx_kernel void @keep_dbg_value_addr(ptr addrspace(1) %in, ptr addrspace(1) %out) !dbg !3 {
entry:
  call void @llvm.dbg.value(metadata ptr addrspace(1) %in, metadata !8, metadata !DIExpression()), !dbg !10
  %idx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 1, !dbg !11
  call void @llvm.dbg.value(metadata ptr addrspace(1) %idx, metadata !9, metadata !DIExpression()), !dbg !11
  %v = load i32, ptr addrspace(1) %idx, align 4, !dbg !12
  store i32 %v, ptr addrspace(1) %out, align 4, !dbg !13
  ret void, !dbg !14
}

; CHECK-LABEL: .visible .entry keep_dbg_value_addr(
; CHECK: ld.param.b64 [[IN:%rd[0-9]+]], [keep_dbg_value_addr_param_0];
; CHECK: add.s64 [[ADDR:%rd[0-9]+]], [[IN]], 4;
; CHECK: ld.global.b32 %r{{[0-9]+}}, {{\[}}[[ADDR]]{{\]}};
; CHECK: st.global.b32

; Without a debug value on the GEP result, the immediate offset can still fold.
define ptx_kernel void @fold_without_dbg_value_addr(ptr addrspace(1) %in, ptr addrspace(1) %out) !dbg !20 {
entry:
  call void @llvm.dbg.value(metadata ptr addrspace(1) %in, metadata !25, metadata !DIExpression()), !dbg !27
  %idx = getelementptr inbounds i32, ptr addrspace(1) %in, i64 1, !dbg !28
  %v = load i32, ptr addrspace(1) %idx, align 4, !dbg !29
  store i32 %v, ptr addrspace(1) %out, align 4, !dbg !30
  ret void, !dbg !31
}

; CHECK-LABEL: .visible .entry fold_without_dbg_value_addr(
; CHECK: ld.param.b64 [[IN2:%rd[0-9]+]], [fold_without_dbg_value_addr_param_0];
; CHECK: ld.global.b32 %r{{[0-9]+}}, {{\[}}[[IN2]]+4{{\]}};
; CHECK: st.global.b32

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "unit-test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "indexed_addressing_debug.cu", directory: ".")
!2 = !{i32 1, !"Debug Info Version", i32 3}

!3 = distinct !DISubprogram(name: "keep_dbg_value_addr", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{!6, !7, !7}
!6 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void")
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64, dwarfAddressSpace: 1)
!8 = !DILocalVariable(name: "in", arg: 1, scope: !3, file: !1, line: 1, type: !7)
!9 = !DILocalVariable(name: "idx", scope: !3, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 1, column: 1, scope: !3)
!11 = !DILocation(line: 2, column: 1, scope: !3)
!12 = !DILocation(line: 3, column: 1, scope: !3)
!13 = !DILocation(line: 4, column: 1, scope: !3)
!14 = !DILocation(line: 5, column: 1, scope: !3)

!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!20 = distinct !DISubprogram(name: "fold_without_dbg_value_addr", scope: !1, file: !1, line: 10, type: !4, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!25 = !DILocalVariable(name: "in", arg: 1, scope: !20, file: !1, line: 10, type: !7)
!27 = !DILocation(line: 10, column: 1, scope: !20)
!28 = !DILocation(line: 11, column: 1, scope: !20)
!29 = !DILocation(line: 12, column: 1, scope: !20)
!30 = !DILocation(line: 13, column: 1, scope: !20)
!31 = !DILocation(line: 14, column: 1, scope: !20)
