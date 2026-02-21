; RUN: llc %s -o %t.o -mcpu=gfx1030 -filetype=obj -O0
; RUN: llvm-debuginfo-analyzer --report=debugger --print=symbols --attribute=level,offset %t.o | FileCheck %s

; The test compiles this module using the AMDGPU backend under `-O0`,
; and makes sure `llvm-debuginfo-analyzer --report=debugger` works for it.

; CHECK: {File}
; CHECK: {CompileUnit} basic_var.hlsl
; CHECK: {Function} main
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:7 [main]
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:11 [main]
; CHECK: {Parameter} dtid: uint3 : reg{{.+}}, piece 4
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:17 [main]
; CHECK: {Parameter} dtid: uint3 : reg{{.+}}, piece 4
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:11 [main]
; CHECK: {Parameter} dtid: uint3 : reg{{.+}}, piece 4
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:14 [main]
; CHECK-DAG: {Parameter} dtid: uint3 : reg{{.+}}, piece 4
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:17 [main]
; CHECK-DAG: {Parameter} dtid: uint3 : reg{{.+}}, piece 4
; CHECK-DAG: {Variable} my_var2: float : reg{{.+}}
; CHECK: {Line} [0x{{[0-9a-f]+}}][003] basic_var.hlsl:19 [main]
; CHECK-DAG: {Parameter} dtid: uint3 : reg{{.+}}, piece 4
; CHECK-DAG: {Variable} my_var2: float : reg{{.+}}

source_filename = "module"
target triple = "amdgcn-amd-amdpal"

%dx.types.ResRet.f32 = type { float, float, float, float, i32 }

; Function Attrs: memory(readwrite)
define dllexport amdgpu_cs void @_amdgpu_cs_main(i32 inreg noundef %globalTable, i32 inreg noundef %userdata4, <3 x i32> inreg noundef %WorkgroupId, i32 inreg noundef %MultiDispatchInfo, <3 x i32> noundef %LocalInvocationId) #0 !dbg !14 {
  %LocalInvocationId.i0 = extractelement <3 x i32> %LocalInvocationId, i64 0, !dbg !28
  %WorkgroupId.i0 = extractelement <3 x i32> %WorkgroupId, i64 0, !dbg !28
  %1 = call i64 @llvm.amdgcn.s.getpc(), !dbg !28
  %2 = shl i32 %WorkgroupId.i0, 6, !dbg !28
  %3 = add i32 %LocalInvocationId.i0, %2, !dbg !28
    #dbg_value(i32 %3, !29, !DIExpression(DW_OP_LLVM_fragment, 0, 32), !28)
  %4 = and i64 %1, -4294967296, !dbg !30
  %5 = zext i32 %userdata4 to i64, !dbg !30
  %6 = or disjoint i64 %4, %5, !dbg !30
  %7 = inttoptr i64 %6 to ptr addrspace(4), !dbg !30
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(4) %7, i32 4), "dereferenceable"(ptr addrspace(4) %7, i32 -1) ], !dbg !30
  %8 = load <4 x i32>, ptr addrspace(4) %7, align 4, !dbg !30, !invariant.load !2
  %9 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %8, i32 %3, i32 0, i32 0, i32 0), !dbg !30
    #dbg_value(%dx.types.ResRet.f32 poison, !31, !DIExpression(), !32)
  %10 = fmul reassoc arcp contract afn float %9, 2.000000e+00, !dbg !33
    #dbg_value(float %10, !34, !DIExpression(), !35)
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(4) %7, i32 4), "dereferenceable"(ptr addrspace(4) %7, i32 -1) ], !dbg !36
  %11 = getelementptr i8, ptr addrspace(4) %7, i64 32, !dbg !36
  %.upto01 = insertelement <4 x float> poison, float %10, i64 0, !dbg !36
  %12 = shufflevector <4 x float> %.upto01, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !36
  %13 = load <4 x i32>, ptr addrspace(4) %11, align 4, !dbg !36, !invariant.load !2
  call void @llvm.amdgcn.struct.buffer.store.format.v4f32(<4 x float> %12, <4 x i32> %13, i32 %3, i32 0, i32 0, i32 0), !dbg !36
  ret void, !dbg !37
}

declare noundef i64 @llvm.amdgcn.s.getpc() #1

declare void @llvm.assume(i1 noundef) #2

declare void @llvm.amdgcn.struct.buffer.store.format.v4f32(<4 x float>, <4 x i32>, i32, i32, i32, i32 immarg) #3

declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32 immarg) #4

attributes #0 = { memory(readwrite) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(write) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "dxcoob 1.7.2308.16 (52da17e29)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "tests\\basic_var.hlsl", directory: "")
!2 = !{}
!3 = !{!4, !10}
!4 = distinct !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "u0", linkageName: "\01?u0@@3V?$RWBuffer@M@@A", scope: !0, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true)
!6 = !DICompositeType(tag: DW_TAG_class_type, name: "RWBuffer<float>", file: !1, line: 2, size: 32, align: 32, elements: !2, templateParams: !7)
!7 = !{!8}
!8 = !DITemplateTypeParameter(name: "element", type: !9)
!9 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!10 = distinct !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = !DIGlobalVariable(name: "u1", linkageName: "\01?u1@@3V?$RWBuffer@M@@A", scope: !0, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true)
!12 = !{i32 2, !"Dwarf Version", i32 5}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !15, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint3", file: !1, baseType: !18)
!18 = !DICompositeType(tag: DW_TAG_class_type, name: "vector<unsigned int, 3>", file: !1, size: 96, align: 32, elements: !19, templateParams: !24)
!19 = !{!20, !22, !23}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !18, file: !1, baseType: !21, size: 32, align: 32, flags: DIFlagPublic)
!21 = !DIBasicType(name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!22 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !18, file: !1, baseType: !21, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "z", scope: !18, file: !1, baseType: !21, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!24 = !{!25, !26}
!25 = !DITemplateTypeParameter(name: "element", type: !21)
!26 = !DITemplateValueParameter(name: "element_count", type: !27, value: i32 3)
!27 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!28 = !DILocation(line: 7, column: 17, scope: !14)
!29 = !DILocalVariable(name: "dtid", arg: 1, scope: !14, file: !1, line: 7, type: !17)
!30 = !DILocation(line: 11, column: 18, scope: !14)
!31 = !DILocalVariable(name: "my_var", scope: !14, file: !1, line: 11, type: !9)
!32 = !DILocation(line: 11, column: 9, scope: !14)
!33 = !DILocation(line: 14, column: 26, scope: !14)
!34 = !DILocalVariable(name: "my_var2", scope: !14, file: !1, line: 14, type: !9)
!35 = !DILocation(line: 14, column: 9, scope: !14)
!36 = !DILocation(line: 17, column: 14, scope: !14)
!37 = !DILocation(line: 19, column: 1, scope: !14)
