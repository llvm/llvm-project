; ModuleID = 'C:\llvm-project\clang\test\CodeGenHLSL\debug\rwbuffer_debug_info.hlsl'
source_filename = "C:\\llvm-project\\clang\\test\\CodeGenHLSL\\debug\\rwbuffer_debug_info.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.6-pc-shadermodel6.6-compute"

%"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", float, 1, 0, 0) }

@_ZL3Out = internal global %"class.hlsl::RWBuffer" poison, align 4, !dbg !0
@.str = private unnamed_addr constant [4 x i8] c"Out\00", align 1

; Function Attrs: alwaysinline convergent nounwind
define internal void @__cxx_global_var_init() #0 !dbg !56 {
entry:
  call void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(ptr dead_on_unwind writable sret(%"class.hlsl::RWBuffer") align 4 @_ZL3Out, i32 noundef 7, i32 noundef 4, i32 noundef 1, i32 noundef 0, ptr noundef @.str) #5, !dbg !59
  ret void, !dbg !59
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc(ptr dead_on_unwind noalias writable sret(%"class.hlsl::RWBuffer") align 4 %agg.result, i32 noundef %registerNo, i32 noundef %spaceNo, i32 noundef %range, i32 noundef %index, ptr noundef %name) #1 align 2 !dbg !60 {
entry:
  %registerNo.addr = alloca i32, align 4
  %spaceNo.addr = alloca i32, align 4
  %range.addr = alloca i32, align 4
  %index.addr = alloca i32, align 4
  %name.addr = alloca ptr, align 4
  %tmp = alloca target("dx.TypedBuffer", float, 1, 0, 0), align 4
  store i32 %registerNo, ptr %registerNo.addr, align 4
    #dbg_declare(ptr %registerNo.addr, !62, !DIExpression(), !63)
  store i32 %spaceNo, ptr %spaceNo.addr, align 4
    #dbg_declare(ptr %spaceNo.addr, !64, !DIExpression(), !63)
  store i32 %range, ptr %range.addr, align 4
    #dbg_declare(ptr %range.addr, !65, !DIExpression(), !63)
  store i32 %index, ptr %index.addr, align 4
    #dbg_declare(ptr %index.addr, !66, !DIExpression(), !63)
  store ptr %name, ptr %name.addr, align 4
    #dbg_declare(ptr %name.addr, !67, !DIExpression(), !63)
    #dbg_declare(ptr %tmp, !68, !DIExpression(), !63)
  %0 = load i32, ptr %registerNo.addr, align 4, !dbg !69
  %1 = load i32, ptr %spaceNo.addr, align 4, !dbg !69
  %2 = load i32, ptr %range.addr, align 4, !dbg !69
  %3 = load i32, ptr %index.addr, align 4, !dbg !69
  %4 = load ptr, ptr %name.addr, align 4, !dbg !69
  %5 = call target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32 %1, i32 %0, i32 %2, i32 %3, i1 false, ptr %4), !dbg !69
  call void @_ZN4hlsl8RWBufferIfEC1EU9_Res_u_CTfu17__hlsl_resource_t(ptr noundef nonnull align 4 dereferenceable(4) %agg.result, target("dx.TypedBuffer", float, 1, 0, 0) %5) #5, !dbg !69
  ret void, !dbg !70
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define internal void @_Z4maini(i32 noundef %GI) #1 !dbg !71 {
entry:
  %GI.addr = alloca i32, align 4
  store i32 %GI, ptr %GI.addr, align 4
    #dbg_declare(ptr %GI.addr, !74, !DIExpression(), !75)
  %0 = load i32, ptr %GI.addr, align 4, !dbg !76
  %call = call noundef nonnull align 4 dereferenceable(4) ptr @_ZN4hlsl8RWBufferIfEixEj(ptr noundef nonnull align 4 dereferenceable(4) @_ZL3Out, i32 noundef %0) #5, !dbg !77
  store float 0.000000e+00, ptr %call, align 4, !dbg !78
  ret void, !dbg !79
}

; Function Attrs: convergent noinline norecurse optnone
define void @main() #2 {
entry:
  call void @_GLOBAL__sub_I_rwbuffer_debug_info.hlsl()
  %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
  call void @_Z4maini(i32 %0)
  ret void
}

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.dx.flattened.thread.id.in.group() #3

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define linkonce_odr hidden noundef nonnull align 4 dereferenceable(4) ptr @_ZN4hlsl8RWBufferIfEixEj(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 noundef %Index) #1 align 2 !dbg !80 {
entry:
  %this.addr = alloca ptr, align 4
  %Index.addr = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 4
    #dbg_declare(ptr %this.addr, !81, !DIExpression(), !83)
  store i32 %Index, ptr %Index.addr, align 4
    #dbg_declare(ptr %Index.addr, !84, !DIExpression(), !85)
  %this1 = load ptr, ptr %this.addr, align 4
  %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %this1, i32 0, i32 0, !dbg !83
  %0 = load target("dx.TypedBuffer", float, 1, 0, 0), ptr %__handle, align 4, !dbg !83
  %1 = load i32, ptr %Index.addr, align 4, !dbg !83
  %2 = call ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t(target("dx.TypedBuffer", float, 1, 0, 0) %0, i32 %1), !dbg !83
  ret ptr %2, !dbg !86
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("dx.TypedBuffer", float, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_f32_1_0_0t(i32, i32, i32, i32, i1, ptr) #4

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC1EU9_Res_u_CTfu17__hlsl_resource_t(ptr noundef nonnull align 4 dereferenceable(4) %this, target("dx.TypedBuffer", float, 1, 0, 0) %handle) unnamed_addr #1 align 2 !dbg !87 {
entry:
  %this.addr = alloca ptr, align 4
  %handle.addr = alloca target("dx.TypedBuffer", float, 1, 0, 0), align 4
  store ptr %this, ptr %this.addr, align 4
    #dbg_declare(ptr %this.addr, !88, !DIExpression(), !89)
  store target("dx.TypedBuffer", float, 1, 0, 0) %handle, ptr %handle.addr, align 4
    #dbg_declare(ptr %handle.addr, !90, !DIExpression(), !91)
  %this1 = load ptr, ptr %this.addr, align 4
  %0 = load target("dx.TypedBuffer", float, 1, 0, 0), ptr %handle.addr, align 4, !dbg !92
  call void @_ZN4hlsl8RWBufferIfEC2EU9_Res_u_CTfu17__hlsl_resource_t(ptr noundef nonnull align 4 dereferenceable(4) %this1, target("dx.TypedBuffer", float, 1, 0, 0) %0) #5, !dbg !92
  ret void, !dbg !93
}

; Function Attrs: alwaysinline convergent mustprogress norecurse nounwind
define linkonce_odr hidden void @_ZN4hlsl8RWBufferIfEC2EU9_Res_u_CTfu17__hlsl_resource_t(ptr noundef nonnull align 4 dereferenceable(4) %this, target("dx.TypedBuffer", float, 1, 0, 0) %handle) unnamed_addr #1 align 2 !dbg !95 {
entry:
  %this.addr = alloca ptr, align 4
  %handle.addr = alloca target("dx.TypedBuffer", float, 1, 0, 0), align 4
  store ptr %this, ptr %this.addr, align 4
    #dbg_declare(ptr %this.addr, !96, !DIExpression(), !97)
  store target("dx.TypedBuffer", float, 1, 0, 0) %handle, ptr %handle.addr, align 4
    #dbg_declare(ptr %handle.addr, !98, !DIExpression(), !99)
  %this1 = load ptr, ptr %this.addr, align 4
  %0 = load target("dx.TypedBuffer", float, 1, 0, 0), ptr %handle.addr, align 4, !dbg !100
  %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %this1, i32 0, i32 0, !dbg !100
  store target("dx.TypedBuffer", float, 1, 0, 0) %0, ptr %__handle, align 4, !dbg !100
  ret void, !dbg !102
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_1_0_0t(target("dx.TypedBuffer", float, 1, 0, 0), i32) #4

; Function Attrs: alwaysinline convergent nounwind
define internal void @_GLOBAL__sub_I_rwbuffer_debug_info.hlsl() #0 !dbg !104 {
entry:
  call void @__cxx_global_var_init(), !dbg !106
  ret void
}

attributes #0 = { alwaysinline convergent nounwind "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { alwaysinline convergent mustprogress norecurse nounwind "approx-func-fp-math"="true" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent noinline norecurse optnone "approx-func-fp-math"="true" "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #5 = { convergent }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!51, !52, !53}
!dx.valver = !{!54}
!llvm.ident = !{!55}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Out", linkageName: "_ZL3Out", scope: !2, file: !5, line: 9, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !3, producer: "clang version 22.0.0git (C:/llvm-project/clang e5cfb97d26f63942eea3c9353680fa4e669b02e7)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "C:\\llvm-project\\clang\\test\\CodeGenHLSL\\debug\\<stdin>", directory: "")
!4 = !{!0}
!5 = !DIFile(filename: "C:\\llvm-project\\clang\\test\\CodeGenHLSL\\debug\\rwbuffer_debug_info.hlsl", directory: "")
!6 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "RWBuffer<float>", scope: !7, size: 32, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !8, templateParams: !49, identifier: "_ZTSN4hlsl8RWBufferIfEE")
!7 = !DINamespace(name: "hlsl", scope: null)
!8 = !{!9, !12, !16, !19, !27, !28, !31, !34, !42, !46}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "__handle", scope: !6, file: !3, line: 9, baseType: !10, size: 32)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 32)
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "__hlsl_resource_t", file: !3, flags: DIFlagFwdDecl)
!12 = !DISubprogram(name: "RWBuffer", scope: !6, file: !3, type: !13, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!13 = !DISubroutineType(types: !14)
!14 = !{null, !15}
!15 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !6, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!16 = !DISubprogram(name: "RWBuffer", scope: !6, file: !3, type: !17, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !15, !10}
!19 = !DISubprogram(name: "__createFromBinding", linkageName: "_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc", scope: !6, file: !3, type: !20, flags: DIFlagPublic | DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!20 = !DISubroutineType(types: !21)
!21 = !{!6, !22, !22, !23, !22, !24}
!22 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !25, size: 32)
!25 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !26)
!26 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!27 = !DISubprogram(name: "__createFromImplicitBinding", linkageName: "_ZN4hlsl8RWBufferIfE27__createFromImplicitBindingEjjijPKc", scope: !6, file: !3, type: !20, flags: DIFlagPublic | DIFlagPrototyped | DIFlagStaticMember, spFlags: 0)
!28 = !DISubprogram(name: "RWBuffer", scope: !6, file: !3, type: !29, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!29 = !DISubroutineType(types: !30)
!30 = !{null, !15, !22, !22, !23, !22, !24}
!31 = !DISubprogram(name: "RWBuffer", scope: !6, file: !3, type: !32, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!32 = !DISubroutineType(types: !33)
!33 = !{null, !15, !22, !23, !22, !22, !24}
!34 = !DISubprogram(name: "operator[]", linkageName: "_ZNK4hlsl8RWBufferIfEixEj", scope: !6, file: !3, type: !35, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!35 = !DISubroutineType(types: !36)
!36 = !{!37, !40, !22}
!37 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !38, size: 32)
!38 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !39)
!39 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!40 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !41, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!41 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!42 = !DISubprogram(name: "operator[]", linkageName: "_ZN4hlsl8RWBufferIfEixEj", scope: !6, file: !3, type: !43, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!43 = !DISubroutineType(types: !44)
!44 = !{!45, !15, !22}
!45 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !39, size: 32)
!46 = !DISubprogram(name: "Load", linkageName: "_ZN4hlsl8RWBufferIfE4LoadEj", scope: !6, file: !3, type: !47, flags: DIFlagPublic | DIFlagPrototyped, spFlags: 0)
!47 = !DISubroutineType(types: !48)
!48 = !{!39, !15, !22}
!49 = !{!50}
!50 = !DITemplateTypeParameter(name: "element_type", type: !39)
!51 = !{i32 7, !"Dwarf Version", i32 4}
!52 = !{i32 2, !"Debug Info Version", i32 3}
!53 = !{i32 1, !"wchar_size", i32 4}
!54 = !{i32 1, i32 8}
!55 = !{!"clang version 22.0.0git (C:/llvm-project/clang e5cfb97d26f63942eea3c9353680fa4e669b02e7)"}
!56 = distinct !DISubprogram(name: "__cxx_global_var_init", scope: !3, file: !3, type: !57, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!57 = !DISubroutineType(types: !58)
!58 = !{null}
!59 = !DILocation(line: 0, scope: !56)
!60 = distinct !DISubprogram(name: "__createFromBinding", linkageName: "_ZN4hlsl8RWBufferIfE19__createFromBindingEjjijPKc", scope: !6, file: !3, line: 14, type: !20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !19, retainedNodes: !61)
!61 = !{}
!62 = !DILocalVariable(name: "registerNo", arg: 1, scope: !60, file: !3, type: !22)
!63 = !DILocation(line: 0, column: 1, scope: !60)
!64 = !DILocalVariable(name: "spaceNo", arg: 2, scope: !60, file: !3, type: !22)
!65 = !DILocalVariable(name: "range", arg: 3, scope: !60, file: !3, type: !23)
!66 = !DILocalVariable(name: "index", arg: 4, scope: !60, file: !3, type: !22)
!67 = !DILocalVariable(name: "name", arg: 5, scope: !60, file: !3, type: !24)
!68 = !DILocalVariable(name: "tmp", scope: !60, file: !3, type: !10)
!69 = !DILocation(line: 0, scope: !60)
!70 = !DILocation(line: 14, column: 1, scope: !60)
!71 = distinct !DISubprogram(name: "main", linkageName: "_Z4maini", scope: !5, file: !5, line: 12, type: !72, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !61)
!72 = !DISubroutineType(types: !73)
!73 = !{null, !23}
!74 = !DILocalVariable(name: "GI", arg: 1, scope: !71, file: !5, line: 12, type: !23)
!75 = !DILocation(line: 12, column: 15, scope: !71)
!76 = !DILocation(line: 13, column: 7, scope: !71)
!77 = !DILocation(line: 13, column: 3, scope: !71)
!78 = !DILocation(line: 13, column: 11, scope: !71)
!79 = !DILocation(line: 14, column: 1, scope: !71)
!80 = distinct !DISubprogram(name: "operator[]", linkageName: "_ZN4hlsl8RWBufferIfEixEj", scope: !6, file: !3, line: 14, type: !43, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !42, retainedNodes: !61)
!81 = !DILocalVariable(name: "this", arg: 1, scope: !80, type: !82, flags: DIFlagArtificial | DIFlagObjectPointer)
!82 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !6, size: 32)
!83 = !DILocation(line: 0, scope: !80)
!84 = !DILocalVariable(name: "Index", arg: 2, scope: !80, file: !3, type: !22)
!85 = !DILocation(line: 0, column: 1, scope: !80)
!86 = !DILocation(line: 14, column: 1, scope: !80)
!87 = distinct !DISubprogram(name: "RWBuffer", linkageName: "_ZN4hlsl8RWBufferIfEC1EU9_Res_u_CTfu17__hlsl_resource_t", scope: !6, file: !3, line: 14, type: !17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !16, retainedNodes: !61)
!88 = !DILocalVariable(name: "this", arg: 1, scope: !87, type: !82, flags: DIFlagArtificial | DIFlagObjectPointer)
!89 = !DILocation(line: 0, scope: !87)
!90 = !DILocalVariable(name: "handle", arg: 2, scope: !87, file: !3, type: !10)
!91 = !DILocation(line: 0, column: 1, scope: !87)
!92 = !DILocation(line: 14, column: 1, scope: !87)
!93 = !DILocation(line: 14, column: 1, scope: !94)
!94 = !DILexicalBlockFile(scope: !87, file: !5, discriminator: 0)
!95 = distinct !DISubprogram(name: "RWBuffer", linkageName: "_ZN4hlsl8RWBufferIfEC2EU9_Res_u_CTfu17__hlsl_resource_t", scope: !6, file: !3, line: 14, type: !17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, declaration: !16, retainedNodes: !61)
!96 = !DILocalVariable(name: "this", arg: 1, scope: !95, type: !82, flags: DIFlagArtificial | DIFlagObjectPointer)
!97 = !DILocation(line: 0, scope: !95)
!98 = !DILocalVariable(name: "handle", arg: 2, scope: !95, file: !3, type: !10)
!99 = !DILocation(line: 0, column: 1, scope: !95)
!100 = !DILocation(line: 0, scope: !101)
!101 = distinct !DILexicalBlock(scope: !95, file: !5, line: 14, column: 1)
!102 = !DILocation(line: 14, column: 1, scope: !103)
!103 = !DILexicalBlockFile(scope: !95, file: !5, discriminator: 0)
!104 = distinct !DISubprogram(linkageName: "_GLOBAL__sub_I_rwbuffer_debug_info.hlsl", scope: !3, file: !3, type: !105, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!105 = !DISubroutineType(types: !61)
!106 = !DILocation(line: 0, scope: !104)
