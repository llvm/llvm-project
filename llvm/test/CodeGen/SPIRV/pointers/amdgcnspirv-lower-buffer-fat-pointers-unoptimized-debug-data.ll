; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

target triple = "spirv64-amd-amdhsa"

; CHECK: OpCapability USMStorageClassesINTEL
; CHECK: OpExtension "SPV_INTEL_usm_storage_classes"
; CHECK: OpExtension "SPV_KHR_non_semantic_info"
; CHECK: OpName %[[#DEBUG_STASH_POINTER:]] "debug_stash_pointer"
; CHECK: OpName %[[#OPAQUE_PTR_CAST_P7_P8:]] "spirv.llvm_spv_opaque_ptr_cast_p7_p8"
; CHECK: %[[#INT8_TY:]] = OpTypeInt 8 0
; CHECK: %3 = OpTypePointer HostOnlyINTEL %2
	; %4 = OpTypeInt 32 0
	; %5 = OpTypeFloat 32
	; %6 = OpTypeFunction %5 %3 %4 %3
	; %7 = OpTypePointer DeviceOnlyINTEL %2
	; %8 = OpTypeFunction %7 %3
	; %9 = OpTypePointer Function %7
	; %10 = OpTypePointer DeviceOnlyINTEL %5
	; %11 = OpTypePointer Function %10
	; %12 = OpTypePointer DeviceOnlyINTEL %10
	; %13 = OpTypePointer CrossWorkgroup %4
	; %14 = OpConstantNull %4
	; %15 = OpTypePointer Function %12
	; %16 = OpVariable %13 CrossWorkgroup %14
	; %41 = OpTypeVoid
	; %42 = OpConstant %4 100
	; %43 = OpConstant %4 0
	; %44 = OpExtInst %41 %37 DebugSource %38
	; %45 = OpExtInst %41 %37 DebugCompilationUnit %42 %43 %44 %43
	; %46 = OpConstant %4 32
	; %47 = OpConstant %4 6
	; %48 = OpExtInst %41 %37 DebugTypeBasic %39 %46 %47 %43
	; %49 = OpConstant %4 256
	; %50 = OpExtInst %41 %37 DebugTypeBasic %40 %49 %47 %43
	; %17 = OpFunction %7 None %8
	; %18 = OpFunctionParameter %3
	; OpFunctionEnd

; CHECK: %[[#DEBUG_STASH_POINTER]] = OpFunction
; CHECK: %[[#BUF:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_8:]]
; CHECK: %[[#IDX:]] = OpFunctionParameter %[[#INT32_TY:]]
; CHECK: %[[#AUX:]] = OpFunctionParameter %[[#I8PTR_ADDRSPACE_8]]
;	CHECK: %[[#BUF_PTR_VAR:]] = OpVariable %[[#PTRI8PTR_ADDRSPACE_7_PRIVATE:]] Function
;	CHECK: %[[#AUX_PTR_VAR:]] = OpVariable %[[#PTRI8PTR_ADDRSPACE_7_PRIVATE:]] Function
; CHECK: DEBUG_VALUE: debug_stash_pointer:1 <- %11
; CHECK: DEBUG_VALUE: debug_stash_pointer:2 <- %14
;	CHECK: %[[#BUF_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7:]] %[[#OPAQUE_PTR_CAST_P7_P8]] %[[#BUF]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:3 <- %16
; CHECK: OpStore %[[#BUF_PTR_VAR]] %[[#BUF_TO_AS7]]
;	CHECK: %[[#AUX_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P7_P8]] %[[#AUX]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:4 <- %21
; CHECK: OpStore %[[#AUX_PTR_VAR]] %[[#AUX_TO_AS7]]
; CHECK: %[[#BC_BUF_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE:]] = OpBitcast %[[#PTRI32PTR_ADDRSPACE_7_PRIVATE:]] %[[#BUF_PTR_VAR]]
;	CHECK: %[[#BUF_PTR_2:]] = OpLoad %[[#I32PTR_ADDRSPACE_7:]] %[[#BC_BUF_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:5 <- %22
; CHECK: %[[#BUF_PTR_3:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BUF_PTR_2]] %[[#IDX]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:6 <- %23
; CHECK: %[[#BC_BUF_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE_1:]] = OpBitcast %[[#PTRI32PTR_ADDRSPACE_7_PRIVATE]] %[[#BUF_PTR_VAR]]
; CHECK: OpStore %[[#BC_BUF_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE_1]] %[[#BUF_PTR_3]]
; CHECK: %[[#BC_BUF_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE_2:]] = OpBitcast %[[#PTRI32PTR_ADDRSPACE_7_PRIVATE]] %[[#BUF_PTR_VAR]]
;	CHECK: %[[#BUF_PTR_4:]] = OpLoad %[[#PTRFLOATPTR_ADDRSPACE_7_PRIVATE:]] %[[#BC_BUF_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE_2]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:7 <- %25
; CHECK: %[[#RET:]] = OpLoad %[[#FLOAT:]] %[[#BUF_PTR_4]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:8 <- %26
; CHECK: %[[#BC_AUX_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE_3:]] = OpBitcast %[[#PTRPTRI32PTR_ADDRSPACE_7_PRIVATE_PRIVATE:]] %[[#AUX_PTR_VAR]]
; CHECK: %[[#AUX_PTR_2:]] = OpLoad %[[#]] %[[#BC_AUX_PTR_VAR_TO_PTRI32PTR_ADDRSPACE_7_PRIVATE_3]]
; CHECK: DEBUG_VALUE: debug_stash_pointer:9 <- %27
;	CHECK: OpStore %[[#AUX_PTR_2]] %[[#BUF_PTR_4]]
; CHECK: OpReturnValue %[[#RET]]

define float @debug_stash_pointer(ptr addrspace(8) %buf, i32 %idx, ptr addrspace(8) %aux) !dbg !5 {
  %buf.ptr.var = alloca ptr addrspace(7), align 32, !dbg !20
  call void @llvm.dbg.value(metadata ptr %buf.ptr.var, metadata !9, metadata !DIExpression()), !dbg !20
  %aux.ptr.var = alloca ptr addrspace(7), align 32, !dbg !21
  call void @llvm.dbg.value(metadata ptr %aux.ptr.var, metadata !11, metadata !DIExpression()), !dbg !21
  %buf.ptr = addrspacecast ptr addrspace(8) %buf to ptr addrspace(7), !dbg !22
  call void @llvm.dbg.value(metadata ptr addrspace(7) %buf.ptr, metadata !12, metadata !DIExpression()), !dbg !22
  store ptr addrspace(7) %buf.ptr, ptr %buf.ptr.var, align 32, !dbg !23, !DIAssignID !40
  call void @llvm.dbg.assign(metadata ptr addrspace(7) %buf.ptr, metadata !12, metadata !DIExpression(), metadata !40, metadata ptr %buf.ptr.var, metadata !DIExpression()), !dbg !20
  %aux.ptr = addrspacecast ptr addrspace(8) %aux to ptr addrspace(7), !dbg !24
  call void @llvm.dbg.value(metadata ptr addrspace(7) %aux.ptr, metadata !14, metadata !DIExpression()), !dbg !24
  store ptr addrspace(7) %aux.ptr, ptr %aux.ptr.var, align 32, !dbg !25
  %buf.ptr.2 = load ptr addrspace(7), ptr %buf.ptr.var, align 32, !dbg !26
  call void @llvm.dbg.value(metadata ptr addrspace(7) %buf.ptr.2, metadata !15, metadata !DIExpression()), !dbg !26
  %buf.ptr.3 = getelementptr float, ptr addrspace(7) %buf.ptr.2, i32 %idx, !dbg !27
  call void @llvm.dbg.value(metadata ptr addrspace(7) %buf.ptr.3, metadata !16, metadata !DIExpression()), !dbg !27
  store ptr addrspace(7) %buf.ptr.3, ptr %buf.ptr.var, align 32, !dbg !28
  %buf.ptr.4 = load ptr addrspace(7), ptr %buf.ptr.var, align 32, !dbg !29
  call void @llvm.dbg.value(metadata ptr addrspace(7) %buf.ptr.4, metadata !17, metadata !DIExpression()), !dbg !29
  %ret = load float, ptr addrspace(7) %buf.ptr.4, align 4, !dbg !30
  call void @llvm.dbg.value(metadata float %ret, metadata !18, metadata !DIExpression()), !dbg !30
  %aux.ptr.2 = load ptr addrspace(7), ptr %aux.ptr.var, align 32, !dbg !31
  call void @llvm.dbg.value(metadata ptr addrspace(7) %aux.ptr.2, metadata !19, metadata !DIExpression()), !dbg !31
  store ptr addrspace(7) %buf.ptr.4, ptr addrspace(7) %aux.ptr.2, align 32, !dbg !32
  ret float %ret, !dbg !33
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{i32 14}
!3 = !{i32 9}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "debug_stash_pointer", linkageName: "debug_stash_pointer", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !12, !14, !15, !16, !17, !18, !19}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !10)
!12 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 3, type: !13)
!13 = !DIBasicType(name: "ty256", size: 256, encoding: DW_ATE_unsigned)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 5, type: !13)
!15 = !DILocalVariable(name: "5", scope: !5, file: !1, line: 7, type: !13)
!16 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 8, type: !13)
!17 = !DILocalVariable(name: "7", scope: !5, file: !1, line: 10, type: !13)
!18 = !DILocalVariable(name: "8", scope: !5, file: !1, line: 11, type: !10)
!19 = !DILocalVariable(name: "9", scope: !5, file: !1, line: 12, type: !13)
!20 = !DILocation(line: 1, column: 1, scope: !5)
!21 = !DILocation(line: 2, column: 1, scope: !5)
!22 = !DILocation(line: 3, column: 1, scope: !5)
!23 = !DILocation(line: 4, column: 1, scope: !5)
!24 = !DILocation(line: 5, column: 1, scope: !5)
!25 = !DILocation(line: 6, column: 1, scope: !5)
!26 = !DILocation(line: 7, column: 1, scope: !5)
!27 = !DILocation(line: 8, column: 1, scope: !5)
!28 = !DILocation(line: 9, column: 1, scope: !5)
!29 = !DILocation(line: 10, column: 1, scope: !5)
!30 = !DILocation(line: 11, column: 1, scope: !5)
!31 = !DILocation(line: 12, column: 1, scope: !5)
!32 = !DILocation(line: 13, column: 1, scope: !5)
!33 = !DILocation(line: 14, column: 1, scope: !5)
!40 = distinct !DIAssignID()
