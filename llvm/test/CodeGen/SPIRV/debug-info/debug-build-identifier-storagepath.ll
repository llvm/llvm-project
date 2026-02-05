; RUN: llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spv-emit-nonsemantic-debug-info --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-DAG: [[i32type:%[0-9]+]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: [[i8type:%[0-9]+]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: [[void_type:%[0-9]+]] = OpTypeVoid
; CHECK-SPIRV-DAG: [[ptr_type:%[0-9]+]] = OpTypePointer CrossWorkgroup [[i8type]]
; CHECK-SPIRV-DAG: [[func_type:%[0-9]+]] = OpTypeFunction [[void_type]] [[ptr_type]] [[ptr_type]] [[ptr_type]]
; CHECK-SPIRV-DAG: [[i32_1:%[0-9]+]] = OpConstant [[i32type]] 1
; CHECK-SPIRV-DAG: [[i32_3:%[0-9]+]] = OpConstant [[i32type]] 3
; CHECK-SPIRV-DAG: [[string_build_id:%[0-9]+]] = OpString "1234567890"
; CHECK-SPIRV-DAG: [[string_storage_path:%[0-9]+]] = OpString "debug_storage_path.spv"
; CHECK-SPIRV-DAG: [[debug_source:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugSource
; CHECK-SPIRV-DAG: [[debug_comp_unit:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugCompilationUnit [[i32_3]] [[i32_1]] [[debug_source]]
; CHECK-SPIRV-DAG: [[debug_build_id:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugBuildIdentifier [[string_build_id]]
; CHECK-SPIRV-DAG: [[debug_storage_path:%[0-9]+]] = OpExtInst [[void_type]] {{%[0-9]+}} DebugStoragePath [[string_storage_path]]


define dso_local spir_kernel void @add_kernel(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %res) !dbg !8 {
entry:
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit( language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugFilename: "debug_storage_path.spv", dwoId: 1234567890, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "debug_build_storage.cl", directory: "/path/to/source")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!8 = distinct !DISubprogram(name: "add_kernel", scope: !1, file: !1, line: 1, type: !9, unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !2)
!9 = !DISubroutineType(types: !2)
!10 = !DILocation(line: 2, column: 1, scope: !8)
