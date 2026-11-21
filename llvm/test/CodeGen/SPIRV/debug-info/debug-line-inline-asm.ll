; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_INTEL_inline_assembly %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa --spirv-ext=+SPV_KHR_non_semantic_info,+SPV_INTEL_inline_assembly %s -o - -filetype=obj | spirv-val %}

; The inline asm in a HIP kernel must keep its source location.
;
; clang++ -x hip --offload-arch=amdgcnspirv -g -O0 --offload-device-only lowers
;
;   __global__ void kern(void) {
;     asm volatile ("s_nop 1");
;   }
;

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}kern.hip"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[V1:%[0-9]+]] = OpConstant [[I32]] 1{{$}}
; CHECK-DAG: [[V2:%[0-9]+]] = OpConstant [[I32]] 2{{$}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}

; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V3]] [[V4]]
; CHECK-NEXT: {{%[0-9]+}} = OpAsmCallINTEL
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V1]] [[V2]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-amd-amdhsa"

define spir_kernel void @_Z4kernv() addrspace(4) #0 !dbg !5 {
  call addrspace(0) void asm sideeffect "s_nop 1", ""() #1, !dbg !8, !srcloc !9
  ret void, !dbg !10
}

attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size" }
attributes #1 = { convergent nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_HIP, file: !1, producer: "AMD clang version 24.0.0git", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/src/kern.hip", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"debugger-tuning", !"gdb"}

!6 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !7)
!7 = !{null}
!11 = !DIFile(filename: "/src/kern.hip", directory: "")
!5 = distinct !DISubprogram(name: "kern", linkageName: "_Z4kernv", scope: !11, file: !11, line: 2, type: !6, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 3, scope: !5)
!9 = !{i64 76}
!10 = !DILocation(line: 4, column: 1, scope: !5)
