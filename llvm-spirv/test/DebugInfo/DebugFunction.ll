; Check for 2 thigs:
; - After round trip translation function definition has !dbg metadata attached
;   specifically if -gline-tables-only was used for Clang
; - Parent operand of DebugFunction is DebugCompileUnit, not an OpString, even
;   if in LLVM IR it points to a DIFile instead of DICompileUnit.

; Source:
; float foo(int i) {
;     return i * 3.14;
; }
; void kernel k() {
;     float a = foo(2);
; }
; Command:
; clang -x cl -cl-std=c++ -emit-llvm -target spir -gline-tables-only -O0

; RUN: llvm-as %s -o - | llvm-spirv -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-SPIRV: String [[foo:[0-9]+]] "foo"
; CHECK-SPIRV: String [[k:[0-9]+]] "k"
; CHECK-SPIRV: [[CU:[0-9]+]] {{[0-9]+}} DebugCompileUnit
; CHECK-SPIRV: DebugFunction [[foo]] {{.*}} [[CU]] {{.*}} [[foo_id:[0-9]+]] {{[0-9]+}} {{$}}
; CHECK-SPIRV: DebugFunction [[k]] {{.*}} [[CU]] {{.*}} [[k_id:[0-9]+]] {{[0-9]+}} {{$}}

; CHECK-SPIRV: Function {{[0-9]+}} [[foo_id]]
; CHECK-LLVM: define spir_func float @_Z3fooi(i32) #{{[0-9]+}} !dbg !{{[0-9]+}} {
define dso_local spir_func float @_Z3fooi(i32) #0 !dbg !9 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4, !dbg !12
  %4 = sitofp i32 %3 to double, !dbg !12
  %5 = fmul double %4, 3.140000e+00, !dbg !13
  %6 = fptrunc double %5 to float, !dbg !12
  ret float %6, !dbg !14
}

; CHECK-SPIRV: Function {{[0-9]+}} [[k_id]]
; CHECK-LLVM: define spir_kernel void @_Z1kv() #{{[0-9]+}} !dbg !{{[0-9]+}}
define dso_local spir_kernel void @_Z1kv() #1 !dbg !15 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 {
  %1 = alloca float, align 4
  %2 = call spir_func float @_Z3fooi(i32 2) #2, !dbg !16
  store float %2, float* %1, align 4, !dbg !17
  ret void, !dbg !18
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!opencl.ocl.version = !{!6}
!opencl.spir.version = !{!7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (trunk 354644)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "/tmp/compiler-explorer-compiler119127-62-o5mw53.ko2sg/example.cpp", directory: "/tmp/compiler-explorer-compiler119127-62-o5mw53.ko2sg")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 0, i32 0}
!7 = !{i32 0, i32 2}
!8 = !{!"clang version 9.0.0 (trunk 354644)"}
!9 = distinct !DISubprogram(name: "foo", scope: !10, file: !10, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!10 = !DIFile(filename: "example.cpp", directory: "/tmp/compiler-explorer-compiler119127-62-o5mw53.ko2sg")
!11 = !DISubroutineType(types: !2)
!12 = !DILocation(line: 3, column: 12, scope: !9)
!13 = !DILocation(line: 3, column: 14, scope: !9)
!14 = !DILocation(line: 3, column: 5, scope: !9)
!15 = distinct !DISubprogram(name: "k", scope: !10, file: !10, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!16 = !DILocation(line: 7, column: 15, scope: !15)
!17 = !DILocation(line: 7, column: 11, scope: !15)
!18 = !DILocation(line: 8, column: 1, scope: !15)
