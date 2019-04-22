; Source:
;__kernel void foo(void) {
;  __local int a;
;}
; clang -cc1 -triple spir -disable-llvm-passes -triple spir /work/tmp/tmp.cl -O0 -debug-info-kind=standalone -emit-llvm -o /work/llvm/projects/llvm-spirv/test/DebugInfo/LocalAddressSpace.ll

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv %t.bc -o - -spirv-text -spirv-mem2reg=false | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll
; RUN: cat %t.ll | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llc -mtriple=%triple -filetype=obj -O0 < %t.ll | llvm-dwarfdump -v -debug-info - | FileCheck %s

; CHECK-SPIRV: Variable {{[0-9]+}} [[foo_a:[0-9]+]]
; CHECK-SPIRV: DebugGlobalVariable {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} [[foo_a]]

; CHECK-LLVM: @foo.a = internal addrspace(3) global i32 undef, align 4, !dbg ![[a_dbg_expr:[0-9]+]]
; CHECK-LLVM: ![[a_dbg_expr]] = !DIGlobalVariableExpression(var: ![[a_dbg_var:[0-9]+]],
; CHECK-LLVM: ![[a_dbg_var]] = distinct !DIGlobalVariable(name: "a"

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}} = "a")
; CHECK-NEXT: DW_AT_type {{.*}} "int")
; CHECK-NEXT: DW_AT_decl_file {{.*}} ("/work/tmp/tmp.cl")
; CHECK-NEXT: DW_AT_decl_line {{.*}} (2)
; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]      (DW_OP_addr 0x0)

; ModuleID = '/work/tmp/tmp.cl'
source_filename = "/work/tmp/tmp.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

@foo.a = internal addrspace(3) global i32 undef, align 4, !dbg !0

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @foo() #0 !dbg !2 !kernel_arg_addr_space !8 !kernel_arg_access_qual !8 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !8 {
entry:
  ret void, !dbg !16
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!11, !12}
!opencl.ocl.version = !{!13}
!opencl.spir.version = !{!14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 2, type: !10, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !8)
!3 = !DIFile(filename: "tmp/tmp.cl", directory: "/work")
!4 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !5)
!5 = !{null}
!6 = distinct !DICompileUnit(language: DW_LANG_C99, file: !7, producer: "clang version 9.0.0 (https://llvm.org/git/clang 92470c6aadff9e614bfac44f48e6e1d430e5a32d) (https://llvm.org/git/llvm 461a7ee6493f997d6dc03ca0e80b6a7bd7943a83)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, globals: !9, nameTableKind: None)
!7 = !DIFile(filename: "/work/tmp/<stdin>", directory: "/work/llvm/build")
!8 = !{}
!9 = !{!0}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 1, i32 0}
!14 = !{i32 1, i32 2}
!15 = !{!"clang version 9.0.0 (https://llvm.org/git/clang 92470c6aadff9e614bfac44f48e6e1d430e5a32d) (https://llvm.org/git/llvm 461a7ee6493f997d6dc03ca0e80b6a7bd7943a83)"}
!16 = !DILocation(line: 3, scope: !2)
