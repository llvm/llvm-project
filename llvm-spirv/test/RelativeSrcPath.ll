; Source:
; __kernel void foo(__global int *a, __global int *b) {
;   a[0] += b[0];
; }

; Command:
; clang -cc1 -triple spir -O0 -debug-info-kind=line-tables-only -emit-llvm -o RelativeSrcPath.ll RelativeSrcPath.cl

; Directory: /tmp

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

; ModuleID = 'RelativeSrcPath.cl'
source_filename = "RelativeSrcPath.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent noinline nounwind optnone
define spir_kernel void @foo(i32 addrspace(1)* %a, i32 addrspace(1)* %b) #0 !dbg !8 !kernel_arg_addr_space !11 !kernel_arg_access_qual !12 !kernel_arg_type !13 !kernel_arg_base_type !13 !kernel_arg_type_qual !14 {
entry:
  %a.addr = alloca i32 addrspace(1)*, align 4
  %b.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %a, i32 addrspace(1)** %a.addr, align 4
  store i32 addrspace(1)* %b, i32 addrspace(1)** %b.addr, align 4
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %b.addr, align 4, !dbg !15
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i32 0, !dbg !15
  %1 = load i32, i32 addrspace(1)* %arrayidx, align 4, !dbg !15
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %a.addr, align 4, !dbg !15
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %2, i32 0, !dbg !15
  %3 = load i32, i32 addrspace(1)* %arrayidx1, align 4, !dbg !15
  %add = add nsw i32 %3, %1, !dbg !15
  store i32 %add, i32 addrspace(1)* %arrayidx1, align 4, !dbg !15
  ret void, !dbg !16
}

attributes #0 = { convergent noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.ocl.version = !{!5}
!opencl.spir.version = !{!6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (cfe/trunk)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, i32 0}
!6 = !{i32 1, i32 2}
!7 = !{!"clang version 8.0.0 (cfe/trunk)"}
!8 = distinct !DISubprogram(name: "foo", scope: !9, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
; CHECK: String [[ID:[0-9]+]] "/tmp/RelativeSrcPath.cl"
; CHECK: Line [[ID]]
!9 = !DIFile(filename: "RelativeSrcPath.cl", directory: "/tmp")
!10 = !DISubroutineType(types: !2)
!11 = !{i32 1, i32 1}
!12 = !{!"none", !"none"}
!13 = !{!"int*", !"int*"}
!14 = !{!"", !""}
!15 = !DILocation(line: 2, scope: !8)
!16 = !DILocation(line: 3, scope: !8)
