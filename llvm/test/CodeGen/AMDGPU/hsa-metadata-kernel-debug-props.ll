; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX800 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

declare void @llvm.dbg.declare(metadata, metadata, metadata)

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]

; CHECK:  Kernels:
; CHECK:    - Name:       test
; CHECK:      SymbolName: 'test@kd'
; CHECK:      DebugProps:
; CHECK:        DebuggerABIVersion:                [ 1, 0 ]
; CHECK:        ReservedNumVGPRs:                  4
; GFX700:       ReservedFirstVGPR:                 8
; GFX800:       ReservedFirstVGPR:                 8
; GFX900:       ReservedFirstVGPR:                 11
; CHECK:        PrivateSegmentBufferSGPR:          0
; CHECK:        WavefrontPrivateSegmentOffsetSGPR: 11
define amdgpu_kernel void @test(i32 addrspace(1)* %A) #0 !dbg !7 !kernel_arg_addr_space !12 !kernel_arg_access_qual !13 !kernel_arg_type !14 !kernel_arg_base_type !14 !kernel_arg_type_qual !15 {
entry:
  %A.addr = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* %A, i32 addrspace(1)** %A.addr, align 4
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %A.addr, metadata !16, metadata !17), !dbg !18
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4, !dbg !19
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 0, !dbg !19
  store i32 777, i32 addrspace(1)* %arrayidx, align 4, !dbg !20
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4, !dbg !21
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 1, !dbg !21
  store i32 888, i32 addrspace(1)* %arrayidx1, align 4, !dbg !22
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %A.addr, align 4, !dbg !23
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %2, i64 2, !dbg !23
  store i32 999, i32 addrspace(1)* %arrayidx2, align 4, !dbg !24
  ret void, !dbg !25
}

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="gfx800" "target-features"="+16-bit-insts,+amdgpu-debugger-emit-prologue,+amdgpu-debugger-insert-nops,+amdgpu-debugger-reserve-regs,+dpp,+fp64-fp16-denormals,+s-memrealtime,-fp32-denormals" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!opencl.ocl.version = !{!3}
!llvm.module.flags = !{!4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 5.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "code-object-metadata-kernel-debug-props.cl", directory: "/some/random/directory")
!2 = !{}
!3 = !{i32 1, i32 0}
!4 = !{i32 2, !"Dwarf Version", i32 2}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 5.0.0"}
!7 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{i32 1}
!13 = !{!"none"}
!14 = !{!"int*"}
!15 = !{!""}
!16 = !DILocalVariable(name: "A", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!17 = !DIExpression(DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef)
!18 = !DILocation(line: 1, column: 30, scope: !7)
!19 = !DILocation(line: 2, column: 3, scope: !7)
!20 = !DILocation(line: 2, column: 8, scope: !7)
!21 = !DILocation(line: 3, column: 3, scope: !7)
!22 = !DILocation(line: 3, column: 8, scope: !7)
!23 = !DILocation(line: 4, column: 3, scope: !7)
!24 = !DILocation(line: 4, column: 8, scope: !7)
!25 = !DILocation(line: 5, column: 1, scope: !7)
