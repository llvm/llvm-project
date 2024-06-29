; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -pass-remarks-output=%t -pass-remarks-analysis=kernel-resource-usage -filetype=null %s 2>&1 | FileCheck -check-prefix=STDERR %s
; RUN: FileCheck -check-prefix=REMARK %s < %t

; STDERR: remark: foo.cl:27:0: Function Name: test_kernel
; STDERR-NEXT: remark: foo.cl:27:0:     SGPRs: 28
; STDERR-NEXT: remark: foo.cl:27:0:     VGPRs: 9
; STDERR-NEXT: remark: foo.cl:27:0:     AGPRs: 43
; STDERR-NEXT: remark: foo.cl:27:0:     ScratchSize [bytes/lane]: 0
; STDERR-NEXT: remark: foo.cl:27:0:     Dynamic Stack: False
; STDERR-NEXT: remark: foo.cl:27:0:     Occupancy [waves/SIMD]: 5
; STDERR-NEXT: remark: foo.cl:27:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:27:0:     VGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:27:0:     LDS Size [bytes/block]: 512

; REMARK-LABEL: --- !Analysis
; REMARK: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            FunctionName
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          'Function Name: '
; REMARK-NEXT:   - FunctionName:      test_kernel
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            NumSGPR
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    SGPRs: '
; REMARK-NEXT:   - NumSGPR:         '28'
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            NumVGPR
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    VGPRs: '
; REMARK-NEXT:   - NumVGPR:         '9'
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            NumAGPR
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    AGPRs: '
; REMARK-NEXT:   - NumAGPR:         '43'
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            ScratchSize
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    ScratchSize [bytes/lane]: '
; REMARK-NEXT:   - ScratchSize:     '0'
; REMARK-NEXT: ..
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            DynamicStack
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String: ' Dynamic Stack:
; REMARK-NEXT:   - DynamicStack: 'False'
; REMARK-NEXT: ..
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            Occupancy
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    Occupancy [waves/SIMD]: '
; REMARK-NEXT:   - Occupancy:       '5'
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            SGPRSpill
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    SGPRs Spill: '
; REMARK-NEXT:   - SGPRSpill:       '0'
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            VGPRSpill
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    VGPRs Spill: '
; REMARK-NEXT:   - VGPRSpill:       '0'
; REMARK-NEXT: ...
; REMARK-NEXT: --- !Analysis
; REMARK-NEXT: Pass:            kernel-resource-usage
; REMARK-NEXT: Name:            BytesLDS
; REMARK-NEXT: DebugLoc:        { File: foo.cl, Line: 27, Column: 0 }
; REMARK-NEXT: Function:        test_kernel
; REMARK-NEXT: Args:
; REMARK-NEXT:   - String:          '    LDS Size [bytes/block]: '
; REMARK-NEXT:   - BytesLDS:        '512'
; REMARK-NEXT: ...

@lds = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4

define amdgpu_kernel void @test_kernel() !dbg !3 {
  call void asm sideeffect "; clobber v8", "~{v8}"()
  call void asm sideeffect "; clobber s23", "~{s23}"()
  call void asm sideeffect "; clobber a42", "~{a42}"()
  call void asm sideeffect "; use $0", "v"(ptr addrspace(3) @lds)
  ret void
}

; STDERR: remark: foo.cl:42:0: Function Name: test_func
; STDERR-NEXT: remark: foo.cl:42:0:     SGPRs: 0
; STDERR-NEXT: remark: foo.cl:42:0:     VGPRs: 0
; STDERR-NEXT: remark: foo.cl:42:0:     AGPRs: 0
; STDERR-NEXT: remark: foo.cl:42:0:     ScratchSize [bytes/lane]: 0
; STDERR-NEXT: remark: foo.cl:42:0:     Dynamic Stack: False
; STDERR-NEXT: remark: foo.cl:42:0:     Occupancy [waves/SIMD]: 0
; STDERR-NEXT: remark: foo.cl:42:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:42:0:     VGPRs Spill: 0
; STDERR-NOT: LDS Size
define void @test_func() !dbg !6 {
  call void asm sideeffect "; clobber v17", "~{v17}"()
  call void asm sideeffect "; clobber s11", "~{s11}"()
  call void asm sideeffect "; clobber a9", "~{a9}"()
  ret void
}

; STDERR: remark: foo.cl:8:0: Function Name: empty_kernel
; STDERR-NEXT: remark: foo.cl:8:0:     SGPRs: 4
; STDERR-NEXT: remark: foo.cl:8:0:     VGPRs: 0
; STDERR-NEXT: remark: foo.cl:8:0:     AGPRs: 0
; STDERR-NEXT: remark: foo.cl:8:0:     ScratchSize [bytes/lane]: 0
; STDERR-NEXT: remark: foo.cl:8:0:     Dynamic Stack: False
; STDERR-NEXT: remark: foo.cl:8:0:     Occupancy [waves/SIMD]: 8
; STDERR-NEXT: remark: foo.cl:8:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:8:0:     VGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:8:0:     LDS Size [bytes/block]: 0
define amdgpu_kernel void @empty_kernel() !dbg !7 {
  ret void
}

; STDERR: remark: foo.cl:52:0: Function Name: empty_func
; STDERR-NEXT: remark: foo.cl:52:0:     SGPRs: 0
; STDERR-NEXT: remark: foo.cl:52:0:     VGPRs: 0
; STDERR-NEXT: remark: foo.cl:52:0:     AGPRs: 0
; STDERR-NEXT: remark: foo.cl:52:0:     ScratchSize [bytes/lane]: 0
; STDERR-NEXT: remark: foo.cl:52:0:     Dynamic Stack: False
; STDERR-NEXT: remark: foo.cl:52:0:     Occupancy [waves/SIMD]: 0
; STDERR-NEXT: remark: foo.cl:52:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:52:0:     VGPRs Spill: 0
define void @empty_func() !dbg !8 {
  ret void
}

; STDERR: remark: foo.cl:64:0: Function Name: test_indirect_call
; STDERR-NEXT: remark: foo.cl:64:0:     SGPRs: 39
; STDERR-NEXT: remark: foo.cl:64:0:     VGPRs: 32
; STDERR-NEXT: remark: foo.cl:64:0:     AGPRs: 10
; STDERR-NEXT: remark: foo.cl:64:0:     ScratchSize [bytes/lane]: 0
; STDERR-NEXT: remark: foo.cl:64:0:     Dynamic Stack: True
; STDERR-NEXT: remark: foo.cl:64:0:     Occupancy [waves/SIMD]: 8
; STDERR-NEXT: remark: foo.cl:64:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:64:0:     VGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:64:0:     LDS Size [bytes/block]: 0
@gv.fptr0 = external hidden unnamed_addr addrspace(4) constant ptr, align 4

define amdgpu_kernel void @test_indirect_call() !dbg !9 {
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

; STDERR: remark: foo.cl:74:0: Function Name: test_indirect_w_static_stack
; STDERR-NEXT: remark: foo.cl:74:0:     SGPRs: 39
; STDERR-NEXT: remark: foo.cl:74:0:     VGPRs: 32
; STDERR-NEXT: remark: foo.cl:74:0:     AGPRs: 10
; STDERR-NEXT: remark: foo.cl:74:0:     ScratchSize [bytes/lane]: 144
; STDERR-NEXT: remark: foo.cl:74:0:     Dynamic Stack: True
; STDERR-NEXT: remark: foo.cl:74:0:     Occupancy [waves/SIMD]: 8
; STDERR-NEXT: remark: foo.cl:74:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:74:0:     VGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:74:0:     LDS Size [bytes/block]: 0

declare void @llvm.memset.p5.i64(ptr addrspace(5) nocapture readonly, i8, i64, i1 immarg)

define amdgpu_kernel void @test_indirect_w_static_stack() !dbg !10 {
  %alloca = alloca <10 x i64>, align 16, addrspace(5)
  call void @llvm.memset.p5.i64(ptr addrspace(5) %alloca, i8 0, i64 40, i1 false)
  %fptr = load ptr, ptr addrspace(4) @gv.fptr0
  call void %fptr()
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.module.flags = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.cl", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, type: !4, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "test_func", scope: !1, file: !1, type: !4, scopeLine: 42, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "empty_kernel", scope: !1, file: !1, type: !4, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "empty_func", scope: !1, file: !1, type: !4, scopeLine: 52, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = distinct !DISubprogram(name: "test_indirect_call", scope: !1, file: !1, type: !4, scopeLine: 64, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = distinct !DISubprogram(name: "test_indirect_w_static_stack", scope: !1, file: !1, type: !4, scopeLine: 74, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !{i32 1, !"amdhsa_code_object_version", i32 500}
