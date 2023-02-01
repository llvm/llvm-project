; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -pass-remarks-output=%t -pass-remarks-analysis=kernel-resource-usage -filetype=obj -o /dev/null %s 2>&1 | FileCheck -check-prefix=STDERR %s
; RUN: FileCheck -check-prefix=REMARK %s < %t

; STDERR: remark: foo.cl:27:0: Function Name: test_kernel
; STDERR-NEXT: remark: foo.cl:27:0:     SGPRs: 24
; STDERR-NEXT: remark: foo.cl:27:0:     VGPRs: 9
; STDERR-NEXT: remark: foo.cl:27:0:     AGPRs: 43
; STDERR-NEXT: remark: foo.cl:27:0:     ScratchSize [bytes/lane]: 0
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
; REMARK-NEXT:   - NumSGPR:         '24'
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
; REMARK-NEXT: ...
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
; STDERR-NEXT: remark: foo.cl:8:0:     SGPRs: 0
; STDERR-NEXT: remark: foo.cl:8:0:     VGPRs: 0
; STDERR-NEXT: remark: foo.cl:8:0:     AGPRs: 0
; STDERR-NEXT: remark: foo.cl:8:0:     ScratchSize [bytes/lane]: 0
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
; STDERR-NEXT: remark: foo.cl:52:0:     Occupancy [waves/SIMD]: 0
; STDERR-NEXT: remark: foo.cl:52:0:     SGPRs Spill: 0
; STDERR-NEXT: remark: foo.cl:52:0:     VGPRs Spill: 0
define void @empty_func() !dbg !8 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "foo.cl", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, type: !4, scopeLine: 27, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "test_func", scope: !1, file: !1, type: !4, scopeLine: 42, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "empty_kernel", scope: !1, file: !1, type: !4, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "empty_func", scope: !1, file: !1, type: !4, scopeLine: 52, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
