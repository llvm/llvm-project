; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC

target triple = "dxil-unknown-shadermodel6.0-compute"

define void @main() #0 {
entry:
  ret void
}
attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }


!dx.rootsignatures = !{!2} ; list of function/root signature pairs
!2 = !{ ptr @main, !3, i32 2 } ; function, root signature
!3 = !{ !5 } ; list of root signature elements
!5 = !{ !"DescriptorTable", i32 0, !6, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20 }

; typedef enum D3D12_DESCRIPTOR_RANGE_FLAGS {
;   NONE = 0,
;   DESCRIPTORS_VOLATILE = 0x1,
;   DATA_VOLATILE = 0x2,
;   DATA_STATIC_WHILE_SET_AT_EXECUTE = 0x4,
;   DATA_STATIC = 0x8,
;   DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS = 0x10000
; } ;

;0
!6 = !{ !"Sampler", i32 1, i32 0, i32 1, i32 -1, i32 0 }
;DESCRIPTORS_VOLATILE
!8 = !{ !"Sampler", i32 1, i32 0, i32 3, i32 -1, i32 1 }
;DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS
!9 = !{ !"Sampler", i32 1, i32 0, i32 4, i32 -1, i32 65536 }
;0
!10 = !{ !"SRV", i32 1, i32 0, i32 5, i32 -1, i32 1 }
;DESCRIPTORS_VOLATILE
!11 = !{ !"UAV", i32 5, i32 1, i32 6, i32 5, i32 1 }
;DATA_VOLATILE
!12 = !{ !"CBV", i32 5, i32 1, i32 7, i32 5, i32 2 }
;DATA_STATIC
!13 = !{ !"SRV", i32 5, i32 1, i32 8, i32 5, i32 8 }
;DATA_STATIC_WHILE_SET_AT_EXECUTE
!14 = !{ !"UAV", i32 5, i32 1, i32 9, i32 5, i32 4 }
;DESCRIPTORS_VOLATILE | DATA_VOLATILE
!15 = !{ !"CBV", i32 5, i32 1, i32 10, i32 5, i32 3 }
;DESCRIPTORS_VOLATILE | DATA_STATIC_WHILE_SET_AT_EXECUTE
!16 = !{ !"SRV", i32 5, i32 1, i32 11, i32 5, i32 5 }
;DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS
!17 = !{ !"UAV", i32 5, i32 1, i32 12, i32 5, i32 65536 }
;DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS | DATA_VOLATILE
!18 = !{ !"CBV", i32 5, i32 1, i32 13, i32 5, i32 65538 }
;DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS | DATA_STATIC
!19 = !{ !"SRV", i32 5, i32 1, i32 14, i32 5, i32 65544 }
;DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS | DATA_STATIC_WHILE_SET_AT_EXECUTE
!20 = !{ !"UAV", i32 5, i32 1, i32 15, i32 5, i32 65540 }

;DXC:- Name:            RTS0
;DXC-NEXT:    Size:            380
;DXC-NEXT:    RootSignature:
;DXC-NEXT:      Version:         2
;DXC-NEXT:      NumRootParameters: 1
;DXC-NEXT:      RootParametersOffset: 24
;DXC-NEXT:      NumStaticSamplers: 0
;DXC-NEXT:      StaticSamplersOffset: 380
;DXC-NEXT:      Parameters:
;DXC-NEXT:        - ParameterType:   0
;DXC-NEXT:          ShaderVisibility: 0
;DXC-NEXT:          Table:
;DXC-NEXT:            NumRanges:       14
;DXC-NEXT:            RangesOffset:    44
;DXC-NEXT:            Ranges:
;DXC-NEXT:              - RangeType:       3
;DXC-NEXT:                NumDescriptors:  1
;DXC-NEXT:                BaseShaderRegister: 0
;DXC-NEXT:                RegisterSpace:   1
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 4294967295
;DXC-NEXT:              - RangeType:       3
;DXC-NEXT:                NumDescriptors:  1
;DXC-NEXT:                BaseShaderRegister: 0
;DXC-NEXT:                RegisterSpace:   3
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 4294967295
;DXC-NEXT:                DESCRIPTORS_VOLATILE: true
;DXC-NEXT:              - RangeType:       3
;DXC-NEXT:                NumDescriptors:  1
;DXC-NEXT:                BaseShaderRegister: 0
;DXC-NEXT:                RegisterSpace:   4
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 4294967295
;DXC-NEXT:                DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS: true
;DXC-NEXT:              - RangeType:       0
;DXC-NEXT:                NumDescriptors:  1
;DXC-NEXT:                BaseShaderRegister: 0
;DXC-NEXT:                RegisterSpace:   5
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 4294967295
;DXC-NEXT:                DESCRIPTORS_VOLATILE: true
;DXC-NEXT:              - RangeType:       1
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   6
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DESCRIPTORS_VOLATILE: true
;DXC-NEXT:              - RangeType:       2
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   7
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DATA_VOLATILE:   true
;DXC-NEXT:              - RangeType:       0
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   8
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DATA_STATIC:     true
;DXC-NEXT:              - RangeType:       1
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   9
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DATA_STATIC_WHILE_SET_AT_EXECUTE: true
;DXC-NEXT:              - RangeType:       2
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   10
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DESCRIPTORS_VOLATILE: true
;DXC-NEXT:                DATA_VOLATILE:   true
;DXC-NEXT:              - RangeType:       0
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   11
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DESCRIPTORS_VOLATILE: true
;DXC-NEXT:                DATA_STATIC_WHILE_SET_AT_EXECUTE: true
;DXC-NEXT:              - RangeType:       1
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   12
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS: true
;DXC-NEXT:              - RangeType:       2
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   13
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DATA_VOLATILE:   true
;DXC-NEXT:                DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS: true
;DXC-NEXT:              - RangeType:       0
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   14
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DATA_STATIC:     true
;DXC-NEXT:                DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS: true
;DXC-NEXT:              - RangeType:       1
;DXC-NEXT:                NumDescriptors:  5
;DXC-NEXT:                BaseShaderRegister: 1
;DXC-NEXT:                RegisterSpace:   15
;DXC-NEXT:                OffsetInDescriptorsFromTableStart: 5
;DXC-NEXT:                DATA_STATIC_WHILE_SET_AT_EXECUTE: true
;DXC-NEXT:                DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS: true
