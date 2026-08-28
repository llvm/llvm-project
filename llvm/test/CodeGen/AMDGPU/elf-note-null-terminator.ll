; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -filetype=obj < %s | llvm-readelf -n - | FileCheck --check-prefix=NOTE-AMDGPU %s
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -filetype=obj < %s | llvm-readelf -x .note - | FileCheck --check-prefix=HEX-AMDGPU %s
; RUN: llc -mtriple=amdgpu9.0a-amd-unknown -filetype=obj < %s | llvm-readelf -n - | FileCheck --check-prefix=NOTE-AMD %s
; RUN: llc -mtriple=amdgpu9.0a-amd-unknown -filetype=obj < %s | llvm-readelf -x .note - | FileCheck --check-prefix=HEX-AMD %s

; Verify ELF notes have explicit null terminators after the name field.
; Tests both "AMDGPU" (NoteNameV3, 7 bytes with null) and "AMD" (NoteNameV2, 4 bytes with null).

; Check "AMDGPU" note is parseable (used in amdhsa triple)
; NOTE-AMDGPU-NOT: Unknown note type
; NOTE-AMDGPU: AMDGPU
; NOTE-AMDGPU: NT_AMDGPU_METADATA

; Check hex shows explicit null: "AMDGPU" (6 bytes) + null = namesz of 7
; HEX-AMDGPU: Hex dump of section '.note':
; HEX-AMDGPU-NEXT: 0x00000000 07000000 {{[0-9a-f]+}} {{[0-9a-f]+}} 414d4447
; HEX-AMDGPU-NEXT: 0x00000010 5055{{00}}

; Check "AMD" note is parseable (used in unknown triple)
; NOTE-AMD-NOT: Unknown note type
; NOTE-AMD: AMD
; NOTE-AMD: NT_AMD_HSA_ISA_NAME

; Check hex shows explicit null: "AMD" (3 bytes) + null = namesz of 4
; HEX-AMD: Hex dump of section '.note':
; HEX-AMD: 04000000 {{[0-9a-f]+}} {{[0-9a-f]+}} 414d44{{00}}

define amdgpu_kernel void @test_note() {
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
