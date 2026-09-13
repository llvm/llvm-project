; RUN: llc -mtriple=amdgpu6.00 -verify-misched < %s | FileCheck --check-prefixes=NOFLAG %s
; RUN: llc -mtriple=amdgpu6.00 -amdgpu-use-amdgpu-trackers=1 -verify-misched < %s | FileCheck --check-prefixes=TRACKERSON %s
; RUN: llc -mtriple=amdgpu6.00 -amdgpu-use-amdgpu-trackers=0 -verify-misched < %s | FileCheck --check-prefixes=TRACKERSOFF %s

; The "amdgpu-use-amdgpu-trackers" function attribute lets a single kernel opt
; in to the AMDGPU specific RP trackers without flipping the
; -amdgpu-use-amdgpu-trackers flag (and therefore the scheduling behavior of
; every other function) for the whole module. An explicit
; -amdgpu-use-amdgpu-trackers=0/1 on the command line still takes precedence
; over the attribute, for every function in the module.

; This kernel does not have the attribute, so it only gets the AMDGPU RP
; tracker behavior when the flag is explicitly passed on the command line.
; CHECK-LABEL: {{^}}no_attribute:
; NOFLAG:      NumVgprs: 20
; NOFLAG:      Occupancy: 7
; TRACKERSON:  NumVgprs: 20
; TRACKERSON:  Occupancy: 10
; TRACKERSOFF: NumVgprs: 20
; TRACKERSOFF: Occupancy: 7
define amdgpu_kernel void @no_attribute(ptr addrspace(1) %out, ptr addrspace(4) %in) {
  %load = load <64 x i16>, ptr addrspace(4) %in
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

; This kernel has the attribute set to true, so it behaves as though
; -amdgpu-use-amdgpu-trackers=1 were passed even when no flag is given, but an
; explicit command-line flag (in either direction) still wins.
; CHECK-LABEL: {{^}}attribute_enables_trackers:
; NOFLAG:      NumVgprs: 20
; NOFLAG:      Occupancy: 10
; TRACKERSON:  NumVgprs: 20
; TRACKERSON:  Occupancy: 10
; TRACKERSOFF: NumVgprs: 20
; TRACKERSOFF: Occupancy: 7
define amdgpu_kernel void @attribute_enables_trackers(ptr addrspace(1) %out, ptr addrspace(4) %in) #0 {
  %load = load <64 x i16>, ptr addrspace(4) %in
  %ext = zext <64 x i16> %load to <64 x i32>
  store <64 x i32> %ext, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-use-amdgpu-trackers"="true" }
