; ModuleID = 'disable_dynamic_devmem.ll'

; Adding this file first in the list of prelink files effectively 
; disables the host services request for device malloc or free.  This only 
; disables the "host-assisted" part dynamic memory management which
; is only needed when the initial device memory heap is exhausted.
; If -fenable-host-devmem ON, then do not add this disable file to list
; of link files. This allows the actual hostrpc or hostcall device stub
;  "__ockl_devmem_request" to make the host service request to allocate
; more device memory to grow the heap.
; Why disable this? Enabling host services requires additional host
; threads to wait for requests which could impact overall performance.


source_filename = "disable_devmem_request.ll"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: convergent nounwind
define external i64 @__ockl_devmem_request(i64 noundef %addr, i64 noundef %size) local_unnamed_addr #0 {
entry:
  ret i64 0
}

attributes #0 = { convergent nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" "uniform-work-group-size"="true" }
