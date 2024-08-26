; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=SM60
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 | FileCheck %s  --check-prefix=SM70
; RUN: %if ptxas-12.2 %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 | %ptxas-verify -arch=sm_70 %}

; CHECK-LABEL: fence_sc_sys
define void @fence_sc_sys() local_unnamed_addr {
  ; SM60: membar.sys
  ; SM70: fence.sc.sys
  fence seq_cst
  ret void
}

; CHECK-LABEL: fence_acq_rel_sys
define void @fence_acq_rel_sys() local_unnamed_addr {
  ; SM60: membar.sys
  ; SM70: fence.acq_rel.sys
  fence acq_rel
  ret void  
}

; CHECK-LABEL: fence_release_sys
define void @fence_release_sys() local_unnamed_addr {
  ; SM60: membar.sys
  ; SM70: fence.acq_rel.sys  
  fence release
  ret void  
}

; CHECK-LABEL: fence_acquire_sys
define void @fence_acquire_sys() local_unnamed_addr {
  ; SM60: membar.sys
  ; SM70: fence.acq_rel.sys  
  fence acquire
  ret void  
}