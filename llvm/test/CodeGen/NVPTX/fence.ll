; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s --check-prefix=SM60
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 | FileCheck %s  --check-prefix=SM70
; RUN: %if ptxas-12.2 %{ llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 | %ptxas-verify -arch=sm_70 %}

; TODO: implement and test thread scope.

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

; CHECK-LABEL: fence_sc_gpu
define void @fence_sc_gpu() local_unnamed_addr {
  ; SM60: membar.gl
  ; SM70: fence.sc.gpu
  fence syncscope("device") seq_cst
  ret void
}

; CHECK-LABEL: fence_acq_rel_gpu
define void @fence_acq_rel_gpu() local_unnamed_addr {
  ; SM60: membar.gl
  ; SM70: fence.acq_rel.gpu
  fence syncscope("device") acq_rel
  ret void
}

; CHECK-LABEL: fence_release_gpu
define void @fence_release_gpu() local_unnamed_addr {
  ; SM60: membar.gl
  ; SM70: fence.acq_rel.gpu
  fence syncscope("device") release
  ret void
}

; CHECK-LABEL: fence_acquire_gpu
define void @fence_acquire_gpu() local_unnamed_addr {
  ; SM60: membar.gl
  ; SM70: fence.acq_rel.gpu
  fence syncscope("device") acquire
  ret void
}

; CHECK-LABEL: fence_sc_cta
define void @fence_sc_cta() local_unnamed_addr {
  ; SM60: membar.cta
  ; SM70: fence.sc.cta
  fence syncscope("block") seq_cst
  ret void
}

; CHECK-LABEL: fence_acq_rel_cta
define void @fence_acq_rel_cta() local_unnamed_addr {
  ; SM60: membar.cta
  ; SM70: fence.acq_rel.cta
  fence syncscope("block") acq_rel
  ret void
}

; CHECK-LABEL: fence_release_cta
define void @fence_release_cta() local_unnamed_addr {
  ; SM60: membar.cta
  ; SM70: fence.acq_rel.cta
  fence syncscope("block") release
  ret void
}

; CHECK-LABEL: fence_acquire_cta
define void @fence_acquire_cta() local_unnamed_addr {
  ; SM60: membar.cta
  ; SM70: fence.acq_rel.cta
  fence syncscope("block") acquire
  ret void
}