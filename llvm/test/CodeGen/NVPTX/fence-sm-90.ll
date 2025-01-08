; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | FileCheck %s
; RUN: %if ptxas-12.2 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx78 | %ptxas-verify -arch=sm_90 %}

; CHECK-LABEL: fence_sc_cluster
define void @fence_sc_cluster() local_unnamed_addr {
  ; CHECK: fence.sc.cluster
  fence syncscope("cluster") seq_cst
  ret void
}

; CHECK-LABEL: fence_acq_rel_cluster
define void @fence_acq_rel_cluster() local_unnamed_addr {
  ; CHECK: fence.acq_rel.cluster
  fence syncscope("cluster") acq_rel
  ret void
}

; CHECK-LABEL: fence_release_cluster
define void @fence_release_cluster() local_unnamed_addr {
  ; CHECK: fence.acq_rel.cluster
  fence syncscope("cluster") release
  ret void
}

; CHECK-LABEL: fence_acquire_cluster
define void @fence_acquire_cluster() local_unnamed_addr {
  ; CHECK: fence.acq_rel.cluster
  fence syncscope("cluster") acquire
  ret void
}
