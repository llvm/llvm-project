; RUN: opt -S -passes='structurizecfg,verify' %s | FileCheck %s

target triple = "amdgcn-amd-amdhsa"

; A known CFG transform preserves a partial profile's original identities on
; surviving blocks and refreshes their successor identities. The synthesized
; flow block gets an identity but no measurement. The absent original entry
; retains its normalization count (11), distinct from the current entry (10).

define amdgpu_kernel void @if_else(i1 %condition, ptr addrspace(1) %out) !wave.profile !0 {
; CHECK-LABEL: define amdgpu_kernel void @if_else(
; CHECK-SAME: !wave.profile [[PROFILE:![0-9]+]] {
; CHECK: entry:
; CHECK: br i1 {{.*}}, label %{{.*}}, label %[[FLOW:.*]], {{.*}}!wave.profile.block [[ENTRY_MD:![0-9]+]]
; CHECK: [[FLOW]]:
; CHECK: br i1 {{.*}}, label %left, label %exit, !wave.profile.block [[FLOW_MD:![0-9]+]]
; CHECK: left:
; CHECK: br label %{{.*}}, !wave.profile.block [[LEFT_MD:![0-9]+]]
; CHECK: right:
; CHECK: br label %{{.*}}, !wave.profile.block [[RIGHT_MD:![0-9]+]]
; CHECK: exit:
; CHECK: ret void, !wave.profile.block [[EXIT_MD:![0-9]+]]
;
entry:
  br i1 %condition, label %left, label %right, !wave.profile.block !1

left:
  store volatile i32 1, ptr addrspace(1) %out
  br label %exit, !wave.profile.block !2

right:
  store volatile i32 2, ptr addrspace(1) %out
  br label %exit, !wave.profile.block !3

exit:
  ret void, !wave.profile.block !4
}

!0 = !{i64 2, i64 2685589004101179296, i64 11, i64 10, i64 7, i64 3, i64 10}
!1 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 2, i64 3}
!2 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 4}
!3 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 4}
!4 = !{i64 2, i64 2685589004101179296, i64 4, i64 1}

; CHECK: [[PROFILE]] = distinct !{i64 2, i64 2685589004101179296, i64 11, i64 10, i64 7, i64 3, i64 10, i64 0}
; CHECK: [[ENTRY_MD]] = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 3, i64 5}
; CHECK: [[FLOW_MD]] = !{i64 2, i64 2685589004101179296, i64 5, i64 0, i64 2, i64 4}
; CHECK: [[LEFT_MD]] = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 4}
; CHECK: [[RIGHT_MD]] = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 5}
; CHECK: [[EXIT_MD]] = !{i64 2, i64 2685589004101179296, i64 4, i64 1}
