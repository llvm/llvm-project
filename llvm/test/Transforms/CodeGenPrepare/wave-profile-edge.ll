; NOTE: Do not autogenerate. Check wave event identities and measured/unknown\n; state across composed CFG rewrites without unrelated instruction checks.\n; REQUIRES: amdgpu-registered-target
; RUN: split-file %s %t
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='require<profile-summary>,function(codegenprepare),verify' %t/edge.ll | FileCheck %s --check-prefix=CLEAN
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='require<profile-summary>,function(codegenprepare,codegenprepare),verify' %t/edge.ll | FileCheck %s --check-prefix=CLEAN
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='require<profile-summary>,function(codegenprepare),verify' %t/invalid.ll | FileCheck %s --check-prefix=INVALID

; An empty block with one predecessor only subdivides an edge. Retain the
; surrounding measurements when removing that block, including repeated
; cleanup. Previously invalid measurements must stay invalid.
;
; CLEAN-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; CLEAN-SAME: !wave.profile [[TABLE:![0-9]+]]
; CLEAN: br i1 %run, label %body, label %exit, !wave.profile.block [[ENTRY:![0-9]+]]
; CLEAN: br i1 %more, label %body, label %exit, !wave.profile.block [[BODY:![0-9]+]]
; CLEAN: ret void, !wave.profile.block [[EXIT:![0-9]+]]
; CLEAN: [[TABLE]] = distinct !{i64 2, i64 2480672276464841217, i64 6, i64 66, i64 6, i64 6{{(, i64 0)*}}}
; CLEAN: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1, i64 3}
; CLEAN: [[BODY]] = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 3}
; CLEAN: [[EXIT]] = !{i64 2, i64 2480672276464841217, i64 3, i64 1}
;
;--- edge.ll
define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out, i32 %n, i1 %run) !wave.profile !0 {
entry:
  br i1 %run, label %body, label %exit, !wave.profile.block !1
body:
  %i = phi i32 [0, %entry], [%next, %body]
  store volatile i32 %i, ptr addrspace(1) %out
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %body, label %cleanup, !wave.profile.block !2
cleanup:
  br label %exit, !wave.profile.block !3
exit:
  store volatile i32 42, ptr addrspace(1) %out
  ret void, !wave.profile.block !4
}
!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 66, i64 6, i64 6}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1, i64 3}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 1, i64 3}
!4 = !{i64 2, i64 2480672276464841217, i64 3, i64 1}

;--- invalid.ll
; The body's old successor list already mismatches. Removing cleanup makes
; those old IDs match again, but must not revive the rejected measurements.
; INVALID-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; INVALID: br i1 %run, label %body, label %exit, !wave.profile.block [[ENTRY:![0-9]+]]
; INVALID: br i1 %more, label %body, label %exit, !wave.profile.block [[BODY:![0-9]+]]
; INVALID: ret void, !wave.profile.block [[EXIT:![0-9]+]]
; INVALID: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1, i64 3}
; INVALID: [[BODY]] = !{i64 2, i64 2480672276464841217, i64 1, i64 0, i64 1, i64 3}
; INVALID: [[EXIT]] = !{i64 2, i64 2480672276464841217, i64 3, i64 0}
define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out, i32 %n, i1 %run) !wave.profile !0 {
entry:
  br i1 %run, label %body, label %exit, !wave.profile.block !1
body:
  %i = phi i32 [0, %entry], [%next, %body]
  store volatile i32 %i, ptr addrspace(1) %out
  %next = add nuw nsw i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %body, label %cleanup, !wave.profile.block !2
cleanup:
  br label %exit, !wave.profile.block !3
exit:
  store volatile i32 42, ptr addrspace(1) %out
  ret void, !wave.profile.block !4
}
!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 66, i64 6, i64 6}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1, i64 3}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 1, i64 3}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 1, i64 3}
!4 = !{i64 2, i64 2480672276464841217, i64 3, i64 1}
