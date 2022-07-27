; RUN: llc -march=amdgcn -mcpu=gfx900 -O3 < %s | FileCheck -check-prefix=GCN %s
; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

%vec_type = type { %vec_base }
%vec_base = type { %union.anon }
%union.anon = type { %"vec_base<char, 3>::n_vec_" }
%"vec_base<char, 3>::n_vec_" = type { [3 x i8] }

$_f1 = comdat any
$_f2 = comdat any
@_f1 = linkonce_odr hidden local_unnamed_addr addrspace(3) global %vec_type undef, comdat, align 1
@_f2 = linkonce_odr hidden local_unnamed_addr addrspace(3) global %vec_type undef, comdat, align 1

; GCN-LABEL: @test
; GCN: v_mov_b32_e32 [[REG:v[0-9]+]], 1
; GCN-NEXT: global_store_byte v{{[0-9]+}}, [[REG]]

; CHECK-LABEL: @test
; CHECK: store i8 3, i8 addrspace(3)* %0, align 4, !alias.scope !0, !noalias !3
; CHECK: tail call void @llvm.memcpy.p3i8.p3i8.i64(i8 addrspace(3)* noundef align 1 dereferenceable(3) %2, i8 addrspace(3)* noundef align 1 dereferenceable(3) %1, i64 3, i1 false), !alias.scope !5, !noalias !6
; CHECK: %4 = load i8, i8 addrspace(3)* %3, align 4, !alias.scope !3, !noalias !0
; CHECK: tail call void @llvm.memcpy.p3i8.p3i8.i64(i8 addrspace(3)* noundef align 1 dereferenceable(3) %7, i8 addrspace(3)* noundef align 1 dereferenceable(3) %6, i64 3, i1 false), !alias.scope !5, !noalias !6
; CHECK: %9 = load i8, i8 addrspace(3)* %8, align 4, !alias.scope !3, !noalias !0

define protected amdgpu_kernel void @test(i8 addrspace(1)* nocapture %ptr.coerce) local_unnamed_addr #0 {
entry:
  store i8 3, i8 addrspace(3)* getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), align 1
  tail call void @llvm.memcpy.p3i8.p3i8.i64(i8 addrspace(3)* noundef align 1 dereferenceable(3) getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), i8 addrspace(3)* noundef align 1 dereferenceable(3) getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), i64 3, i1 false)
  %0 = load i8, i8 addrspace(3)* getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), align 1
  %cmp.i.i = icmp eq i8 %0, 3
  store i8 2, i8 addrspace(3)* getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), align 1
  tail call void @llvm.memcpy.p3i8.p3i8.i64(i8 addrspace(3)* noundef align 1 dereferenceable(3) getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), i8 addrspace(3)* noundef align 1 dereferenceable(3) getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), i64 3, i1 false)
  %1 = load i8, i8 addrspace(3)* getelementptr inbounds (%vec_type, %vec_type addrspace(3)* @_f2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0), align 1
  %cmp.i.i19 = icmp eq i8 %1, 2
  %2 = and i1 %cmp.i.i19, %cmp.i.i
  %frombool8 = zext i1 %2 to i8
  store i8 %frombool8, i8 addrspace(1)* %ptr.coerce, align 1
  ret void
}

declare void @llvm.memcpy.p3i8.p3i8.i64(i8 addrspace(3)* noalias nocapture writeonly, i8 addrspace(3)* noalias nocapture readonly, i64, i1 immarg) #1

; CHECK:!0 = !{!1}
; CHECK:!1 = distinct !{!1, !2}
; CHECK:!2 = distinct !{!2}
; CHECK:!3 = !{!4}
; CHECK:!4 = distinct !{!4, !2}
; CHECK:!5 = !{!4, !1}
; CHECK:!6 = !{}
