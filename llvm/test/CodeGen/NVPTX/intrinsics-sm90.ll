; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80| FileCheck --check-prefixes=CHECK %s
; RUN: %if ptxas-sm_90 && ptxas-isa-8.0 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_90 -mattr=+ptx80| %ptxas-verify -arch=sm_90 %}

; CHECK-LABEL: test_isspacep
define i1 @test_isspacep_shared_cluster(ptr %p) {
; CHECK: isspacep.shared::cluster
  %a = tail call i1 @llvm.nvvm.isspacep.shared.cluster(ptr %p)
; CHECK: ret
  ret i1 %a
}

; CHECK-LABEL: test_mapa(
define ptr @test_mapa(ptr %p, i32 %r) {
; CHECK64: mapa.u64
  %a = call ptr @llvm.nvvm.mapa(ptr %p, i32 %r)
  ret ptr %a
}

; CHECK-LABEL: test_mapa_shared_cluster(
define ptr addrspace(3) @test_mapa_shared_cluster(ptr addrspace(3) %p, i32 %r) {
; CHECK: mapa.shared::cluster.u64
  %a = call ptr addrspace(3) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %p, i32 %r)
  ret ptr addrspace(3) %a
}

; CHECK-LABEL: test_getctarank(
define i32 @test_getctarank(ptr %p) {
; CHECK: getctarank.u64
  %a = call i32 @llvm.nvvm.getctarank(ptr %p)
  ret i32 %a
}

; CHECK-LABEL: test_getctarank_shared_cluster(
define i32 @test_getctarank_shared_cluster(ptr addrspace(3) %p) {
; CHECK64: getctarank.shared::cluster.u64
; CHECK32: getctarank.shared::cluster.u32
  %a = call i32 @llvm.nvvm.getctarank.shared.cluster(ptr addrspace(3) %p)
  ret i32 %a
}

; CHECK-LABEL: test_clusterid_x(
define i32 @test_clusterid_x() {
; CHECK: mov.u32 %r{{[0-9]+}}, %clusterid.x;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.x()
        ret i32 %x
}
; CHECK-LABEL: test_clusterid_y(
define i32 @test_clusterid_y() {
; CHECK: mov.u32 %r{{[0-9]+}}, %clusterid.y;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.y()
        ret i32 %x
}
; CHECK-LABEL: test_clusterid_z(
define i32 @test_clusterid_z() {
; CHECK: mov.u32 %r{{[0-9]+}}, %clusterid.z;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.z()
        ret i32 %x
}
; CHECK-LABEL: test_clusterid_w(
define i32 @test_clusterid_w() {
; CHECK: mov.u32 %r{{[0-9]+}}, %clusterid.w;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.clusterid.w()
        ret i32 %x
}

; CHECK-LABEL: test_nclusterid_x(
define i32 @test_nclusterid_x() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nclusterid.x;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.x()
        ret i32 %x
}
; CHECK-LABEL: test_nclusterid_y(
define i32 @test_nclusterid_y() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nclusterid.y;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.y()
        ret i32 %x
}
; CHECK-LABEL: test_nclusterid_z(
define i32 @test_nclusterid_z() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nclusterid.z;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.z()
        ret i32 %x
}
; CHECK-LABEL: test_nclusterid_w(
define i32 @test_nclusterid_w() {
; CHECK: mov.u32 %r{{[0-9]+}}, %nclusterid.w;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.w()
        ret i32 %x
}

; CHECK-LABEL: test_cluster_ctarank(
define i32 @test_cluster_ctarank() {
; CHECK: mov.u32 %r{{[0-9]+}}, %cluster_ctarank;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctarank()
        ret i32 %x
}

; CHECK-LABEL: test_cluster_nctarank(
define i32 @test_cluster_nctarank() {
; CHECK: mov.u32 %r{{[0-9]+}}, %cluster_nctarank;
; CHECK: ret;
        %x = call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctarank()
        ret i32 %x
}

; CHECK-LABEL: test_is_explicit_cluster(
define i1 @test_is_explicit_cluster() {
; CHECK: mov.pred %p{{[0-9]+}}, %is_explicit_cluster;
; CHECK: ret;
        %x = call i1 @llvm.nvvm.is_explicit_cluster()
        ret i1 %x
}

; CHECK-LABEL: test_barrier_cluster(
define void @test_barrier_cluster() {
; CHECK: barrier.cluster.arrive;
       call void @llvm.nvvm.barrier.cluster.arrive()
; CHECK: barrier.cluster.arrive.relaxed;
       call void @llvm.nvvm.barrier.cluster.arrive.relaxed()
; CHECK: barrier.cluster.wait;
       call void @llvm.nvvm.barrier.cluster.wait()
; CHECK: fence.sc.cluster
       call void @llvm.nvvm.fence.sc.cluster()
       ret void
}

; CHECK-LABEL: test_barrier_cluster_aligned(
define void @test_barrier_cluster_aligned() {
; CHECK: barrier.cluster.arrive.aligned;
       call void @llvm.nvvm.barrier.cluster.arrive.aligned()
; CHECK: barrier.cluster.arrive.relaxed.aligned;
       call void @llvm.nvvm.barrier.cluster.arrive.relaxed.aligned()
; CHECK: barrier.cluster.wait.aligned;
       call void @llvm.nvvm.barrier.cluster.wait.aligned()
       ret void
}

; CHECK-LABEL: test_cp_async_bulk_commit_group(
define void @test_cp_async_bulk_commit_group() {
; CHECK: cp.async.bulk.commit_group;
       call void @llvm.nvvm.cp.async.bulk.commit.group()
       ret void
}

; CHECK-LABEL: test_cp_async_bulk_wait_group(
define void @test_cp_async_bulk_wait_group() {
; CHECK: cp.async.bulk.wait_group 8;
       call void @llvm.nvvm.cp.async.bulk.wait.group(i32 8)
; CHECK: cp.async.bulk.wait_group 0;
       call void @llvm.nvvm.cp.async.bulk.wait.group(i32 0)
       ret void
}

; CHECK-LABEL: test_cp_async_bulk_wait_group_read(
define void @test_cp_async_bulk_wait_group_read() {
; CHECK: cp.async.bulk.wait_group.read 8;
       call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 8)
; CHECK: cp.async.bulk.wait_group.read 0;
       call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 0)
       ret void
}

declare i1 @llvm.nvvm.isspacep.shared.cluster(ptr %p);
declare ptr @llvm.nvvm.mapa(ptr %p, i32 %r);
declare ptr addrspace(3) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %p, i32 %r);
declare i32 @llvm.nvvm.getctarank(ptr %p);
declare i32 @llvm.nvvm.getctarank.shared.cluster(ptr addrspace(3) %p);
declare i32 @llvm.nvvm.read.ptx.sreg.clusterid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.clusterid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.clusterid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.clusterid.w()
declare i32 @llvm.nvvm.read.ptx.sreg.nclusterid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nclusterid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nclusterid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nclusterid.w()
declare i32 @llvm.nvvm.read.ptx.sreg.cluster.ctarank()
declare i32 @llvm.nvvm.read.ptx.sreg.cluster.nctarank()
declare i1 @llvm.nvvm.is_explicit_cluster()
declare void @llvm.nvvm.barrier.cluster.arrive()
declare void @llvm.nvvm.barrier.cluster.arrive.relaxed()
declare void @llvm.nvvm.barrier.cluster.wait()
declare void @llvm.nvvm.barrier.cluster.arrive.aligned()
declare void @llvm.nvvm.barrier.cluster.arrive.relaxed.aligned()
declare void @llvm.nvvm.barrier.cluster.wait.aligned()
declare void @llvm.nvvm.fence.sc.cluster()
declare void @llvm.nvvm.cp.async.bulk.commit.group()
declare void @llvm.nvvm.cp.async.bulk.wait.group(i32)
declare void @llvm.nvvm.cp.async.bulk.wait.group.read(i32)
