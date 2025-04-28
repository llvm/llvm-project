; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK: DIVERGENT:  %ret1 = call i64 @llvm.amdgcn.rts.read.result.all.stop()
define amdgpu_cs void @test_rts_read_result_all_stop() {
  %ret1 = call i64 @llvm.amdgcn.rts.read.result.all.stop()
  ret void
}

; CHECK: DIVERGENT:  %ret1 = call i64 @llvm.amdgcn.rts.read.result.ongoing()
define amdgpu_cs void @test_rts_read_result_ongoing() {
  %ret1 = call i64 @llvm.amdgcn.rts.read.result.ongoing()
  ret void
}

; CHECK: DIVERGENT:  %ret_two = call i32 @llvm.amdgcn.rts.ray.save()
define amdgpu_cs void @test_rts_save(){
  %ret_two = call i32 @llvm.amdgcn.rts.ray.save()
  ret void
}

; CHECK: DIVERGENT:  %ret = call i32 @llvm.amdgcn.rts.update.ray(i64 inreg %arg)
define amdgpu_cs void @test_rts_update_ray(i64 inreg %arg){
  %ret = call i32 @llvm.amdgcn.rts.update.ray(i64 inreg %arg)
  ret void
}

; CHECK: ALL VALUES UNIFORM
define amdgpu_cs void @rts_trace_ray_nonblock_test(i32 inreg %ray_init_data, <3 x i32> inreg %ray_init_flag, float inreg %ray_extent, <3 x float> inreg %ray_origin, <3 x float> inreg %ray_dir,  <4 x i32> inreg %rsrc){
  %ret = call i32 @llvm.amdgcn.rts.trace.ray.nonblock(i32 inreg %ray_init_data, <3 x i32> inreg %ray_init_flag, float inreg %ray_extent, <3 x float> inreg %ray_origin, <3 x float> inreg %ray_dir, <4 x i32> inreg %rsrc)
  ret void
}

; CHECK: DIVERGENT:  %ret = call <9 x float> @llvm.amdgcn.rts.read.vertex(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2)
define amdgpu_cs <9 x float> @rts_read_vertex_test(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2){
  %ret = call <9 x float> @llvm.amdgcn.rts.read.vertex(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2)
  ret  <9 x float> %ret
}

; CHECK: DIVERGENT:  %ret = call i32 @llvm.amdgcn.rts.read.packet.info(ptr addrspace(1) inreg %gaddr)
define amdgpu_cs void @test_rts_read_packet_info(ptr addrspace(1) inreg %gaddr){
  %ret = call i32 @llvm.amdgcn.rts.read.packet.info(ptr addrspace(1) inreg %gaddr)
  call void @llvm.amdgcn.exp.i32(i32 20, i32 15, i32 %ret, i32 undef, i32 undef, i32 undef, i1 true, i1 false)
  ret void
}

; CHECK: DIVERGENT:  %ret = call <3 x float> @llvm.amdgcn.rts.read.vertex.coords(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2)
define amdgpu_cs <3 x float>  @rts_read_vertex_coords(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2){
  %ret = call <3 x float> @llvm.amdgcn.rts.read.vertex.coords(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2)
  ret <3 x float> %ret
}

; CHECK:DIVERGENT:  %ret = call <3 x float> @llvm.amdgcn.rts.read.prim.info(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2)
define amdgpu_cs <3 x float>  @rts_read_prim_info(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2){
  %ret = call <3 x float> @llvm.amdgcn.rts.read.prim.info(ptr addrspace(1) inreg %gaddr, i32 inreg %addr2)
  ret <3 x float> %ret
}

; CHECK: DIVERGENT:  %ret = call i32 @llvm.amdgcn.swc.reorder(i64 inreg %arg)
define amdgpu_cs void @test_swc_reorder(i64 inreg %arg){
  %ret = call i32 @llvm.amdgcn.swc.reorder(i64 inreg %arg)
  ret void
}

; CHECK: DIVERGENT:  %ret = call i32 @llvm.amdgcn.swc.reorder.swap(i64 inreg %arg)
define amdgpu_cs void @test_swc_reorder_swap(i64 inreg %arg){
  %ret = call i32 @llvm.amdgcn.swc.reorder.swap(i64 inreg %arg)
  ret void
}

; CHECK: DIVERGENT:  %ret = call i32 @llvm.amdgcn.swc.reorder.swap.resume()
define amdgpu_cs void @test_swc_reorder_swap_resume(){
  %ret = call i32 @llvm.amdgcn.swc.reorder.swap.resume()
  ret void
}
