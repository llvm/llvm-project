define amdgpu_kernel void @divergent_value(ptr addrspace(1) %out, ptr addrspace(1) %maskarr, i32 %in) {
    
    entry:
    %id.x = call i32 @llvm.amdgcn.workitem.id.x()
    %mask_ptr = getelementptr inbounds i32, ptr addrspace(1) %maskarr, i32 %id.x  
    ; %mask_ptr_casted = bitcast ptr addrspace(1) %mask_ptr to ptr  
    %mask = load i32, ptr addrspace(1) %mask_ptr
    %result = call i32 @llvm.amdgcn.wave.reduce.umax.i32(i32 %id.x, i32 %mask, i32 1)
    store i32 %result, ptr addrspace(1) %out
    ret void
    
}