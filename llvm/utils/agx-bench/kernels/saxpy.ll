; y[i] = a * x[i] + y[i]
declare i32 @llvm.agx.thread.position.in.grid.x()
define agx_kernel void @saxpy(ptr addrspace(1) %x, ptr addrspace(1) %y, float %a, i32 %n) {
  %i = call i32 @llvm.agx.thread.position.in.grid.x()
  %in = icmp ult i32 %i, %n
  br i1 %in, label %body, label %end
body:
  %px = getelementptr float, ptr addrspace(1) %x, i32 %i
  %py = getelementptr float, ptr addrspace(1) %y, i32 %i
  %vx = load float, ptr addrspace(1) %px
  %vy = load float, ptr addrspace(1) %py
  %r = call float @llvm.fma.f32(float %a, float %vx, float %vy)
  store float %r, ptr addrspace(1) %py
  br label %end
end:
  ret void
}
declare float @llvm.fma.f32(float, float, float)
