; RUN: opt -S -scalarizer -scalarize-load-store -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; RUN: llc %s -mtriple=dxil-pc-shadermodel6.3-library --filetype=asm -o - | FileCheck %s

@"sharedData" = local_unnamed_addr addrspace(3) global [2 x <3 x float>] zeroinitializer, align 16 
; CHECK-LABEL: store_test
define void @store_test () local_unnamed_addr {
    ; CHECK: store float 1.000000e+00, ptr addrspace(3) {{.*}}, align {{.*}} 
    ; CHECK: store float 2.000000e+00, ptr addrspace(3) {{.*}}, align {{.*}}
    ; CHECK: store float 3.000000e+00, ptr addrspace(3) {{.*}}, align {{.*}} 
    ; CHECK: store float 2.000000e+00, ptr addrspace(3) {{.*}}, align {{.*}} 
    ; CHECK: store float 4.000000e+00, ptr addrspace(3) {{.*}}, align {{.*}} 
    ; CHECK: store float 6.000000e+00, ptr addrspace(3) {{.*}}, align {{.*}} 

    store <3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, ptr addrspace(3) @"sharedData", align 16 
    store <3 x float> <float 2.000000e+00, float 4.000000e+00, float 6.000000e+00>, ptr addrspace(3)   getelementptr inbounds (i8, ptr addrspace(3) @"sharedData", i32 16), align 16 
    ret void
 } 
