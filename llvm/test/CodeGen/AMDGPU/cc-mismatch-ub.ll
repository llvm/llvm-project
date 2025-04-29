; RUN: not llc -mtriple=amdgcn -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck -check-prefix=CHECK-NO-ISEL %s  
; CHECK-NO-ISEL: error: {{.*}} in function main {{.*}} calling convention mismatch (undefined behavior) foo  
  
; RUN: not --crash llc -debug -global-isel=1 -mtriple=amdgcn -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck -check-prefix=CHECK-ISEL %s  
; CHECK-ISEL: Failed to lower call: calling convention mismatch (undefined behavior)  

; COM: This test aims to identify invalid method calls.  
; COM: By doing so, it simplifies debugging by exposing issues earlier in the pipeline.  

define amdgpu_ps i32 @foo(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1) {
 ret i32 0  
}

define amdgpu_ps i32 @main(ptr addrspace(4) inreg %arg, ptr addrspace(4) inreg %arg1) {
main_body:
  %C = call i32 @foo(ptr addrspace(4) null, ptr addrspace(4) %arg)
  ret i32  %C
}