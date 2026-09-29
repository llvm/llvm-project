; RUN: opt -S --dxil-prepare %s | FileCheck %s

; Test that global arrays do not get a bitcast instruction
; after the dxil-prepare pass.

target triple = "dxilv1.2-unknown-shadermodel6.2-compute"

@inputTile.1dim = local_unnamed_addr addrspace(3) global [3 x float] zeroinitializer, align 2

; CHECK-LABEL: testload
define float @testload() local_unnamed_addr {
  ; NOTE: this would be "bitcast ptr addrspace(3)..." before the change that introduced this test,
  ; after the dxil-prepare pass is run
  ; CHECK-NEXT: load float, ptr addrspace(3) @inputTile.1dim, align 2
  %v = load float, ptr addrspace(3) @inputTile.1dim, align 2  
  
  ret float %v
}

; CHECK-LABEL: teststore
define void @teststore() local_unnamed_addr {  
  ; CHECK-next: store float 2.000000e+00, ptr addrspace(3) @inputTile.1dim, align 2
  store float 2.000000e+00, ptr addrspace(3) @inputTile.1dim, align 2  
  
  ret void
}

; CHECK-LABEL: testGEPConst
define float @testGEPConst() local_unnamed_addr {  
  ; CHECK-NEXT: load float, ptr addrspace(3) getelementptr (float, ptr addrspace(3) @inputTile.1dim, i32 1), align 4
  %v = load float, ptr addrspace(3) getelementptr (float, ptr addrspace(3) @inputTile.1dim, i32 1), align 4
  
  ret float %v
}

; CHECK-LABEL: testGEPNonConst
define float @testGEPNonConst(i32 %i) local_unnamed_addr {  
  ; CHECK-NEXT: getelementptr float, ptr addrspace(3) @inputTile.1dim, i32 %i
  %gep = getelementptr float, ptr addrspace(3) @inputTile.1dim, i32 %i
  %v = load float, ptr addrspace(3) %gep
  
  ret float %v
}

; CHECK-LABEL: testAlloca
define float @testAlloca(i32 %i) local_unnamed_addr {  
  ; CHECK-NEXT: alloca [3 x float], align 4
  %arr = alloca [3 x float], align 4
  ; CHECK-NEXT: getelementptr [3 x float], ptr %arr, i32 1
  %gep = getelementptr [3 x float], ptr %arr, i32 1
  %v = load float, ptr %gep
  ret float %v
}
