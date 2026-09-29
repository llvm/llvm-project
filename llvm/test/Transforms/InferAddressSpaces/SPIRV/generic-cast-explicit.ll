; This test checks that the address space casts for SPIR-V generic pointer casts
; are lowered correctly by the infer-address-spaces pass.
; RUN: opt < %s -passes=infer-address-spaces -S --mtriple=spirv64-unknown-unknown | FileCheck %s

; Casting a global pointer to a global pointer. 
; The uses of c2 will be replaced with %global.
; CHECK: @kernel1(ptr addrspace(1) %global)
define i1 @kernel1(ptr addrspace(1) %global) {
    %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c2 = call ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr addrspace(1) %global, null
    %b1 = icmp eq ptr addrspace(1) %c2, null 
    ret i1 %b1
}

; Casting a global pointer to a local pointer.
; The uses of c2 will be replaced with null.
; CHECK: @kernel2(ptr addrspace(1) %global)
define i1 @kernel2(ptr addrspace(1) %global) {
    %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c2 = call ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr addrspace(3) null, null
    %b1 = icmp eq ptr addrspace(3) %c2, null
    ret i1 %b1
}

; Casting a global pointer to a private pointer.
; The uses of c2 will be replaced with null.
; CHECK: @kernel3(ptr addrspace(1) %global)
define i1 @kernel3(ptr addrspace(1) %global) {
    %c1 = addrspacecast ptr addrspace(1) %global to ptr addrspace(4)
    %c2 = call ptr @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr null, null
    %b1 = icmp eq ptr %c2, null 
    ret i1 %b1
}

; Casting a local pointer to a local pointer.
; The uses of c2 will be replaced with %local.
; CHECK: @kernel4(ptr addrspace(3) %local)
define i1 @kernel4(ptr addrspace(3) %local) {
    %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c2 = call ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr addrspace(3) %local, null
    %b1 = icmp eq ptr addrspace(3) %c2, null 
    ret i1 %b1
}

; Casting a local pointer to a global pointer.
; The uses of c2 will be replaced with null.
; CHECK: @kernel5(ptr addrspace(3) %local)
define i1 @kernel5(ptr addrspace(3) %local) {
    %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c2 = call ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr addrspace(1) null, null
    %b1 = icmp eq ptr addrspace(1) %c2, null 
    ret i1 %b1
}

; Casting a local pointer to a private pointer.
; The uses of c2 will be replaced with null.
; CHECK: @kernel6(ptr addrspace(3) %local)
define i1 @kernel6(ptr addrspace(3) %local) {
    %c1 = addrspacecast ptr addrspace(3) %local to ptr addrspace(4)
    %c2 = call ptr @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr null, null
    %b1 = icmp eq ptr %c2, null 
    ret i1 %b1
}

; Casting a private pointer to a private pointer.
; The uses of c2 will be replaced with %private.
; CHECK: @kernel7(ptr %private)
define i1 @kernel7(ptr %private) {
    %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c2 = call ptr @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr %private, null
    %b1 = icmp eq ptr %c2, null 
    ret i1 %b1
}

; Casting a private pointer to a global pointer.
; The uses of c2 will be replaced with null.
; CHECK: @kernel8(ptr %private)
define i1 @kernel8(ptr %private) {
    %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c2 = call ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr addrspace(1) null, null
    %b1 = icmp eq ptr addrspace(1) %c2, null
    ret i1 %b1
}

; Casting a private pointer to a local pointer.
; The uses of c2 will be replaced with null.
; CHECK: @kernel9(ptr %private)
define i1 @kernel9(ptr %private) {
    %c1 = addrspacecast ptr %private to ptr addrspace(4)
    %c2 = call ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit(ptr addrspace(4) %c1)
    ; CHECK: %b1 = icmp eq ptr addrspace(3) null, null
    %b1 = icmp eq ptr addrspace(3) %c2, null
    ret i1 %b1
}
