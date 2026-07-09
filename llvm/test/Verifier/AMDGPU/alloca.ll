; RUN: not llvm-as %s --disable-output 2>&1 | FileCheck %s

target triple = "amdgpu7.00-amd-amdhsa"

; A static alloca is allowed in addrspace(5) (private) and addrspace(13) (VGPR);
; any other address space is rejected.
; CHECK: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.0 = alloca i32, align 4
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.1 = alloca i32, align 4, addrspace(1)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.2 = alloca i32, align 4, addrspace(2)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.3 = alloca i32, align 4, addrspace(3)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.4 = alloca i32, align 4, addrspace(4)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.6 = alloca i32, align 4, addrspace(6)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.7 = alloca i32, align 4, addrspace(7)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.8 = alloca i32, align 4, addrspace(8)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.9 = alloca i32, align 4, addrspace(9)
define void @static_alloca() {
entry:
  %alloca.0 = alloca i32, align 4
  %alloca.1 = alloca i32, align 4, addrspace(1)
  %alloca.2 = alloca i32, align 4, addrspace(2)
  %alloca.3 = alloca i32, align 4, addrspace(3)
  %alloca.4 = alloca i32, align 4, addrspace(4)
  %alloca.5 = alloca i32, align 4, addrspace(5)
  %alloca.6 = alloca i32, align 4, addrspace(6)
  %alloca.7 = alloca i32, align 4, addrspace(7)
  %alloca.8 = alloca i32, align 4, addrspace(8)
  %alloca.9 = alloca i32, align 4, addrspace(9)
  %alloca.13 = alloca i32, align 4, addrspace(13)
  ret void
}

; A dynamically sized alloca is only allowed in addrspace(5); addrspace(13) is
; rejected because it has no register-file representation.
; CHECK: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.0 = alloca i32, i32 %n, align 4
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.1 = alloca i32, i32 %n, align 4, addrspace(1)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.2 = alloca i32, i32 %n, align 4, addrspace(2)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.3 = alloca i32, i32 %n, align 4, addrspace(3)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.4 = alloca i32, i32 %n, align 4, addrspace(4)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.6 = alloca i32, i32 %n, align 4, addrspace(6)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.7 = alloca i32, i32 %n, align 4, addrspace(7)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.8 = alloca i32, i32 %n, align 4, addrspace(8)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.9 = alloca i32, i32 %n, align 4, addrspace(9)
; CHECK-NEXT: dynamic alloca on amdgpu must be in addrspace(5)
; CHECK-NEXT: %alloca.13 = alloca i32, i32 %n, align 4, addrspace(13)
define void @dynamic_alloca_i32(i32 %n) {
entry:
  %alloca.0 = alloca i32, i32 %n, align 4
  %alloca.1 = alloca i32, i32 %n, align 4, addrspace(1)
  %alloca.2 = alloca i32, i32 %n, align 4, addrspace(2)
  %alloca.3 = alloca i32, i32 %n, align 4, addrspace(3)
  %alloca.4 = alloca i32, i32 %n, align 4, addrspace(4)
  %alloca.5 = alloca i32, i32 %n, align 4, addrspace(5)
  %alloca.6 = alloca i32, i32 %n, align 4, addrspace(6)
  %alloca.7 = alloca i32, i32 %n, align 4, addrspace(7)
  %alloca.8 = alloca i32, i32 %n, align 4, addrspace(8)
  %alloca.9 = alloca i32, i32 %n, align 4, addrspace(9)
  %alloca.13 = alloca i32, i32 %n, align 4, addrspace(13)
  ret void
}

; CHECK: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.0 = alloca i32, i64 %n, align 4
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.1 = alloca i32, i64 %n, align 4, addrspace(1)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.2 = alloca i32, i64 %n, align 4, addrspace(2)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.3 = alloca i32, i64 %n, align 4, addrspace(3)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.4 = alloca i32, i64 %n, align 4, addrspace(4)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.6 = alloca i32, i64 %n, align 4, addrspace(6)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.7 = alloca i32, i64 %n, align 4, addrspace(7)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.8 = alloca i32, i64 %n, align 4, addrspace(8)
; CHECK-NEXT: alloca on amdgpu must be in addrspace(5) or addrspace(13)
; CHECK-NEXT: %alloca.9 = alloca i32, i64 %n, align 4, addrspace(9)
define void @dynamic_alloca_i64(i64 %n) {
entry:
  %alloca.0 = alloca i32, i64 %n, align 4
  %alloca.1 = alloca i32, i64 %n, align 4, addrspace(1)
  %alloca.2 = alloca i32, i64 %n, align 4, addrspace(2)
  %alloca.3 = alloca i32, i64 %n, align 4, addrspace(3)
  %alloca.4 = alloca i32, i64 %n, align 4, addrspace(4)
  %alloca.5 = alloca i32, i64 %n, align 4, addrspace(5)
  %alloca.6 = alloca i32, i64 %n, align 4, addrspace(6)
  %alloca.7 = alloca i32, i64 %n, align 4, addrspace(7)
  %alloca.8 = alloca i32, i64 %n, align 4, addrspace(8)
  %alloca.9 = alloca i32, i64 %n, align 4, addrspace(9)
  ret void
}

; A static alloca that is not in the entry block is treated as dynamic, so
; addrspace(13) is rejected there too.
; CHECK: dynamic alloca on amdgpu must be in addrspace(5)
; CHECK-NEXT: %alloca.13 = alloca i32, align 4, addrspace(13)
define void @nonentry_alloca() {
entry:
  br label %nonentry

nonentry:
  %alloca.13 = alloca i32, align 4, addrspace(13)
  ret void
}
