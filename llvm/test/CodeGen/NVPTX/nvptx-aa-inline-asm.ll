; RUN: opt -passes=aa-eval -aa-pipeline=nvptx-aa,basic-aa -print-all-alias-modref-info < %s -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefixes CHECK-ALIAS

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

;;CHECK-ALIAS-LABEL: Function: test_sideeffect
;;CHECK-ALIAS: Both ModRef: Ptr: i32* %0 <-> call
define void @test_sideeffect(ptr %out) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  call void asm sideeffect "membar.gl;", ""()
  store i32 5, ptr addrspace(1) %0, align 4
  ret void
}

;;CHECK-ALIAS-LABEL: Function: test_indirect
;;CHECK-ALIAS: Both ModRef: Ptr: i32* %0 <-> %1 = call
define i32 @test_indirect(ptr %out) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  store i32 0, ptr addrspace(1) %0, align 4
  %1 = call i32 asm "ld.global.u32 $0, [$1];", "=r,*m"(ptr addrspace(1) elementtype(i32) %0)
  store i32 0, ptr addrspace(1) %0, align 4
  ret i32 %1
}

;;CHECK-ALIAS-LABEL: Function: test_memory
;;CHECK-ALIAS: Both ModRef: Ptr: i32* %0 <-> %1 = call
define i32 @test_memory(ptr %out) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  store i32 0, ptr addrspace(1) %0, align 4
  %1 = call i32 asm "ld.global.u32 $0, [$1];", "=r,l,~{memory}"(ptr addrspace(1) %0)
  store i32 0, ptr addrspace(1) %0, align 4
  ret i32 %1
}

;;CHECK-ALIAS-LABEL: Function: test_no_sideeffect
;;CHECK-ALIAS: NoModRef: Ptr: i32* %0 <-> %1 = call
define void @test_no_sideeffect(ptr %in, ptr %out) {
entry:
  %0 = addrspacecast ptr %out to ptr addrspace(1)
  %1 = call i32 asm "cvt.u32.u64 $0, $1;", "=r,l"(ptr %in)
  store i32 %1, ptr addrspace(1) %0, align 4
  ret void
}
