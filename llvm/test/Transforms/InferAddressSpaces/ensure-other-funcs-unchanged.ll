; RUN: opt -assume-default-is-flat-addrspace -print-module-scope -print-after-all -S -disable-output -passes=infer-address-spaces <%s 2>&1 | FileCheck %s

; CHECK: IR Dump After InferAddressSpacesPass on f2

; Check that after running infer-address-spaces on f2, the redundant addrspace cast %x1 in f2 is gone.
; CHECK-LABEL: define spir_func void @f2()
; CHECK:         [[X:%.*]] = addrspacecast ptr addrspace(1) @x to ptr
; CHECK-NEXT:    call spir_func void @f1(ptr noundef [[X]])

; But it should not affect f3.
; CHECK-LABEL: define spir_func void @f3()
; CHECK:         %x1 = addrspacecast ptr addrspacecast (ptr addrspace(1) @x to ptr) to ptr addrspace(1)
; CHECK-NEXT:    %x2 = addrspacecast ptr addrspace(1) %x1 to ptr
; CHECK-NEXT:    call spir_func void @f1(ptr noundef %x2)

; Ensure that the pass hasn't run on f3 yet.
; CHECK: IR Dump After InferAddressSpacesPass on f3

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

@x = addrspace(1) global i32 0, align 4

define spir_func void @f2() {
entry:
  %x1 = addrspacecast ptr addrspacecast (ptr addrspace(1) @x to ptr) to ptr addrspace(1)
  %x2 = addrspacecast ptr addrspace(1) %x1 to ptr
  call spir_func void @f1(ptr noundef %x2)
  ret void
}

define spir_func void @f3() {
entry:
  %x1 = addrspacecast ptr addrspacecast (ptr addrspace(1) @x to ptr) to ptr addrspace(1)
  %x2 = addrspacecast ptr addrspace(1) %x1 to ptr
  call spir_func void @f1(ptr noundef %x2)
  ret void
}

declare spir_func void @f1(ptr noundef)
