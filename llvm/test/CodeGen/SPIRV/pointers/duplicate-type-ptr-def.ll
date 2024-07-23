; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#Char:]] = OpTypeInt 8 0
; CHECK: %[[#Ptr:]] = OpTypePointer Generic %[[#Char]]
; CHECK: %[[#TypeDef:]] = OpTypePointer Generic %[[#Ptr]]
; CHECK-NOT: %[[#TypeDef]] = OpTypePointer Generic %[[#Ptr]]

%Range = type { %Array }
%Array = type { [1 x i64] }

define spir_func ptr addrspace(4) @foo(ptr addrspace(4) dereferenceable_or_null(32) %this) {
entry:
  %addr = alloca ptr addrspace(4)
  store ptr addrspace(4) %this, ptr %addr
  %arrayidx = getelementptr inbounds ptr addrspace(4), ptr addrspace(1) null, i64 0
  %r = addrspacecast ptr addrspace(1) %arrayidx to ptr addrspace(4)
  ret ptr addrspace(4) %r
}

define spir_func void @bar() {
entry:
  %retval = alloca ptr addrspace(4)
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  ret void
}
