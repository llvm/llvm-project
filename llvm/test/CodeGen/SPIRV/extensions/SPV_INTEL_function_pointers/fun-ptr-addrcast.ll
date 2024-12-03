; The goal of this test case is to check that cases covered by pointers/PtrCast-in-OpSpecConstantOp.ll and
; pointers/PtrCast-null-in-OpSpecConstantOp.ll (that is OpSpecConstantOp with ptr-cast operation) correctly
; work also for function pointers.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - --spirv-ext=+SPV_INTEL_function_pointers | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Running with -verify-machineinstrs would lead to "Reading virtual register without a def"
; error, because OpConstantFunctionPointerINTEL forward-refers to a function definition.

; CHECK-COUNT-3: %[[#]] = OpSpecConstantOp %[[#]] 121 %[[#]]
; CHECK-COUNT-3: OpPtrCastToGeneric

@G1 = addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @bar to ptr addrspace(4))] }
@G2 = addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr null to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @bar to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4))] }

define void @foo(ptr addrspace(4) %p) {
entry:
  %r1 = addrspacecast ptr @foo to ptr addrspace(4)
  %r2 = addrspacecast ptr null to ptr addrspace(4)
  ret void
}

define void @bar(ptr addrspace(4) %p) {
entry:
  %r1 = addrspacecast ptr @bar to ptr addrspace(4)
  ret void
}
