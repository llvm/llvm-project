; RUN: not llc -mtriple spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_float_controls2 -o -
; Assertion `isImm() && "Wrong MachineOperand accessor"' failed
; On TypeMI->getOperand(1).getImm() then TypeMI is OpTypeArray %8, %17

@A = addrspace(1) constant [1 x i8] zeroinitializer

define spir_kernel void @foo() {
entry:
  ret void
}
