; RUN: not --crash llc -O2 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash llc -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

; addrspacecast(inttoptr(4096)) followed by a constant GEP also reaches
; G_PTR_ADD with a G_ADDRSPACE_CAST base. Selection aborts on the same base
; assertion as a cast of a global; it does not emit a diagnostic.

; CHECK{LITERAL}: Assertion `((*II).getOpcode() == TargetOpcode::G_GLOBAL_VALUE || (*II).getOpcode() == TargetOpcode::COPY || (*II).getOpcode() == SPIRV::OpVariable || (*II).getOpcode() == SPIRV::OpUntypedVariableKHR) && getImm(I.getOperand(2), MRI)' failed.
; CHECK: Running pass 'InstructionSelect' on function '@f'

@p = addrspace(1) global ptr addrspace(4) getelementptr (
    i8, ptr addrspace(4) addrspacecast (
        ptr addrspace(1) inttoptr (i64 4096 to ptr addrspace(1))
        to ptr addrspace(4)),
    i64 8), align 8

define spir_func void @f() addrspace(4) {
  unreachable
}
