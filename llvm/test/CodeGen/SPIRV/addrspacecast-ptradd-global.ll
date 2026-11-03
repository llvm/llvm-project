; RUN: not --crash llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash llc -verify-machineinstrs -O2 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not --crash llc -verify-machineinstrs -O2 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

; A constant byte offset from an address-space cast reaches instruction
; selection as G_ADDRSPACE_CAST followed by G_PTR_ADD. The selector does not
; accept that base and aborts.

; CHECK{LITERAL}: Assertion `((*II).getOpcode() == TargetOpcode::G_GLOBAL_VALUE || (*II).getOpcode() == TargetOpcode::COPY || (*II).getOpcode() == SPIRV::OpVariable || (*II).getOpcode() == SPIRV::OpUntypedVariableKHR) && getImm(I.getOperand(2), MRI)' failed.
; CHECK: Running pass 'InstructionSelect' on function '@f'

%struct.test = type { i64, [2 x i64], i64 }

@obj = addrspace(1) global %struct.test zeroinitializer, align 8
@obj_ptr = addrspace(1) global ptr addrspace(4) addrspacecast (ptr addrspace(1) @obj to ptr addrspace(4)), align 8
@obj_y = addrspace(1) global ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @obj to ptr addrspace(4)), i64 8), align 8
@obj_n = addrspace(1) global ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @obj to ptr addrspace(4)), i64 16), align 8

@lds = addrspace(3) global [16 x i8] poison, align 1
@lds_off = addrspace(1) global ptr addrspace(1) getelementptr (i8, ptr addrspace(1) addrspacecast (ptr addrspace(3) @lds to ptr addrspace(1)), i64 4), align 8

@null_off = addrspace(1) global ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) null to ptr addrspace(4)), i64 72), align 8

@arr = addrspace(1) global [32 x i8] zeroinitializer, align 1
@table = addrspace(1) global [3 x ptr addrspace(4)] [ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) @arr to ptr addrspace(4)), i64 8), ptr addrspace(4) addrspacecast (ptr addrspace(1) @arr to ptr addrspace(4)), ptr addrspace(4) getelementptr (i8, ptr addrspace(4) addrspacecast (ptr addrspace(1) null to ptr addrspace(4)), i64 8)], align 8

define spir_func void @f() addrspace(4) {
  unreachable
}
