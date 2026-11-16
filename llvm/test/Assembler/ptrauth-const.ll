; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@var = global i32 0

; CHECK: @basic = global ptr ptrauth (ptr @var, i32 0)
@basic = global ptr ptrauth (ptr @var, i32 0)

; CHECK: @keyed = global ptr ptrauth (ptr @var, i32 3)
@keyed = global ptr ptrauth (ptr @var, i32 3)

; CHECK: @intdisc = global ptr ptrauth (ptr @var, i32 0, i64 -1)
@intdisc = global ptr ptrauth (ptr @var, i32 0, i64 -1)

; CHECK: @addrdisc = global ptr ptrauth (ptr @var, i32 2, i64 1234, ptr @addrdisc)
@addrdisc = global ptr ptrauth (ptr @var, i32 2, i64 1234, ptr @addrdisc)

; No operand refers to a global, so this constant is only destroyed with the
; context, together with its constant expression operand.
; CHECK: @inttoptr = global ptr ptrauth (ptr inttoptr (i64 4096 to ptr), i32 0)
@inttoptr = global ptr ptrauth (ptr inttoptr (i64 4096 to ptr), i32 0)


@var1 = addrspace(1) global i32 0

; CHECK: @addrspace = global ptr addrspace(1) ptrauth (ptr addrspace(1) @var1, i32 0)
@addrspace = global ptr addrspace(1) ptrauth (ptr addrspace(1) @var1, i32 0)

; CHECK: @addrspace_addrdisc = addrspace(2) global ptr addrspace(1) ptrauth (ptr addrspace(1) @var1, i32 2, i64 1234, ptr addrspace(2) @addrspace_addrdisc)
@addrspace_addrdisc = addrspace(2) global ptr addrspace(1) ptrauth (ptr addrspace(1) @var1, i32 2, i64 1234, ptr addrspace(2) @addrspace_addrdisc)
