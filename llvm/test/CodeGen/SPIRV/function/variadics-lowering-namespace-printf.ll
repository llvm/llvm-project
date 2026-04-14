; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_function_pointers < %s 2>&1 | FileCheck %s

; CHECK: OpName [[MAIN:%.*]] "main"
; CHECK: [[MAIN]] = OpFunction
; CHECK-NOT: OPFunctionEnd
; CHECK: OpLifetimeStart
; CHECK-NOT: OPFunctionEnd
; CHECK: OpLifetimeStop
; CHECK: OpFunctionEnd

@.str = private unnamed_addr addrspace(1) constant [3 x i8] c"%s\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [4 x i8] c"hey\00", align 1

declare spir_func noundef i32 @_Z7vprintfPKcz(ptr addrspace(4) noundef %0, ...) addrspace(9)

define spir_func noundef i32 @_ZN4ompx6printfEPKcz(ptr addrspace(4) noundef %Format, ...) addrspace(9) {
entry:
  %retval = alloca i32, align 4
  %Format.addr = alloca ptr addrspace(4), align 8
  %vlist = alloca ptr addrspace(4), align 8
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  %Format.addr.ascast = addrspacecast ptr %Format.addr to ptr addrspace(4)
  %vlist.ascast = addrspacecast ptr %vlist to ptr addrspace(4)
  store ptr addrspace(4) %Format, ptr addrspace(4) %Format.addr.ascast, align 8
  call addrspace(9) void @llvm.va_start.p4(ptr addrspace(4) %vlist.ascast)
  %0 = load ptr addrspace(4), ptr addrspace(4) %Format.addr.ascast, align 8
  %1 = load ptr addrspace(4), ptr addrspace(4) %vlist.ascast, align 8
  %call = call spir_func noundef addrspace(9) i32 (ptr addrspace(4), ...) @_Z7vprintfPKcz(ptr addrspace(4) noundef %0, ptr addrspace(4) noundef %1)
  ret i32 %call
}

declare void @llvm.va_start.p4(ptr addrspace(4)) addrspace(9)

define noundef i32 @main() addrspace(9) {
entry:
  %retval = alloca i32, align 4
  %retval.ascast = addrspacecast ptr %retval to ptr addrspace(4)
  store i32 0, ptr addrspace(4) %retval.ascast, align 4
  %call = call spir_func noundef addrspace(9) i32 (ptr addrspace(4), ...) @_ZN4ompx6printfEPKcz(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @.str.1 to ptr addrspace(4))) #5
  ret i32 0
}
