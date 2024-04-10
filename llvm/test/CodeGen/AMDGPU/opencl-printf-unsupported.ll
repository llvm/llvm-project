; RUN: not llc -mtriple=amdgcn-- -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck --strict-whitespace %s

; FIXME: Should be adequate to run just
; -passes=amdgpu-printf-runtime-binding, but llc and opt have
; different behavior on error diagnostics. opt uses the default error
; handler which immediately exits, and llc does not. We want to report
; errors for all functions in the module.


; CHECK: error: foo.cl:1:42: in function test_inttoptr_argument void (i32): printf format string must be a trivially resolved constant string global variable
define amdgpu_kernel void @test_inttoptr_argument(i32 %n) {
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) inttoptr (i64 1234 to ptr addrspace(4)), ptr addrspace(5) %str, i32 %n), !dbg !4
  ret void
}

; CHECK: error: foo.cl:1:42: in function test_formatstr_argument void (ptr addrspace(4), i32): printf format string must be a trivially resolved constant string global variable
define amdgpu_kernel void @test_formatstr_argument(ptr addrspace(4) %format.str, i32 %n) {
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) %format.str, ptr addrspace(5) %str, i32 %n), !dbg !4
  ret void
}

; CHECK: error: foo.cl:1:42: in function test_formatstr_instruction void (ptr addrspace(4), i32): printf format string must be a trivially resolved constant string global variable
define amdgpu_kernel void @test_formatstr_instruction(ptr addrspace(4) %kernarg, i32 %n) {
  %str = alloca [9 x i8], align 1, addrspace(5)
  %format.str = load ptr addrspace(4), ptr addrspace(4) %kernarg
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) %format.str, ptr addrspace(5) %str, i32 %n), !dbg !4
  ret void
}

@no.initializer = external hidden addrspace(4) constant [6 x i8]

; CHECK: error: foo.cl:1:42: in function test_no_initializer_gv void (i32): printf format string must be a trivially resolved constant string global variable
define amdgpu_kernel void @test_no_initializer_gv(i32 %n) {
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @no.initializer, ptr addrspace(5) %str, i32 %n), !dbg !4
  ret void
}

@interposable.initializer = weak unnamed_addr addrspace(4) constant [6 x i8] c"%s:%d\00"

; CHECK: error: foo.cl:1:42: in function poison_interposable_initializer_gv void (i32): printf format string must be a trivially resolved constant string global variable
define amdgpu_kernel void @poison_interposable_initializer_gv(i32 %n) {
  %str = alloca [9 x i8], align 1, addrspace(5)
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @interposable.initializer, ptr addrspace(5) %str, i32 %n), !dbg !4
  ret void
}

@not.constant = private unnamed_addr addrspace(4) global [6 x i8] c"%s:%d\00", align 1


; CHECK: error: <unknown>:0:0: in function not_constant_gv void (i32): printf format string must be a trivially resolved constant string global variable
define amdgpu_kernel void @not_constant_gv(i32 %n) {
entry:
  %str = alloca [9 x i8], align 1, addrspace(5)
  %arraydecay = getelementptr inbounds [9 x i8], ptr addrspace(5) %str, i32 0, i32 0
  %call1 = call i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) @not.constant, ptr addrspace(5) %arraydecay, i32 %n)
  ret void
}

declare i32 @printf(ptr addrspace(4), ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "foo.cl", directory: ".")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocation(line: 1, column: 42, scope: !5)
!5 = distinct !DISubprogram(name: "arst", scope: null, file: !1, line: 1, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0)
