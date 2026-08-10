; Adapted from Khronos Translator:
; https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/test/type-scavenger/equivalence.ll
; The goal of the test is to ensure that the Backend doesn't crash during
; the 'finalize lowering' stage on management of function forward calls.

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-COUNT-9: OpFunction

define spir_func void @_func1() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3 = call spir_func ptr addrspace(4) @_func2()
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3, i64 0)
  br label %for.cond
}

define spir_func void @_func3() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3 = call spir_func ptr @_func4()
  %call3.ascast = addrspacecast ptr %call3 to ptr addrspace(4)
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3.ascast, i64 0)
  br label %for.cond
}

declare spir_func ptr addrspace(4) @_func5()

define spir_func void @_func6(ptr addrspace(4) %call3.ascast) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3.ascast, i64 0)
  br label %for.cond
}

define spir_func void @_func7() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call3 = call spir_func ptr addrspace(4) @_func5()
  %call5 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %call3, i64 0)
  br label %for.cond
}

declare spir_func ptr @_func4()

declare spir_func ptr addrspace(4) @_func2()

define spir_func ptr addrspace(4) @_func0(ptr addrspace(4) %this, i64 %index) {
entry:
  %arrayidx = getelementptr [5 x i32], ptr addrspace(4) %this, i64 0, i64 %index
  ret ptr addrspace(4) null
}

define spir_func void @_func8() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %call8 = call spir_func ptr addrspace(4) @_func0(ptr addrspace(4) null, i64 0)
  br label %for.cond
}

uselistorder ptr @_func0, { 0, 4, 3, 2, 1 }
