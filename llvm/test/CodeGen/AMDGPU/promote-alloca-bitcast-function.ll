; RUN: not llc -march=amdgcn < %s 2>&1 | FileCheck %s

; Make sure that AMDGPUPromoteAlloca doesn't crash if the called
; function is a constantexpr cast of a function.

declare void @foo(float*) #0
declare void @foo.varargs(...) #0

; CHECK: in function crash_call_constexpr_cast{{.*}}: unsupported call to function foo
define void @crash_call_constexpr_cast() #0 {
  %alloca = alloca i32
  call void bitcast (void (float*)* @foo to void (i32*)*)(i32* %alloca) #0
  ret void
}

; CHECK: in function crash_call_constexpr_cast{{.*}}: unsupported call to function foo.varargs
define void @crash_call_constexpr_cast_varargs() #0 {
  %alloca = alloca i32
  call void bitcast (void (...)* @foo.varargs to void (i32*)*)(i32* %alloca) #0
  ret void
}

attributes #0 = { nounwind }
