; RUN: opt < %s -S -passes='cgscc(inline),function(early-cse),globalopt' | FileCheck %s

%0 = type { i32, ptr, ptr }
%struct.A = type { i8 }
%struct.B = type { }

@a = global %struct.A zeroinitializer, align 1
@__dso_handle = external global ptr
@llvm.global_ctors = appending global [1 x %0] [%0 { i32 65535, ptr @_GLOBAL__I_a, ptr null }]

; CHECK-NOT: call i32 @__cxa_atexit

define internal void @__cxx_global_var_init() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
  %1 = call i32 @__cxa_atexit(ptr @_ZN1AD1Ev, ptr @a, ptr @__dso_handle)
  ret void
}

define linkonce_odr void @_ZN1AD1Ev(ptr %this) nounwind align 2 {
  call void @_ZN1BD1Ev(ptr %this)
  ret void
}

declare i32 @__cxa_atexit(ptr, ptr, ptr)

define linkonce_odr void @_ZN1BD1Ev(ptr %this) nounwind align 2 {
  ret void
}

define internal void @_GLOBAL__I_a() nounwind section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}
