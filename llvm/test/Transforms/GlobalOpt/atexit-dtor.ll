; RUN: opt < %s -S -passes='cgscc(inline),function(early-cse),globalopt' | FileCheck %s

%struct.A = type { i32 }

$"??1A@@QEAA@XZ" = comdat any

@"?g@@3UA@@A" = dso_local global %struct.A zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_atexit-dtor, ptr null }]

; CHECK-NOT: call i32 @atexit

define internal void @"??__Eg@@YAXXZ"() {
  %1 = call i32 @atexit(ptr @"??__Fg@@YAXXZ")
  ret void
}

define linkonce_odr dso_local void @"??1A@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %0) unnamed_addr #1 comdat align 2 {
  ret void
}

define internal void @"??__Fg@@YAXXZ"() {
  call void @"??1A@@QEAA@XZ"(ptr @"?g@@3UA@@A")
  ret void
}

declare dso_local i32 @atexit(ptr)

define internal void @_GLOBAL__sub_I_atexit-dtor() {
  call void @"??__Eg@@YAXXZ"()
  ret void
}

define dso_local void @atexit_handler() {
  ret void
}

; CHECK-NOT: call i32 @atexit

; Check that a removed `atexit` call returns `0` which is the value that denotes success.
define dso_local noundef i32 @register_atexit_handler() {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  %2 = call i32 @atexit(ptr @"atexit_handler")
; CHECK: ret i32 0
  ret i32 %2
}
