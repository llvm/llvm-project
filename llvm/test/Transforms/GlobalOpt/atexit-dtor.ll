; RUN: opt < %s -S -passes='cgscc(inline),function(early-cse),globalopt' | FileCheck %s

%struct.A = type { i32 }

$"??1A@@QEAA@XZ" = comdat any

@"?g@@3UA@@A" = dso_local global %struct.A zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_atexit-dtor, ptr null }]

; CHECK-NOT: call i32 @atexit

define internal void @"??__Eg@@YAXXZ"() #0 {
  %1 = call i32 @atexit(ptr @"??__Fg@@YAXXZ") #2
  ret void
}

define linkonce_odr dso_local void @"??1A@@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %0) unnamed_addr #1 comdat align 2 {
  ret void
}

define internal void @"??__Fg@@YAXXZ"() #0 {
  call void @"??1A@@QEAA@XZ"(ptr @"?g@@3UA@@A")
  ret void
}

declare dso_local i32 @atexit(ptr) #2

define internal void @_GLOBAL__sub_I_atexit-dtor() #0 {
  call void @"??__Eg@@YAXXZ"()
  ret void
}
