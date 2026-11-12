; Check the optimizer doesn't crash at inlining the function top and all of its callees are inlined.
; RUN: opt < %s -O3 -S | FileCheck %s

define dso_local ptr @second(ptr %p) {
entry:
  %p.addr = alloca ptr, align 8
  store ptr %p, ptr %p.addr, align 8
  %tmp = load ptr, ptr %p.addr, align 8
  %tmp1 = load ptr, ptr %tmp, align 8
  ret ptr %tmp1
}

define dso_local void @top()  {
entry:
  ; CHECK: {{.*}} = {{.*}} call {{.*}} @ext
  ; CHECK-NOT: {{.*}} = {{.*}} call {{.*}} @third
  ; CHECK-NOT: {{.*}} = {{.*}} call {{.*}} @second
  ; CHECK-NOT: {{.*}} = {{.*}} call {{.*}} @wrapper
  %q = alloca ptr, align 8
  store ptr @third, ptr %q, align 8
  %tmp = call ptr @second(ptr %q)
  ; The call to 'wrapper' here is to ensure that its function attributes
  ; i.e., returning its parameter and having no side effect, will be decuded
  ; before the next round of inlining happens to 'top' to expose the bug.
  %call =  call ptr @wrapper(ptr %tmp) 
  ; The indirect call here is to confuse the alias analyzer so that
  ; an incomplete graph will be built during the first round of inlining.
  ; This allows the current function to be processed before the actual 
  ; callee, i.e., the function 'run', is processed. Once it's simplified to 
  ; a direct call, it also enables an additional round of inlining with all
  ; function attributes deduced. 
  call void (...) %call()
  ret void
}

define dso_local ptr @gen() {
entry:
  %call = call ptr (...) @ext()
  ret ptr %call
}

declare dso_local ptr @ext(...) 

define dso_local ptr @wrapper(ptr %fn) {
entry:
  ret ptr %fn
}

define dso_local void @run(ptr %fn) {
entry:
  %fn.addr = alloca ptr, align 8
  %f = alloca ptr, align 8
  store ptr %fn, ptr %fn.addr, align 8
  %tmp = load ptr, ptr %fn.addr, align 8
  %call = call ptr @wrapper(ptr %tmp)
  store ptr %call, ptr %f, align 8
  %tmp1 = load ptr, ptr %f, align 8
  call void (...) %tmp1()
  ret void
}

define dso_local void @third() {
entry:
  %f = alloca ptr, align 8
  %call = call ptr @gen()
  store ptr %call, ptr %f, align 8
  %tmp = load ptr, ptr %f, align 8
  call void @run(ptr %tmp)
  ret void
}