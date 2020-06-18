; Check the optimizer doesn't crash at inlining the function top and all of its callees are inlined.
; RUN: opt < %s -O3 -S | FileCheck %s

define dso_local void (...)* @second(i8** %p) {
entry:
  %p.addr = alloca i8**, align 8
  store i8** %p, i8*** %p.addr, align 8
  %tmp = load i8**, i8*** %p.addr, align 8
  %tmp1 = load i8*, i8** %tmp, align 8
  %tmp2 = bitcast i8* %tmp1 to void (...)*
  ret void (...)* %tmp2
}

define dso_local void @top()  {
entry:
  ; CHECK: {{.*}} = {{.*}} call {{.*}} @ext
  ; CHECK-NOT: {{.*}} = {{.*}} call {{.*}} @third
  ; CHECK-NOT: {{.*}} = {{.*}} call {{.*}} @second
  ; CHECK-NOT: {{.*}} = {{.*}} call {{.*}} @wrapper
  %q = alloca i8*, align 8
  store i8* bitcast (void ()* @third to i8*), i8** %q, align 8
  %tmp = call void (...)* @second(i8** %q)
  ; The call to 'wrapper' here is to ensure that its function attributes
  ; i.e., returning its parameter and having no side effect, will be decuded
  ; before the next round of inlining happens to 'top' to expose the bug.
  %call =  call void (...)* @wrapper(void (...)* %tmp) 
  ; The indirect call here is to confuse the alias analyzer so that
  ; an incomplete graph will be built during the first round of inlining.
  ; This allows the current function to be processed before the actual 
  ; callee, i.e., the function 'run', is processed. Once it's simplified to 
  ; a direct call, it also enables an additional round of inlining with all
  ; function attributes deduced. 
  call void (...) %call()
  ret void
}

define dso_local void (...)* @gen() {
entry:
  %call = call void (...)* (...) @ext()
  ret void (...)* %call
}

declare dso_local void (...)* @ext(...) 

define dso_local void (...)* @wrapper(void (...)* %fn) {
entry:
  ret void (...)* %fn
}

define dso_local void @run(void (...)* %fn) {
entry:
  %fn.addr = alloca void (...)*, align 8
  %f = alloca void (...)*, align 8
  store void (...)* %fn, void (...)** %fn.addr, align 8
  %tmp = load void (...)*, void (...)** %fn.addr, align 8
  %call = call void (...)* @wrapper(void (...)* %tmp)
  store void (...)* %call, void (...)** %f, align 8
  %tmp1 = load void (...)*, void (...)** %f, align 8
  call void (...) %tmp1()
  ret void
}

define dso_local void @third() {
entry:
  %f = alloca void (...)*, align 8
  %call = call void (...)* @gen()
  store void (...)* %call, void (...)** %f, align 8
  %tmp = load void (...)*, void (...)** %f, align 8
  call void @run(void (...)* %tmp)
  ret void
}