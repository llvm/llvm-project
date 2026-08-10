; RUN: opt -S -passes=sroa %s | FileCheck %s
;
;; Check that we do not assert and that we retain the fake_use instruction that
;; uses the address of bar.
;
; CHECK: define{{.*}}foo
; CHECK: call{{.*llvm\.fake\.use.*}}(ptr %bar.addr)

define void @_Z3fooPi(ptr %bar) {
entry:
  %bar.addr = alloca ptr, align 8
  %baz = alloca ptr, align 8
  store ptr %bar, ptr %bar.addr, align 8
  store ptr %bar.addr, ptr %baz, align 8
  %0 = load ptr, ptr %bar.addr, align 8
  %1 = load ptr, ptr %baz, align 8
  call void (...) @llvm.fake.use(ptr %1)
  ret void
}

declare void @llvm.fake.use(...)
