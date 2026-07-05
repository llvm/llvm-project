; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

%Foo = type { i32, i32 }

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)
declare void @init(ptr)
declare void @foo_p(ptr preallocated(%Foo))

; CHECK-LABEL: define void @preallocated_with_init() {
; INTERESTING: call void @foo_p

; RESULT: %t = call token @llvm.call.preallocated.setup(i32 1)
; RESULT-NEXT: call void @foo_p(ptr preallocated(%Foo) null) [ "preallocated"(token %t) ]
; RESULT-NEXT: ret void
define void @preallocated_with_init() {
  %t = call token @llvm.call.preallocated.setup(i32 1)
  %a = call ptr @llvm.call.preallocated.arg(token %t, i32 0) preallocated(%Foo)
  call void @init(ptr %a)
  call void @foo_p(ptr preallocated(%Foo) %a) ["preallocated"(token %t)]
  ret void
}
