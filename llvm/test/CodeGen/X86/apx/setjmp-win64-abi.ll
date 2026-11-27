; RUN: llc < %s -mtriple=x86_64-windows-msvc -mattr=+egpr -regalloc=greedy | FileCheck %s --check-prefixes=CHECK,ALLOC
; RUN: llc < %s -mtriple=x86_64-windows-msvc -mattr=+egpr -regalloc=basic | FileCheck %s --check-prefixes=CHECK,ALLOC
; RUN: llc < %s -mtriple=x86_64-windows-msvc -mattr=+egpr -regalloc=pbqp | FileCheck %s --check-prefixes=CHECK,ALLOC
; RUN: llc < %s -mtriple=x86_64-windows-msvc -mattr=+egpr -x86-setjmp-csr-warning-threshold=1 2>&1 | FileCheck %s --check-prefix=WARN
;
; Test that warnings are emitted exactly once per setjmp function.
; WARN: warning: {{.*}}callee-saved register(s) reserved due to setjmp in 'test_setjmp_i64'
; WARN-NOT: warning: {{.*}}callee-saved register(s) reserved due to setjmp in 'test_setjmp_i64'
; WARN: warning: {{.*}}callee-saved register(s) reserved due to setjmp in 'test_setjmp_i32'
; WARN-NOT: warning: {{.*}}callee-saved register(s) reserved due to setjmp in 'test_setjmp_i32'

; Test that R30 and R31 and all their sub-registers (r30d, r31d) are not
; allocated in functions that call setjmp (returns_twice) with Win64 APX
; ABI support.
;
; Generated from the following C++ code:
;
;   #include <cstdint>
;   #include <csetjmp>
;
;   extern "C" jmp_buf jmpbuf;
;   extern "C" void external_call();
;
;   template<typename T> T get_val();
;   template<typename T> void use_val(T val);
;
;   template<typename T>
;   T test_setjmp() {
;     T v1 = get_val<T>();
;     T v2 = get_val<T>();
;     T v3 = get_val<T>();
;     T v4 = get_val<T>();
;     T v5 = get_val<T>();
;     T v6 = get_val<T>();
;     T v7 = get_val<T>();
;     T v8 = get_val<T>();
;     T v9 = get_val<T>();
;     T v10 = get_val<T>();
;     T v11 = get_val<T>();
;     T v12 = get_val<T>();
;
;     if (_setjmp(jmpbuf) == 0) {
;       external_call();
;       use_val(v1);
;       use_val(v2);
;       use_val(v3);
;       use_val(v4);
;       use_val(v5);
;       use_val(v6);
;       use_val(v7);
;       use_val(v8);
;       use_val(v9);
;       use_val(v10);
;       use_val(v11);
;       use_val(v12);
;     }
;     return v1;
;   }
;
;   template uint32_t test_setjmp<uint32_t>();
;   template uint64_t test_setjmp<uint64_t>();

declare i32 @_setjmp(ptr) returns_twice
declare i32 @_fake_setjmp(ptr)
declare void @external_call()
declare i64 @get_val_i64()
declare void @use_val_i64(i64)
declare i32 @get_val_i32()
declare void @use_val_i32(i32)

@buf = external global [256 x i8], align 16

; Without returns_twice, r30/r31 SHOULD be used.
define i64 @test_no_setjmp_i64() nounwind {
; CHECK-LABEL: test_no_setjmp_i64:
; ALLOC:       %r30
; ALLOC:       %r31
; CHECK:       retq
entry:
  %v1 = call i64 @get_val_i64()
  %v2 = call i64 @get_val_i64()
  %v3 = call i64 @get_val_i64()
  %v4 = call i64 @get_val_i64()
  %v5 = call i64 @get_val_i64()
  %v6 = call i64 @get_val_i64()
  %v7 = call i64 @get_val_i64()
  %v8 = call i64 @get_val_i64()
  %v9 = call i64 @get_val_i64()
  %v10 = call i64 @get_val_i64()
  %v11 = call i64 @get_val_i64()
  %v12 = call i64 @get_val_i64()
  %r = call i32 @_fake_setjmp(ptr @buf)
  %cmp = icmp eq i32 %r, 0
  br i1 %cmp, label %normal, label %longjmp_return

normal:
  call void @external_call()
  call void @use_val_i64(i64 %v1)
  call void @use_val_i64(i64 %v2)
  call void @use_val_i64(i64 %v3)
  call void @use_val_i64(i64 %v4)
  call void @use_val_i64(i64 %v5)
  call void @use_val_i64(i64 %v6)
  call void @use_val_i64(i64 %v7)
  call void @use_val_i64(i64 %v8)
  call void @use_val_i64(i64 %v9)
  call void @use_val_i64(i64 %v10)
  call void @use_val_i64(i64 %v11)
  call void @use_val_i64(i64 %v12)
  br label %longjmp_return

longjmp_return:
  ret i64 %v1
}

define i32 @test_no_setjmp_i32() nounwind {
; CHECK-LABEL: test_no_setjmp_i32:
; ALLOC:       %r30d
; ALLOC:       %r31d
; CHECK:       retq
entry:
  %v1 = call i32 @get_val_i32()
  %v2 = call i32 @get_val_i32()
  %v3 = call i32 @get_val_i32()
  %v4 = call i32 @get_val_i32()
  %v5 = call i32 @get_val_i32()
  %v6 = call i32 @get_val_i32()
  %v7 = call i32 @get_val_i32()
  %v8 = call i32 @get_val_i32()
  %v9 = call i32 @get_val_i32()
  %v10 = call i32 @get_val_i32()
  %v11 = call i32 @get_val_i32()
  %v12 = call i32 @get_val_i32()
  %r = call i32 @_fake_setjmp(ptr @buf)
  %cmp = icmp eq i32 %r, 0
  br i1 %cmp, label %normal, label %longjmp_return

normal:
  call void @external_call()
  call void @use_val_i32(i32 %v1)
  call void @use_val_i32(i32 %v2)
  call void @use_val_i32(i32 %v3)
  call void @use_val_i32(i32 %v4)
  call void @use_val_i32(i32 %v5)
  call void @use_val_i32(i32 %v6)
  call void @use_val_i32(i32 %v7)
  call void @use_val_i32(i32 %v8)
  call void @use_val_i32(i32 %v9)
  call void @use_val_i32(i32 %v10)
  call void @use_val_i32(i32 %v11)
  call void @use_val_i32(i32 %v12)
  br label %longjmp_return

longjmp_return:
  ret i32 %v1
}

; Without returns_twice, r30/r31 must NOT be used.
define i64 @test_setjmp_i64() nounwind {
; CHECK-LABEL: test_setjmp_i64:
; CHECK-NOT:   r30
; CHECK-NOT:   r31
; CHECK:       retq
entry:
  %v1 = call i64 @get_val_i64()
  %v2 = call i64 @get_val_i64()
  %v3 = call i64 @get_val_i64()
  %v4 = call i64 @get_val_i64()
  %v5 = call i64 @get_val_i64()
  %v6 = call i64 @get_val_i64()
  %v7 = call i64 @get_val_i64()
  %v8 = call i64 @get_val_i64()
  %v9 = call i64 @get_val_i64()
  %v10 = call i64 @get_val_i64()
  %v11 = call i64 @get_val_i64()
  %v12 = call i64 @get_val_i64()
  %r = call i32 @_setjmp(ptr @buf) returns_twice
  %cmp = icmp eq i32 %r, 0
  br i1 %cmp, label %normal, label %longjmp_return

normal:
  call void @external_call()
  call void @use_val_i64(i64 %v1)
  call void @use_val_i64(i64 %v2)
  call void @use_val_i64(i64 %v3)
  call void @use_val_i64(i64 %v4)
  call void @use_val_i64(i64 %v5)
  call void @use_val_i64(i64 %v6)
  call void @use_val_i64(i64 %v7)
  call void @use_val_i64(i64 %v8)
  call void @use_val_i64(i64 %v9)
  call void @use_val_i64(i64 %v10)
  call void @use_val_i64(i64 %v11)
  call void @use_val_i64(i64 %v12)
  br label %longjmp_return

longjmp_return:
  ret i64 %v1
}

define i32 @test_setjmp_i32() nounwind {
; CHECK-LABEL: test_setjmp_i32:
; CHECK-NOT:   r30d
; CHECK-NOT:   r31d
; CHECK:       retq
entry:
  %v1 = call i32 @get_val_i32()
  %v2 = call i32 @get_val_i32()
  %v3 = call i32 @get_val_i32()
  %v4 = call i32 @get_val_i32()
  %v5 = call i32 @get_val_i32()
  %v6 = call i32 @get_val_i32()
  %v7 = call i32 @get_val_i32()
  %v8 = call i32 @get_val_i32()
  %v9 = call i32 @get_val_i32()
  %v10 = call i32 @get_val_i32()
  %v11 = call i32 @get_val_i32()
  %v12 = call i32 @get_val_i32()
  %r = call i32 @_setjmp(ptr @buf) returns_twice
  %cmp = icmp eq i32 %r, 0
  br i1 %cmp, label %normal, label %longjmp_return

normal:
  call void @external_call()
  call void @use_val_i32(i32 %v1)
  call void @use_val_i32(i32 %v2)
  call void @use_val_i32(i32 %v3)
  call void @use_val_i32(i32 %v4)
  call void @use_val_i32(i32 %v5)
  call void @use_val_i32(i32 %v6)
  call void @use_val_i32(i32 %v7)
  call void @use_val_i32(i32 %v8)
  call void @use_val_i32(i32 %v9)
  call void @use_val_i32(i32 %v10)
  call void @use_val_i32(i32 %v11)
  call void @use_val_i32(i32 %v12)
  br label %longjmp_return

longjmp_return:
  ret i32 %v1
}
