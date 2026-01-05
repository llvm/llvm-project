; Test that invalid reductions are not introduced by stripping the
; required appending linkage from intrinsic global variables.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=global-values --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefix=CHECK-FINAL %s < %t

; CHECK-INTERESTINGNESS: @llvm.global_ctors
; CHECK-INTERESTINGNESS: @llvm.global_dtors
; CHECK-INTERESTINGNESS: @llvm.used

; CHECK-INTERESTINGNESS: define
; CHECK-INTERESTINGNESS-SAME: void @f

; CHECK-FINAL: define void @f()


@llvm.global_ctors = appending global [2 x { i32, ptr, ptr  }] [{ i32, ptr, ptr  } { i32 1, ptr @f, ptr null  }, { i32, ptr, ptr  } { i32 1, ptr @g, ptr null  }]
@llvm.global_dtors = appending global [2 x { i32, ptr, ptr  }] [{ i32, ptr, ptr  } { i32 1, ptr @f, ptr null  }, { i32, ptr, ptr  } { i32 1, ptr @g, ptr null  }]
@llvm.used = appending global [1 x ptr] [ptr @h], section "llvm.metadata"

define void @f() {
  ret void
}

; CHECK-INTERESTINGNESS: define
; CHECK-INTERESTINGNESS-SAME: void @g

; CHECK-FINAL: define void @g()

define internal void @g() {
  ret void
}

; CHECK-INTERESTINGNESS: define
; CHECK-INTERESTINGNESS-SAME: void @h

define internal void @h() {
  ret void
}
