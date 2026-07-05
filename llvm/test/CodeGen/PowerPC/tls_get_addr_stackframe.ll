; RUN: llc -verify-machineinstrs -mtriple="powerpc64le-unknown-linux-gnu" -relocation-model=pic < %s | FileCheck %s
; CHECK-LABEL: foo_test:
; CHECK: mflr 0
; CHECK: __tls_get_addr

%struct1.2.41 = type { %struct2.0.39, %struct3.1.40, ptr }
%struct2.0.39 = type { i64, i32, i32, i32, i32 }
%struct3.1.40 = type { [160 x i8] }

@tls_var = external thread_local global ptr, align 8

define i32 @foo_test() {
  %1 = load ptr, ptr @tls_var, align 8

  %2 = getelementptr inbounds %struct1.2.41, ptr %1, i64 0, i32 0, i32 3
  %3 = load i32, ptr %2, align 8
  %4 = add nsw i32 %3, -1
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %bb7, label %foo.exit

bb7:                                       ; preds = %3
  tail call void undef(ptr undef, ptr nonnull undef)
  br label %foo.exit

foo.exit:                                         ; preds = %8, %3, %2, %0
  ret i32 %4
}
