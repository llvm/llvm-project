; RUN: llc -mtriple="powerpc64le-unknown-linux-gnu" -relocation-model=pic < %s | FileCheck %s
; CHECK-LABEL: foo_test:
; CHECK: mflr 0
; CHECK: __tls_get_addr

%struct1.2.41 = type { %struct2.0.39, %struct3.1.40, %struct1.2.41* }
%struct2.0.39 = type { i64, i32, i32, i32, i32 }
%struct3.1.40 = type { [160 x i8] }

@tls_var = external thread_local global %struct1.2.41*, align 8

define void @foo_test() {
  %1 = load %struct1.2.41*, %struct1.2.41** @tls_var, align 8
  br i1 undef, label %foo.exit, label %2

; <label>:2                                       ; preds = %0
  br i1 undef, label %foo.exit, label %3

; <label>:3                                       ; preds = %2
  %4 = getelementptr inbounds %struct1.2.41, %struct1.2.41* %1, i64 0, i32 0, i32 3
  %5 = load i32, i32* %4, align 8
  %6 = add nsw i32 %5, -1
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %foo.exit

; <label>:8                                       ; preds = %3
  tail call void undef(%struct1.2.41* undef, %struct1.2.41* nonnull undef)
  br label %foo.exit

foo.exit:                                         ; preds = %8, %3, %2, %0
  ret void
}
