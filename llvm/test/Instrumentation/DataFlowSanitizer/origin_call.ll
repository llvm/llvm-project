; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1  -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
define i1 @arg_overflow(
i1   %a0, i1   %a1, i1   %a2, i1   %a3, i1   %a4, i1   %a5, i1   %a6, i1   %a7, i1   %a8, i1   %a9,
i1  %a10, i1  %a11, i1  %a12, i1  %a13, i1  %a14, i1  %a15, i1  %a16, i1  %a17, i1  %a18, i1  %a19,
i1  %a20, i1  %a21, i1  %a22, i1  %a23, i1  %a24, i1  %a25, i1  %a26, i1  %a27, i1  %a28, i1  %a29,
i1  %a30, i1  %a31, i1  %a32, i1  %a33, i1  %a34, i1  %a35, i1  %a36, i1  %a37, i1  %a38, i1  %a39,
i1  %a40, i1  %a41, i1  %a42, i1  %a43, i1  %a44, i1  %a45, i1  %a46, i1  %a47, i1  %a48, i1  %a49,
i1  %a50, i1  %a51, i1  %a52, i1  %a53, i1  %a54, i1  %a55, i1  %a56, i1  %a57, i1  %a58, i1  %a59,
i1  %a60, i1  %a61, i1  %a62, i1  %a63, i1  %a64, i1  %a65, i1  %a66, i1  %a67, i1  %a68, i1  %a69,
i1  %a70, i1  %a71, i1  %a72, i1  %a73, i1  %a74, i1  %a75, i1  %a76, i1  %a77, i1  %a78, i1  %a79,
i1  %a80, i1  %a81, i1  %a82, i1  %a83, i1  %a84, i1  %a85, i1  %a86, i1  %a87, i1  %a88, i1  %a89,
i1  %a90, i1  %a91, i1  %a92, i1  %a93, i1  %a94, i1  %a95, i1  %a96, i1  %a97, i1  %a98, i1  %a99,
i1 %a100, i1 %a101, i1 %a102, i1 %a103, i1 %a104, i1 %a105, i1 %a106, i1 %a107, i1 %a108, i1 %a109,
i1 %a110, i1 %a111, i1 %a112, i1 %a113, i1 %a114, i1 %a115, i1 %a116, i1 %a117, i1 %a118, i1 %a119,
i1 %a120, i1 %a121, i1 %a122, i1 %a123, i1 %a124, i1 %a125, i1 %a126, i1 %a127, i1 %a128, i1 %a129,
i1 %a130, i1 %a131, i1 %a132, i1 %a133, i1 %a134, i1 %a135, i1 %a136, i1 %a137, i1 %a138, i1 %a139,
i1 %a140, i1 %a141, i1 %a142, i1 %a143, i1 %a144, i1 %a145, i1 %a146, i1 %a147, i1 %a148, i1 %a149,
i1 %a150, i1 %a151, i1 %a152, i1 %a153, i1 %a154, i1 %a155, i1 %a156, i1 %a157, i1 %a158, i1 %a159,
i1 %a160, i1 %a161, i1 %a162, i1 %a163, i1 %a164, i1 %a165, i1 %a166, i1 %a167, i1 %a168, i1 %a169,
i1 %a170, i1 %a171, i1 %a172, i1 %a173, i1 %a174, i1 %a175, i1 %a176, i1 %a177, i1 %a178, i1 %a179,
i1 %a180, i1 %a181, i1 %a182, i1 %a183, i1 %a184, i1 %a185, i1 %a186, i1 %a187, i1 %a188, i1 %a189,
i1 %a190, i1 %a191, i1 %a192, i1 %a193, i1 %a194, i1 %a195, i1 %a196, i1 %a197, i1 %a198, i1 %a199,
i1 %a200
) {
  ; CHECK: @arg_overflow.dfsan
  ; CHECK: [[A199:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 199), align 4
  ; CHECK: store i32 [[A199]], ptr @__dfsan_retval_origin_tls, align 4

  %r = add i1 %a199, %a200
  ret i1 %r
}

define i1 @param_overflow(i1 %a) {
  ; CHECK: @param_overflow.dfsan
  ; CHECK: store i32 %1, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 199), align 4
  ; CHECK-NEXT: store i8 %2, ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 398), align 2
  ; CHECK-NEXT: store i8 %2, ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 400), align 2
  ; CHECK-NEXT: %r = call i1 @arg_overflow.dfsan
  ; CHECK: %_dfsret_o = load i32, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK: store i32 %_dfsret_o, ptr @__dfsan_retval_origin_tls, align 4

  %r = call i1 @arg_overflow(
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a, i1 %a,
i1 %a
)
  ret i1 %r
}

declare void @foo(i1 %a)

define void @param_with_zero_shadow() {
  ; CHECK: @param_with_zero_shadow.dfsan
  ; CHECK-NEXT: store i8 0, ptr @__dfsan_arg_tls, align 2
  ; CHECK-NEXT: call void @foo.dfsan(i1 true)

  call void @foo(i1 1)
  ret void
}
