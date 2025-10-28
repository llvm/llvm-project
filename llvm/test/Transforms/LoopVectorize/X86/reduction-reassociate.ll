; Check that the loops with a floating-point reduction are vectorized
; according to llvm.loop.vectorize.reassociate_fpreductions.enable metadata.
; RUN: opt -passes=loop-vectorize -S < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

define float @test_enable(ptr readonly captures(none) %array, float %init) {
; CHECK-LABEL: define float @test_enable(
; CHECK:    fadd contract <4 x float> {{.*}}
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD0:[0-9]+]]
; CHECK:    call contract float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> {{.*}})
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD3:[0-9]+]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi float [ %init, %entry ], [ %red.next, %loop ]
  %gep = getelementptr float, ptr %array, i64 %iv
  %element = load float, ptr %gep, align 4
  %red.next = fadd contract float %red, %element
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  %result = phi float [ %red.next, %loop ]
  ret float %result
}

; The reduction is unsafe, and the metadata does not allow
; vectorizing it:
define float @test_disable(ptr readonly captures(none) %array, float %init) {
; CHECK-LABEL: define float @test_disable(
; CHECK-NOT:    <4 x float>
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD4:[0-9]+]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi float [ %init, %entry ], [ %red.next, %loop ]
  %gep = getelementptr float, ptr %array, i64 %iv
  %element = load float, ptr %gep, align 4
  %red.next = fadd contract float %red, %element
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !2

exit:
  %result = phi float [ %red.next, %loop ]
  ret float %result
}

; Forced vectorization "makes" the reduction reassociation safe,
; so setting llvm.loop.vectorize.reassociate_fpreductions.enable
; to false does not have effect:
define float @test_disable_with_forced_vectorization(ptr readonly captures(none) %array, float %init) {
; CHECK-LABEL: define float @test_disable_with_forced_vectorization(
; CHECK:    fadd contract <4 x float> {{.*}}
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD6:[0-9]+]]
; CHECK:    call contract float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> {{.*}})
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD7:[0-9]+]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi float [ %init, %entry ], [ %red.next, %loop ]
  %gep = getelementptr float, ptr %array, i64 %iv
  %element = load float, ptr %gep, align 4
  %red.next = fadd contract float %red, %element
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !4

exit:
  %result = phi float [ %red.next, %loop ]
  ret float %result
}

; 'fast' math makes reduction reassociation safe,
; so setting llvm.loop.vectorize.reassociate_fpreductions.enable
; to false does not have effect:
define float @test_disable_with_fast_math(ptr readonly captures(none) %array, float %init) {
; CHECK-LABEL: define float @test_disable_with_fast_math(
; CHECK:    fadd fast <4 x float> {{.*}}
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD8:[0-9]+]]
; CHECK:    call fast float @llvm.vector.reduce.fadd.v4f32(float 0.000000e+00, <4 x float> {{.*}})
; CHECK:    br i1 %{{.*}}, !llvm.loop ![[MD9:[0-9]+]]
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi float [ %init, %entry ], [ %red.next, %loop ]
  %gep = getelementptr float, ptr %array, i64 %iv
  %element = load float, ptr %gep, align 4
  %red.next = fadd fast float %red, %element
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1000
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !2

exit:
  %result = phi float [ %red.next, %loop ]
  ret float %result
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.reassociate_fpreductions.enable", i1 true}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.vectorize.reassociate_fpreductions.enable", i1 false}
!4 = distinct !{!4, !3, !5}
!5 = !{!"llvm.loop.vectorize.enable", i1 true}

; CHECK-NOT: llvm.loop.vectorize.reassociate_fpreductions.enable
; CHECK: ![[MD0]] = distinct !{![[MD0]], ![[MD1:[0-9]+]], ![[MD2:[0-9]+]]}
; CHECK: ![[MD1]] = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: ![[MD2]] = !{!"llvm.loop.unroll.runtime.disable"}
; CHECK: ![[MD3]] = distinct !{![[MD3]], ![[MD2]], ![[MD1]]}
; CHECK: ![[MD4]] = distinct !{![[MD4]], ![[MD5:[0-9]+]]}
; CHECK: ![[MD5]] = !{!"llvm.loop.vectorize.reassociate_fpreductions.enable", i1 false}
; CHECK: ![[MD6]] = distinct !{![[MD6]], ![[MD1]], ![[MD2]]}
; CHECK: ![[MD7]] = distinct !{![[MD7]], ![[MD2]], ![[MD1]]}
; CHECK: ![[MD8]] = distinct !{![[MD8]], ![[MD1]], ![[MD2]]}
; CHECK: ![[MD9]] = distinct !{![[MD9]], ![[MD2]], ![[MD1]]}
