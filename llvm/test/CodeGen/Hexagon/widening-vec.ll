; RUN: llc -march=hexagon -mv73 -mhvx -mattr=+hvx-length128b < %s
; REQUIRES: asserts

; This test checks for an assert. It happens when we attempt to generate widening vector instructions for vector length that isn't not a multiple of HW vector size (1024).

; Function Attrs: nofree norecurse nounwind
define dllexport i32 @foo(ptr noalias nocapture %0, ptr noalias nocapture readonly %1, ptr noalias nocapture readonly %2) local_unnamed_addr {
entry:
  %3 = load <121 x i8>, ptr %2, align 1
  %4 = zext <121 x i8> %3 to <121 x i32>
  %5 = mul nuw nsw <121 x i32> %4, <i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22, i32 22>
  %6 = load <121 x i8>, ptr %1, align 1
  %7 = zext <121 x i8> %6 to <121 x i32>
  %8 = mul nuw nsw <121 x i32> %7, <i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96, i32 96>
  %9 = add nsw <121 x i32> %8, <i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648, i32 -9648>
  %10 = add nsw <121 x i32> %9, %5
  store <121 x i32> %10, ptr %0, align 4
  ret i32 0
}

; The tests below check lowering of add, sub, mul when inputs are extended from 8 to 32 bits.

; CHECK-LABEL: test_vadd1
; CHECK: v{{.*}}.h = vadd(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)

; Function Attrs: norecurse nounwind
define dso_local void @test_vadd1(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %wide.load19 = load <128 x i8>, ptr %b, align 1
  %1 = zext <128 x i8> %wide.load19 to <128 x i32>
  %2 = add nuw nsw <128 x i32> %1, %0
  store <128 x i32> %2, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vsub1
; CHECK: v{{.*}}.h = vsub(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)

; Function Attrs: norecurse nounwind
define dso_local void @test_vsub1(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %wide.load19 = load <128 x i8>, ptr %b, align 1
  %1 = zext <128 x i8> %wide.load19 to <128 x i32>
  %2 = sub nuw nsw <128 x i32> %1, %0
  store <128 x i32> %2, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy1
; CHECK: v{{.*}}.uh = vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.ub)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy1(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = zext <128 x i8> %wide.load to <128 x i32>
  %wide.load19 = load <128 x i8>, ptr %b, align 1
  %1 = zext <128 x i8> %wide.load19 to <128 x i32>
  %2 = mul nuw nsw <128 x i32> %1, %0
  store <128 x i32> %2, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy4
; CHECK: v{{[0-9:]+}}.h = vmpy(v{{[0-9]+}}.b,v{{[0-9]+}}.b)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy4(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = sext <128 x i8> %wide.load to <128 x i32>
  %wide.load19 = load <128 x i8>, ptr %b, align 1
  %1 = sext <128 x i8> %wide.load19 to <128 x i32>
  %2 = mul nuw nsw <128 x i32> %1, %0
  store <128 x i32> %2, ptr %r, align 4
  ret void
}

; CHECK-LABEL: test_vmpy7
; CHECK: v{{[0-9:]+}}.h = vmpy(v{{[0-9]+}}.ub,v{{[0-9]+}}.b)

; Function Attrs: norecurse nounwind
define dso_local void @test_vmpy7(ptr nocapture readonly %a, ptr nocapture readonly %b, ptr nocapture %r) local_unnamed_addr {
entry:
  %wide.load = load <128 x i8>, ptr %a, align 1
  %0 = sext <128 x i8> %wide.load to <128 x i32>
  %wide.load19 = load <128 x i8>, ptr %b, align 1
  %1 = zext <128 x i8> %wide.load19 to <128 x i32>
  %2 = mul nuw nsw <128 x i32> %1, %0
  store <128 x i32> %2, ptr %r, align 4
  ret void
}
