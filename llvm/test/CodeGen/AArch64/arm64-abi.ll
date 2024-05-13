; RUN: llc     -mtriple=arm64-apple-darwin -mcpu=cyclone -enable-misched=false < %s | FileCheck %s
; RUN: llc -O0 -fast-isel -mtriple=arm64-apple-darwin                          < %s | FileCheck --check-prefix=FAST %s
; RUN: llc -global-isel -mtriple=arm64-apple-darwin -verify-machineinstrs < %s | FileCheck %s --check-prefix=GISEL

; rdar://9932559
define i64 @i8i16callee(i64 %a1, i64 %a2, i64 %a3, i8 signext %a4, i16 signext %a5, i64 %a6, i64 %a7, i64 %a8, i8 signext %b1, i16 signext %b2, i8 signext %b3, i8 signext %b4) nounwind readnone noinline {
entry:
; CHECK-LABEL: i8i16callee:
; The 8th, 9th, 10th and 11th arguments are passed at sp, sp+2, sp+4, sp+5.
; They are i8, i16, i8 and i8.
; CHECK-DAG: ldrsb {{w[0-9]+}}, [sp, #5]
; CHECK-DAG: ldrsb {{w[0-9]+}}, [sp, #4]
; CHECK-DAG: ldrsh {{w[0-9]+}}, [sp, #2]
; CHECK-DAG: ldrsb {{w[0-9]+}}, [sp]
; FAST-LABEL: i8i16callee:
; FAST-DAG: ldrsb  {{w[0-9]+}}, [sp, #5]
; FAST-DAG: ldrsb  {{w[0-9]+}}, [sp, #4]
; FAST-DAG: ldrsh  {{w[0-9]+}}, [sp, #2]
; FAST-DAG: ldrsb  {{w[0-9]+}}, [sp]
; GISEL-LABEL: i8i16callee:
; GISEL-DAG: ldrsb  {{w[0-9]+}}, [sp, #5]
; GISEL-DAG: ldrsb  {{w[0-9]+}}, [sp, #4]
; GISEL-DAG: ldrsh  {{w[0-9]+}}, [sp, #2]
; GISEL-DAG: ldrsb  {{w[0-9]+}}, [sp]
  %conv = sext i8 %a4 to i64
  %conv3 = sext i16 %a5 to i64
  %conv8 = sext i8 %b1 to i64
  %conv9 = sext i16 %b2 to i64
  %conv11 = sext i8 %b3 to i64
  %conv13 = sext i8 %b4 to i64
  %add10 = add i64 %a2, %a1
  %add12 = add i64 %add10, %a3
  %add14 = add i64 %add12, %conv
  %add = add i64 %add14, %conv3
  %add1 = add i64 %add, %a6
  %add2 = add i64 %add1, %a7
  %add4 = add i64 %add2, %a8
  %add5 = add i64 %add4, %conv8
  %add6 = add i64 %add5, %conv9
  %add7 = add i64 %add6, %conv11
  %add15 = add i64 %add7, %conv13
  %sext = shl i64 %add15, 32
  %conv17 = ashr exact i64 %sext, 32
  ret i64 %conv17
}

define i32 @i8i16caller() nounwind readnone {
entry:
; CHECK-LABEL: i8i16caller
; The 8th, 9th, 10th and 11th arguments are passed at sp, sp+2, sp+4, sp+5.
; They are i8, i16, i8 and i8.
; CHECK-DAG: stur {{w[0-9]+}}, [sp, #2]
; CHECK-DAG: strb {{w[0-9]+}}, [sp]
; CHECK: bl
; FAST-LABEL: i8i16caller
; FAST: strb {{w[0-9]+}}, [sp]
; FAST: strh {{w[0-9]+}}, [sp, #2]
; FAST: strb {{w[0-9]+}}, [sp, #4]
; FAST: strb {{w[0-9]+}}, [sp, #5]
; FAST: bl
  %call = tail call i64 @i8i16callee(i64 0, i64 1, i64 2, i8 signext 3, i16 signext 4, i64 5, i64 6, i64 7, i8 signext 97, i16 signext 98, i8 signext 99, i8 signext 100)
  %conv = trunc i64 %call to i32
  ret i32 %conv
}

; rdar://12651543
define double @circle_center([2 x float] %a) nounwind ssp {
  %call = tail call double @ext([2 x float] %a) nounwind
; CHECK-LABEL: circle_center
; CHECK: bl
  ret double %call
}
declare double @ext([2 x float])

; rdar://12656141
; 16-byte vector should be aligned at 16-byte when passing on stack.
; A double argument will be passed on stack, so vecotr should be at sp+16.
define double @fixed_4i(ptr nocapture %in) nounwind {
entry:
; CHECK-LABEL: fixed_4i
; CHECK: str [[REG_1:q[0-9]+]], [sp, #16]
; FAST-LABEL: fixed_4i
; FAST: sub sp, sp
; FAST: mov x[[ADDR:[0-9]+]], sp
; FAST: str [[REG_1:q[0-9]+]], [x[[ADDR]], #16]
  %0 = load <4 x i32>, ptr %in, align 16
  %call = tail call double @args_vec_4i(double 3.000000e+00, <4 x i32> %0, <4 x i32> %0, <4 x i32> %0, <4 x i32> %0, <4 x i32> %0, <4 x i32> %0, <4 x i32> %0, double 3.000000e+00, <4 x i32> %0, i8 signext 3)
  ret double %call
}
declare double @args_vec_4i(double, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, double, <4 x i32>, i8 signext)

; rdar://12695237
; d8 at sp, i in register w0.
@g_d = common global double 0.000000e+00, align 8
define void @test1(float %f1, double %d1, double %d2, double %d3, double %d4,
       double %d5, double %d6, double %d7, double %d8, i32 %i) nounwind ssp {
entry:
; CHECK-LABEL: test1
; CHECK: ldr [[REG_1:d[0-9]+]], [sp]
; CHECK: scvtf [[REG_2:s[0-9]+]], w0
; CHECK: fadd s0, [[REG_2]], s0
  %conv = sitofp i32 %i to float
  %add = fadd float %conv, %f1
  %conv1 = fpext float %add to double
  %add2 = fadd double %conv1, %d7
  %add3 = fadd double %add2, %d8
  store double %add3, ptr @g_d, align 8
  ret void
}

; i9 at sp, d1 in register s0.
define void @test2(i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6,
            i32 %i7, i32 %i8, i32 %i9, float %d1) nounwind ssp {
entry:
; CHECK-LABEL: test2
; CHECK: scvtf [[REG_2:s[0-9]+]], w0
; CHECK: fadd s0, [[REG_2]], s0
; CHECK: ldr [[REG_1:s[0-9]+]], [sp]
  %conv = sitofp i32 %i1 to float
  %add = fadd float %conv, %d1
  %conv1 = fpext float %add to double
  %conv2 = sitofp i32 %i8 to double
  %add3 = fadd double %conv2, %conv1
  %conv4 = sitofp i32 %i9 to double
  %add5 = fadd double %conv4, %add3
  store double %add5, ptr @g_d, align 8
  ret void
}

; rdar://12648441
; Check alignment on stack for v64, f64, i64, f32, i32.
define double @test3(ptr nocapture %in) nounwind {
entry:
; CHECK-LABEL: test3
; CHECK: str [[REG_1:d[0-9]+]], [sp, #8]
; FAST-LABEL: test3
; FAST: sub sp, sp, #{{[0-9]+}}
; FAST: mov x[[ADDR:[0-9]+]], sp
; FAST: str [[REG_1:d[0-9]+]], [x[[ADDR]], #8]
  %0 = load <2 x i32>, ptr %in, align 8
  %call = tail call double @args_vec_2i(double 3.000000e+00, <2 x i32> %0,
          <2 x i32> %0, <2 x i32> %0, <2 x i32> %0, <2 x i32> %0, <2 x i32> %0,
          <2 x i32> %0, float 3.000000e+00, <2 x i32> %0, i8 signext 3)
  ret double %call
}
declare double @args_vec_2i(double, <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32>,
               <2 x i32>, <2 x i32>, <2 x i32>, float, <2 x i32>, i8 signext)

define double @test4(ptr nocapture %in) nounwind {
entry:
; CHECK-LABEL: test4
; CHECK: str [[REG_1:d[0-9]+]], [sp, #8]
; CHECK: str [[REG_2:w[0-9]+]], [sp]
; CHECK: mov w0, #3
  %0 = load double, ptr %in, align 8
  %call = tail call double @args_f64(double 3.000000e+00, double %0, double %0,
          double %0, double %0, double %0, double %0, double %0,
          float 3.000000e+00, double %0, i8 signext 3)
  ret double %call
}
declare double @args_f64(double, double, double, double, double, double, double,
               double, float, double, i8 signext)

define i64 @test5(ptr nocapture %in) nounwind {
entry:
; CHECK-LABEL: test5
; CHECK: strb [[REG_3:w[0-9]+]], [sp, #16]
; CHECK: str [[REG_1:x[0-9]+]], [sp, #8]
; CHECK: str [[REG_2:w[0-9]+]], [sp]
  %0 = load i64, ptr %in, align 8
  %call = tail call i64 @args_i64(i64 3, i64 %0, i64 %0, i64 %0, i64 %0, i64 %0,
                         i64 %0, i64 %0, i32 3, i64 %0, i8 signext 3)
  ret i64 %call
}
declare i64 @args_i64(i64, i64, i64, i64, i64, i64, i64, i64, i32, i64,
             i8 signext)

define i32 @test6(ptr nocapture %in) nounwind {
entry:
; CHECK-LABEL: test6
; CHECK: strb [[REG_2:w[0-9]+]], [sp, #8]
; CHECK: str [[REG_1:s[0-9]+]], [sp, #4]
; CHECK: strh [[REG_3:w[0-9]+]], [sp]
  %0 = load float, ptr %in, align 4
  %call = tail call i32 @args_f32(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6,
          i32 7, i32 8, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0,
          float 6.0, float 7.0, float 8.0, i16 signext 3, float %0,
          i8 signext 3)
  ret i32 %call
}
declare i32 @args_f32(i32, i32, i32, i32, i32, i32, i32, i32,
                      float, float, float, float, float, float, float, float,
                      i16 signext, float, i8 signext)

define i32 @test7(ptr nocapture %in) nounwind {
entry:
; CHECK-LABEL: test7
; CHECK: strb [[REG_2:w[0-9]+]], [sp, #8]
; CHECK: str [[REG_1:w[0-9]+]], [sp, #4]
; CHECK: strh [[REG_3:w[0-9]+]], [sp]
  %0 = load i32, ptr %in, align 4
  %call = tail call i32 @args_i32(i32 3, i32 %0, i32 %0, i32 %0, i32 %0, i32 %0,
                         i32 %0, i32 %0, i16 signext 3, i32 %0, i8 signext 4)
  ret i32 %call
}
declare i32 @args_i32(i32, i32, i32, i32, i32, i32, i32, i32, i16 signext, i32,
             i8 signext)

define i32 @test8(i32 %argc, ptr nocapture %argv) nounwind {
entry:
; CHECK-LABEL: test8
; CHECK: str w8, [sp]
; CHECK: bl
; FAST-LABEL: test8
; FAST: strb {{w[0-9]+}}, [sp]
; FAST: strb {{w[0-9]+}}, [sp, #1]
; FAST: strb {{w[0-9]+}}, [sp, #2]
; FAST: strb {{w[0-9]+}}, [sp, #3]
; FAST: bl
  tail call void @args_i1(i1 zeroext false, i1 zeroext true, i1 zeroext false,
                  i1 zeroext true, i1 zeroext false, i1 zeroext true,
                  i1 zeroext false, i1 zeroext true, i1 zeroext false,
                  i1 zeroext true, i1 zeroext false, i1 zeroext true)
  ret i32 0
}

declare void @args_i1(i1 zeroext, i1 zeroext, i1 zeroext, i1 zeroext,
                      i1 zeroext, i1 zeroext, i1 zeroext, i1 zeroext,
                      i1 zeroext, i1 zeroext, i1 zeroext, i1 zeroext)

define i32 @i1_stack_incoming(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f,
                               i64 %g, i64 %h, i64 %i, i1 zeroext %j) {
; CHECK-LABEL: i1_stack_incoming:
; CHECK: ldrb w0, [sp, #8]
; CHECK: ret
; GISEL-LABEL: i1_stack_incoming:
; GISEL: ldrb w0, [sp, #8]
; GISEL: ret
  %v = zext i1 %j to i32
  ret i32 %v
}
