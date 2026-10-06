; RUN: llubi --vscale=1 --verbose < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,VS1
; RUN: llubi --vscale=2 --verbose < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,VS2
; RUN: llubi --vscale=4 --verbose < %s 2>&1 | FileCheck %s --check-prefixes=CHECK,VS4

define void @main() {
  %scale = call i32 @llvm.vscale.i32()
  %length = mul i32 %scale, 4
  %div = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32(<vscale x 4 x i32> splat(i32 8), <vscale x 4 x i32> splat(i32 2), <vscale x 4 x i1> splat(i1 true), i32 %scale)
  %first = extractelement <vscale x 4 x i32> %div, i32 0
  %tail = extractelement <vscale x 4 x i32> %div, i32 %scale
  %add = call i32 @llvm.vp.reduce.add.nxv4i32(i32 3, <vscale x 4 x i32> splat(i32 2), <vscale x 4 x i1> splat(i1 true), i32 %length)
  %fp = call double @llvm.vp.reduce.fadd.nxv4f64(double 1.0, <vscale x 4 x double> splat(double 2.0), <vscale x 4 x i1> splat(i1 true), i32 3)
  %indices = call <vscale x 4 x i32> @llvm.stepvector.nxv4i32()
  %cttz = call i64 @llvm.vp.cttz.elts.i64.nxv4i32(<vscale x 4 x i32> %indices, i1 false, <vscale x 4 x i1> splat(i1 true), i32 %length)
  %zeros = call i32 @llvm.vp.cttz.elts.i32.nxv4i32(<vscale x 4 x i32> poison, i1 false, <vscale x 4 x i1> zeroinitializer, i32 %length)
  %count_matches = icmp eq i32 %zeros, %length
  %narrow = call i1 @llvm.vp.cttz.elts.i1.nxv4i32(<vscale x 4 x i32> zeroinitializer, i1 false, <vscale x 4 x i1> zeroinitializer, i32 0)
  %boundary = call i4 @llvm.vp.cttz.elts.i4.nxv4i32(<vscale x 4 x i32> zeroinitializer, i1 false, <vscale x 4 x i1> splat(i1 true), i32 %length)
  ret void
}

; CHECK-LABEL: Entering function: main
; VS1: %scale = call i32 @llvm.vscale.i32() => i32 1
; VS2: %scale = call i32 @llvm.vscale.i32() => i32 2
; VS4: %scale = call i32 @llvm.vscale.i32() => i32 4
; VS1: %length = mul i32 %scale, 4 => i32 4
; VS2: %length = mul i32 %scale, 4 => i32 8
; VS4: %length = mul i32 %scale, 4 => i32 16
; VS1: %div = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32{{.*}} => { i32 4, poison, poison, poison }
; VS2: %div = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32{{.*}} => { i32 4, i32 4, poison, poison, poison, poison, poison, poison }
; VS4: %div = call <vscale x 4 x i32> @llvm.vp.udiv.nxv4i32{{.*}} => { i32 4, i32 4, i32 4, i32 4, poison, poison, poison, poison, poison, poison, poison, poison, poison, poison, poison, poison }
; CHECK: %first = extractelement{{.*}} => i32 4
; CHECK: %tail = extractelement{{.*}} => poison
; VS1: %add = call i32 @llvm.vp.reduce.add.nxv4i32{{.*}} => i32 11
; VS2: %add = call i32 @llvm.vp.reduce.add.nxv4i32{{.*}} => i32 19
; VS4: %add = call i32 @llvm.vp.reduce.add.nxv4i32{{.*}} => i32 35
; CHECK: %fp = call double @llvm.vp.reduce.fadd.nxv4f64{{.*}} => double 7.000000e+00
; CHECK: %cttz = call i64 @llvm.vp.cttz.elts.i64.nxv4i32{{.*}} => i64 1
; VS1: %zeros = call i32 @llvm.vp.cttz.elts.i32.nxv4i32{{.*}} => i32 4
; VS2: %zeros = call i32 @llvm.vp.cttz.elts.i32.nxv4i32{{.*}} => i32 8
; VS4: %zeros = call i32 @llvm.vp.cttz.elts.i32.nxv4i32{{.*}} => i32 16
; CHECK: %count_matches = icmp eq i32 %zeros, %length => T
; CHECK: %narrow = call i1 @llvm.vp.cttz.elts.i1.nxv4i32{{.*}} => poison
; VS1: %boundary = call i4 @llvm.vp.cttz.elts.i4.nxv4i32{{.*}} => i4 4
; VS2: %boundary = call i4 @llvm.vp.cttz.elts.i4.nxv4i32{{.*}} => i4 -8
; VS4: %boundary = call i4 @llvm.vp.cttz.elts.i4.nxv4i32{{.*}} => poison
