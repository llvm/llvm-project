; RUN: llc -mtriple=aarch64-unknown-linux-gnu -mattr=+sve -o - %s | FileCheck %s

; Just check that llc does not crash
define void @sve_setcc_promote_nxv8i8_crash(ptr %0, <vscale x 8 x i8> %1, <vscale x 8 x i8> %2) {
; CHECK-LABEL: sve_setcc_promote_nxv8i8_crash:
iter.check:
  %wide.load61.pre = load <vscale x 8 x i8>, ptr null, align 1
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %iter.check
  %3 = icmp eq <vscale x 8 x i8> %wide.load61.pre, zeroinitializer
  %4 = icmp slt <vscale x 8 x i8> %wide.load61.pre, zeroinitializer
  %5 = select <vscale x 8 x i1> %3, <vscale x 8 x i8> zeroinitializer, <vscale x 8 x i8> %1
  %6 = select <vscale x 8 x i1> %4, <vscale x 8 x i8> zeroinitializer, <vscale x 8 x i8> %2
  %7 = xor <vscale x 8 x i8> %5, %6
  store <vscale x 8 x i8> %7, ptr %0, align 1
  br label %vec.epilog.vector.body
}

