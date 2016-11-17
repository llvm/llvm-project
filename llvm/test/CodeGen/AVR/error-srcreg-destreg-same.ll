; RUN: llc < %s -march=avr | FileCheck %s
; XFAIL: *

; This occurs when compiling Rust libcore.
;
; Assertion failed:
; (DstReg != SrcReg && "SrcReg and DstReg cannot be the same")
;   lib/Target/AVR/AVRExpandPseudoInsts.cpp, line 817
;
; https://github.com/avr-llvm/llvm/issues/229

; CHECK-LABEL: rust_eh_personality
declare void @rust_eh_personality()

; CHECK-LABEL: __udivmoddi4
define void @__udivmoddi4(i64 %arg, i64 %arg1) personality i32 (...)* bitcast (void ()* @rust_eh_personality to i32 (...)*) {
entry-block:
  %tmp = lshr i64 %arg, 32
  %tmp2 = trunc i64 %tmp to i32
  %tmp3 = trunc i64 %arg to i32
  %tmp4 = add i64 %arg1, -1
  br label %bb135

bb133.loopexit:
  ret void

bb135:
  %carry.0120 = phi i64 [ 0, %entry-block ], [ %phitmp, %bb135 ]
  %q.sroa.12.1119 = phi i32 [ %tmp3, %entry-block ], [ %q.sroa.12.0.extract.trunc, %bb135 ]
  %q.sroa.0.1118 = phi i32 [ 0, %entry-block ], [ %q.sroa.0.0.extract.trunc, %bb135 ]
  %r.sroa.0.1116 = phi i32 [ %tmp2, %entry-block ], [ undef, %bb135 ]
  %r.sroa.0.0.insert.ext62 = zext i32 %r.sroa.0.1116 to i64
  %r.sroa.0.0.insert.insert64 = or i64 0, %r.sroa.0.0.insert.ext62
  %tmp5 = shl nuw nsw i64 %r.sroa.0.0.insert.ext62, 1
  %q.sroa.12.0.insert.ext101 = zext i32 %q.sroa.12.1119 to i64
  %q.sroa.12.0.insert.shift102 = shl nuw i64 %q.sroa.12.0.insert.ext101, 32
  %q.sroa.0.0.insert.ext87 = zext i32 %q.sroa.0.1118 to i64
  %q.sroa.0.0.insert.insert89 = or i64 %q.sroa.12.0.insert.shift102, %q.sroa.0.0.insert.ext87
  %tmp6 = lshr i64 %q.sroa.12.0.insert.ext101, 31
  %tmp7 = lshr i64 %r.sroa.0.0.insert.insert64, 31
  %tmp8 = shl nuw nsw i64 %q.sroa.0.0.insert.ext87, 1
  %tmp9 = or i64 %tmp8, %carry.0120
  %q.sroa.0.0.extract.trunc = trunc i64 %tmp9 to i32
  %tmp10 = lshr i64 %q.sroa.0.0.insert.insert89, 31
  %q.sroa.12.0.extract.trunc = trunc i64 %tmp10 to i32
  %r.sroa.13.0.insert.shift72 = shl i64 %tmp7, 32
  %.masked114 = and i64 %tmp5, 4294967294
  %r.sroa.0.0.insert.ext57 = or i64 %tmp6, %.masked114
  %r.sroa.0.0.insert.insert59 = or i64 %r.sroa.0.0.insert.ext57, %r.sroa.13.0.insert.shift72
  %tmp11 = sub i64 %tmp4, %r.sroa.0.0.insert.insert59
  %tmp12 = ashr i64 %tmp11, 63
  %phitmp = and i64 %tmp12, 1
  %tmp13 = icmp ult i32 undef, 32
  br i1 %tmp13, label %bb135, label %bb133.loopexit
}

