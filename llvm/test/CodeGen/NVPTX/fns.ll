; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | %ptxas-verify %}

declare i32 @llvm.nvvm.fns(i32, i32, i32)

; CHECK-LABEL: .func{{.*}}fns
define i32 @fns(i32 %mask, i32 %base, i32 %offset) {
  ; CHECK: ld.param.u32 	[[MASK:%r[0-9]+]], [fns_param_0];
  ; CHECK: ld.param.u32 	[[BASE:%r[0-9]+]], [fns_param_1];
  ; CHECK: ld.param.u32 	[[OFFSET:%r[0-9]+]], [fns_param_2];

  ; CHECK:  fns.b32 	{{%r[0-9]+}}, [[MASK]], [[BASE]], [[OFFSET]];
  %r0 = call i32 @llvm.nvvm.fns(i32 %mask, i32 %base, i32 %offset);
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, [[MASK]], [[BASE]], 0;
  %r1 = call i32 @llvm.nvvm.fns(i32 %mask, i32 %base, i32 0);
  %r01 = add i32 %r0, %r1;
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, [[MASK]], 1, [[OFFSET]];
  %r2 = call i32 @llvm.nvvm.fns(i32 %mask, i32 1, i32 %offset);
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, [[MASK]], 1, 0;
  %r3 = call i32 @llvm.nvvm.fns(i32 %mask, i32 1, i32 0);
  %r23 = add i32 %r2, %r3;
  %r0123 = add i32 %r01, %r23;
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, 2, [[BASE]], [[OFFSET]];
  %r4 = call i32 @llvm.nvvm.fns(i32 2, i32 %base, i32 %offset);
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, 2, [[BASE]], 0;
  %r5 = call i32 @llvm.nvvm.fns(i32 2, i32 %base, i32 0);
  %r45 = add i32 %r4, %r5;
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, 2, 1, [[OFFSET]];
  %r6 = call i32 @llvm.nvvm.fns(i32 2, i32 1, i32 %offset);
  ; CHECK:  fns.b32 	{{%r[0-9]+}}, 2, 1, 0;
  %r7 = call i32 @llvm.nvvm.fns(i32 2, i32 1, i32 0);
  %r67 = add i32 %r6, %r7;
  %r4567 = add i32 %r45, %r67;
  %r = add i32 %r0123, %r4567;
  ret i32 %r;
}

