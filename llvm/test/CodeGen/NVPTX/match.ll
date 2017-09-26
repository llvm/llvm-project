; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 | FileCheck %s

declare i32 @llvm.nvvm.match.any.sync.i32(i32, i32)
declare i64 @llvm.nvvm.match.any.sync.i64(i32, i64)

; CHECK-LABEL: .func{{.*}}match.any.sync.i32
define i32 @match.any.sync.i32(i32 %mask, i32 %value) {
  ; CHECK: ld.param.u32 	[[MASK:%r[0-9]+]], [match.any.sync.i32_param_0];
  ; CHECK: ld.param.u32 	[[VALUE:%r[0-9]+]], [match.any.sync.i32_param_1];

  ; CHECK:  match.any.sync.b32  [[V0:%r[0-9]+]], [[VALUE]], [[MASK]];
  %v0 = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 %value)
  ; CHECK:  match.any.sync.b32  [[V1:%r[0-9]+]], [[VALUE]], 1;
  %v1 = call i32 @llvm.nvvm.match.any.sync.i32(i32 1, i32 %value)
  ; CHECK:  match.any.sync.b32  [[V2:%r[0-9]+]], 2, [[MASK]];
  %v2 = call i32 @llvm.nvvm.match.any.sync.i32(i32 %mask, i32 2)
  ; CHECK:  match.any.sync.b32  [[V3:%r[0-9]+]], 4, 3;
  %v3 = call i32 @llvm.nvvm.match.any.sync.i32(i32 3, i32 4)
  %sum1 = add i32 %v0, %v1
  %sum2 = add i32 %v2, %v3
  %sum3 = add i32 %sum1, %sum2
  ret i32 %sum3;
}

; CHECK-LABEL: .func{{.*}}match.any.sync.i64
define i64 @match.any.sync.i64(i32 %mask, i64 %value) {
  ; CHECK: ld.param.u32 	[[MASK:%r[0-9]+]], [match.any.sync.i64_param_0];
  ; CHECK: ld.param.u64 	[[VALUE:%rd[0-9]+]], [match.any.sync.i64_param_1];

  ; CHECK:  match.any.sync.b64  [[V0:%rd[0-9]+]], [[VALUE]], [[MASK]];
  %v0 = call i64 @llvm.nvvm.match.any.sync.i64(i32 %mask, i64 %value)
  ; CHECK:  match.any.sync.b64  [[V1:%rd[0-9]+]], [[VALUE]], 1;
  %v1 = call i64 @llvm.nvvm.match.any.sync.i64(i32 1, i64 %value)
  ; CHECK:  match.any.sync.b64  [[V2:%rd[0-9]+]], 2, [[MASK]];
  %v2 = call i64 @llvm.nvvm.match.any.sync.i64(i32 %mask, i64 2)
  ; CHECK:  match.any.sync.b64  [[V3:%rd[0-9]+]], 4, 3;
  %v3 = call i64 @llvm.nvvm.match.any.sync.i64(i32 3, i64 4)
  %sum1 = add i64 %v0, %v1
  %sum2 = add i64 %v2, %v3
  %sum3 = add i64 %sum1, %sum2
  ret i64 %sum3;
}

declare {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32, i32)
declare {i64, i1} @llvm.nvvm.match.all.sync.i64p(i32, i64)

; CHECK-LABEL: .func{{.*}}match.all.sync.i32p(
define {i32,i1} @match.all.sync.i32p(i32 %mask, i32 %value) {
  ; CHECK: ld.param.u32 	[[MASK:%r[0-9]+]], [match.all.sync.i32p_param_0];
  ; CHECK: ld.param.u32 	[[VALUE:%r[0-9]+]], [match.all.sync.i32p_param_1];

  ; CHECK:  match.all.sync.b32 {{%r[0-9]+\|%p[0-9]+}}, [[VALUE]], [[MASK]];
  %r1 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 %value)
  %v1 = extractvalue {i32, i1} %r1, 0
  %p1 = extractvalue {i32, i1} %r1, 1

  ; CHECK:  match.all.sync.b32 {{%r[0-9]+\|%p[0-9]+}}, 1, [[MASK]];
  %r2 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 %mask, i32 1)
  %v2 = extractvalue {i32, i1} %r2, 0
  %p2 = extractvalue {i32, i1} %r2, 1

  ; CHECK:  match.all.sync.b32 {{%r[0-9]+\|%p[0-9]+}}, [[VALUE]], 2;
  %r3 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 2, i32 %value)
  %v3 = extractvalue {i32, i1} %r3, 0
  %p3 = extractvalue {i32, i1} %r3, 1

  ; CHECK:  match.all.sync.b32 {{%r[0-9]+\|%p[0-9]+}}, 4, 3;
  %r4 = call {i32, i1} @llvm.nvvm.match.all.sync.i32p(i32 3, i32 4)
  %v4 = extractvalue {i32, i1} %r4, 0
  %p4 = extractvalue {i32, i1} %r4, 1

  %vsum1 = add i32 %v1, %v2
  %vsum2 = add i32 %v3, %v4
  %vsum3 = add i32 %vsum1, %vsum2
  %psum1 = add i1 %p1, %p2
  %psum2 = add i1 %p3, %p4
  %psum3 = add i1 %psum1, %psum2
  %ret0 = insertvalue {i32, i1} undef, i32 %vsum3, 0
  %ret1 = insertvalue {i32, i1} %ret0, i1 %psum3, 1
  ret {i32, i1} %ret1;
}

; CHECK-LABEL: .func{{.*}}match.all.sync.i64p(
define {i64,i1} @match.all.sync.i64p(i32 %mask, i64 %value) {
  ; CHECK: ld.param.u32 	[[MASK:%r[0-9]+]], [match.all.sync.i64p_param_0];
  ; CHECK: ld.param.u64 	[[VALUE:%rd[0-9]+]], [match.all.sync.i64p_param_1];

  ; CHECK:  match.all.sync.b64 {{%rd[0-9]+\|%p[0-9]+}}, [[VALUE]], [[MASK]];
  %r1 = call {i64, i1} @llvm.nvvm.match.all.sync.i64p(i32 %mask, i64 %value)
  %v1 = extractvalue {i64, i1} %r1, 0
  %p1 = extractvalue {i64, i1} %r1, 1

  ; CHECK:  match.all.sync.b64 {{%rd[0-9]+\|%p[0-9]+}}, 1, [[MASK]];
  %r2 = call {i64, i1} @llvm.nvvm.match.all.sync.i64p(i32 %mask, i64 1)
  %v2 = extractvalue {i64, i1} %r2, 0
  %p2 = extractvalue {i64, i1} %r2, 1

  ; CHECK:  match.all.sync.b64 {{%rd[0-9]+\|%p[0-9]+}}, [[VALUE]], 2;
  %r3 = call {i64, i1} @llvm.nvvm.match.all.sync.i64p(i32 2, i64 %value)
  %v3 = extractvalue {i64, i1} %r3, 0
  %p3 = extractvalue {i64, i1} %r3, 1

  ; CHECK:  match.all.sync.b64 {{%rd[0-9]+\|%p[0-9]+}}, 4, 3;
  %r4 = call {i64, i1} @llvm.nvvm.match.all.sync.i64p(i32 3, i64 4)
  %v4 = extractvalue {i64, i1} %r4, 0
  %p4 = extractvalue {i64, i1} %r4, 1

  %vsum1 = add i64 %v1, %v2
  %vsum2 = add i64 %v3, %v4
  %vsum3 = add i64 %vsum1, %vsum2
  %psum1 = add i1 %p1, %p2
  %psum2 = add i1 %p3, %p4
  %psum3 = add i1 %psum1, %psum2
  %ret0 = insertvalue {i64, i1} undef, i64 %vsum3, 0
  %ret1 = insertvalue {i64, i1} %ret0, i1 %psum3, 1
  ret {i64, i1} %ret1;
}
