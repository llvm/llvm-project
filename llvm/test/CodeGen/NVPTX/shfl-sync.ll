; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s

declare i32 @llvm.nvvm.shfl.sync.down.i32(i32, i32, i32, i32)
declare float @llvm.nvvm.shfl.sync.down.f32(float, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.up.i32(i32, i32, i32, i32)
declare float @llvm.nvvm.shfl.sync.up.f32(float, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)
declare float @llvm.nvvm.shfl.sync.bfly.f32(float, i32, i32, i32)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)
declare float @llvm.nvvm.shfl.sync.idx.f32(float, i32, i32, i32)

; CHECK-LABEL: .func{{.*}}shfl.sync.rrr
define i32 @shfl.sync.rrr(i32 %mask, i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], [[B]], [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %mask, i32 %a, i32 %b, i32 %c)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.irr
define i32 @shfl.sync.irr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], [[B]], [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 1, i32 %a, i32 %b, i32 %c)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.rri
define i32 @shfl.sync.rri(i32 %mask, i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], [[B]], 1, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %mask, i32 %a, i32 %b, i32 1)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.iri
define i32 @shfl.sync.iri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], [[B]], 2, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 1, i32 %a, i32 %b, i32 2)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.rir
define i32 @shfl.sync.rir(i32 %mask, i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], 1, [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %mask, i32 %a, i32 1, i32 %c)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.iir
define i32 @shfl.sync.iir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], 2, [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 1, i32 %a, i32 2, i32 %c)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.rii
define i32 @shfl.sync.rii(i32 %mask, i32 %a) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], 1, 2, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 %mask, i32 %a, i32 1, i32 2)
  ret i32 %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.iii
define i32 @shfl.sync.iii(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]], [[A]], 2, 3, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call i32 @llvm.nvvm.shfl.sync.down.i32(i32 1, i32 %a, i32 2, i32 3)
  ret i32 %val
}
