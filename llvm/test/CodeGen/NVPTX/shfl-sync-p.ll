; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | FileCheck %s

declare {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32, float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.up.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.up.f32p(i32, float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.bfly.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.bfly.f32p(i32, float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.idx.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.idx.f32p(i32, float, i32, i32)

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_rrr
define {i32, i1} @shfl_sync_i32_rrr(i32 %mask, i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_irr
define {i32, i1} @shfl_sync_i32_irr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_rri
define {i32, i1} @shfl_sync_i32_rri(i32 %mask, i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 %b, i32 1)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_iri
define {i32, i1} @shfl_sync_i32_iri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 %b, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_rir
define {i32, i1} @shfl_sync_i32_rir(i32 %mask, i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 1, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_iir
define {i32, i1} @shfl_sync_i32_iir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 2, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_rii
define {i32, i1} @shfl_sync_i32_rii(i32 %mask, i32 %a) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 1, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_i32_iii
define {i32, i1} @shfl_sync_i32_iii(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 2, i32 3)
  ret {i32, i1} %val
}

;; Same intrinsics, but for float

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_rrr
define {float, i1} @shfl_sync_f32_rrr(i32 %mask, float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_irr
define {float, i1} @shfl_sync_f32_irr(float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_rri
define {float, i1} @shfl_sync_f32_rri(i32 %mask, float %a, i32 %b) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 %b, i32 1)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_iri
define {float, i1} @shfl_sync_f32_iri(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 %b, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_rir
define {float, i1} @shfl_sync_f32_rir(i32 %mask, float %a, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 1, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_iir
define {float, i1} @shfl_sync_f32_iir(float %a, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 2, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_rii
define {float, i1} @shfl_sync_f32_rii(i32 %mask, float %a) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 1, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_sync_f32_iii
define {float, i1} @shfl_sync_f32_iii(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 2, i32 3)
  ret {float, i1} %val
}
