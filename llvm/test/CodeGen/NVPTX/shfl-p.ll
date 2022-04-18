; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s

declare {i32, i1} @llvm.nvvm.shfl.down.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.down.f32p(float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.up.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.up.f32p(float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.bfly.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.bfly.f32p(float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.idx.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.idx.f32p(float, i32, i32)

; CHECK-LABEL: .func{{.*}}shfl_i32_rrr
define {i32, i1} @shfl_i32_rrr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_irr
define {i32, i1} @shfl_i32_irr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_rri
define {i32, i1} @shfl_i32_rri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 1)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_iri
define {i32, i1} @shfl_i32_iri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_rir
define {i32, i1} @shfl_i32_rir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 1, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_iir
define {i32, i1} @shfl_i32_iir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 2, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_rii
define {i32, i1} @shfl_i32_rii(i32 %a) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 1, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_i32_iii
define {i32, i1} @shfl_i32_iii(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 2, i32 3)
  ret {i32, i1} %val
}

;; Same intrinsics, but for float

; CHECK-LABEL: .func{{.*}}shfl_f32_rrr
define {float, i1} @shfl_f32_rrr(float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_irr
define {float, i1} @shfl_f32_irr(float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_rri
define {float, i1} @shfl_f32_rri(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 1)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_iri
define {float, i1} @shfl_f32_iri(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_rir
define {float, i1} @shfl_f32_rir(float %a, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 1, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_iir
define {float, i1} @shfl_f32_iir(float %a, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 2, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_rii
define {float, i1} @shfl_f32_rii(float %a) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 1, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl_f32_iii
define {float, i1} @shfl_f32_iii(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 2, i32 3)
  ret {float, i1} %val
}
