; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 | FileCheck %s

declare {i32, i1} @llvm.nvvm.shfl.down.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.down.f32p(float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.up.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.up.f32p(float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.bfly.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.bfly.f32p(float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.idx.i32p(i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.idx.f32p(float, i32, i32)

; CHECK-LABEL: .func{{.*}}shfl.i32.rrr
define {i32, i1} @shfl.i32.rrr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.irr
define {i32, i1} @shfl.i32.irr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.rri
define {i32, i1} @shfl.i32.rri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 1)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.iri
define {i32, i1} @shfl.i32.iri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 %b, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.rir
define {i32, i1} @shfl.i32.rir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 1, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.iir
define {i32, i1} @shfl.i32.iir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 2, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.rii
define {i32, i1} @shfl.i32.rii(i32 %a) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 1, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.i32.iii
define {i32, i1} @shfl.i32.iii(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.down.i32p(i32 %a, i32 2, i32 3)
  ret {i32, i1} %val
}

;; Same intrinsics, but for float

; CHECK-LABEL: .func{{.*}}shfl.f32.rrr
define {float, i1} @shfl.f32.rrr(float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.irr
define {float, i1} @shfl.f32.irr(float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.rri
define {float, i1} @shfl.f32.rri(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 1)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.iri
define {float, i1} @shfl.f32.iri(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 %b, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.rir
define {float, i1} @shfl.f32.rir(float %a, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 1, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.iir
define {float, i1} @shfl.f32.iir(float %a, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 2, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.rii
define {float, i1} @shfl.f32.rii(float %a) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 1, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.f32.iii
define {float, i1} @shfl.f32.iii(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.down.f32p(float %a, i32 2, i32 3)
  ret {float, i1} %val
}
