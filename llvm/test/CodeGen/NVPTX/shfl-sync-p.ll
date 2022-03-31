; RUN: llc < %s -march=nvptx64 -mcpu=sm_30 -mattr=+ptx60 | FileCheck %s

declare {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32, float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.up.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.up.f32p(i32, float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.bfly.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.bfly.f32p(i32, float, i32, i32)
declare {i32, i1} @llvm.nvvm.shfl.sync.idx.i32p(i32, i32, i32, i32)
declare {float, i1} @llvm.nvvm.shfl.sync.idx.f32p(i32, float, i32, i32)

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.rrr
define {i32, i1} @shfl.sync.i32.rrr(i32 %mask, i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.irr
define {i32, i1} @shfl.sync.i32.irr(i32 %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 %b, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.rri
define {i32, i1} @shfl.sync.i32.rri(i32 %mask, i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 %b, i32 1)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.iri
define {i32, i1} @shfl.sync.i32.iri(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 %b, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.rir
define {i32, i1} @shfl.sync.i32.rir(i32 %mask, i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 1, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.iir
define {i32, i1} @shfl.sync.i32.iir(i32 %a, i32 %c) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 2, i32 %c)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.rii
define {i32, i1} @shfl.sync.i32.rii(i32 %mask, i32 %a) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 %mask, i32 %a, i32 1, i32 2)
  ret {i32, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.i32.iii
define {i32, i1} @shfl.sync.i32.iii(i32 %a, i32 %b) {
  ; CHECK: ld.param.u32 [[A:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%r[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {i32, i1} @llvm.nvvm.shfl.sync.down.i32p(i32 1, i32 %a, i32 2, i32 3)
  ret {i32, i1} %val
}

;; Same intrinsics, but for float

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.rrr
define {float, i1} @shfl.sync.f32.rrr(i32 %mask, float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.irr
define {float, i1} @shfl.sync.f32.irr(float %a, i32 %b, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 %b, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.rri
define {float, i1} @shfl.sync.f32.rri(i32 %mask, float %a, i32 %b) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 1, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 %b, i32 1)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.iri
define {float, i1} @shfl.sync.f32.iri(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[B:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], [[B]], 2, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 %b, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.rir
define {float, i1} @shfl.sync.f32.rir(i32 %mask, float %a, i32 %c) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, [[C]], [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 1, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.iir
define {float, i1} @shfl.sync.f32.iir(float %a, i32 %c) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: ld.param.u32 [[C:%r[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, [[C]], 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 2, i32 %c)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.rii
define {float, i1} @shfl.sync.f32.rii(i32 %mask, float %a) {
  ; CHECK: ld.param.u32 [[MASK:%r[0-9]+]]
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 1, 2, [[MASK]];
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 %mask, float %a, i32 1, i32 2)
  ret {float, i1} %val
}

; CHECK-LABEL: .func{{.*}}shfl.sync.f32.iii
define {float, i1} @shfl.sync.f32.iii(float %a, i32 %b) {
  ; CHECK: ld.param.f32 [[A:%f[0-9]+]]
  ; CHECK: shfl.sync.down.b32 [[OUT:%f[0-9]+]]|[[OUTP:%p[0-9]+]], [[A]], 2, 3, 1;
  ; CHECK: st.param.{{.}}32 {{.*}}, [[OUT]]
  %val = call {float, i1} @llvm.nvvm.shfl.sync.down.f32p(i32 1, float %a, i32 2, i32 3)
  ret {float, i1} %val
}
