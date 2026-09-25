; RUN: llc -march=hexagon -mcpu=hexagonv73 -O2 -hexagon-fuse-intrinsic-vminub=true < %s | FileCheck %s --check-prefix=FUSE
; RUN: llc -march=hexagon -mcpu=hexagonv73 -O2 -hexagon-fuse-intrinsic-vminub=false < %s | FileCheck %s --check-prefix=NOFUSE

@g = external global i64
@r = external global i32

; FUSE-LABEL: fused:
; FUSE-NOT: cmp.gtu
; FUSE: [[V:r[0-9]+:[0-9]+]],[[P:p[0-3]]] = vminub(
; FUSE-DAG: r{{[0-9]+}} = [[P]]
; FUSE-DAG: memd(gp+#g) = [[V]]
define i32 @fused(i64 %a, i64 %b) {
entry:
  %p = tail call i32 @llvm.hexagon.C2.cmpgtup(i64 %a, i64 %b)
  %v = tail call i64 @llvm.hexagon.A2.vminub(i64 %a, i64 %b)
  store i64 %v, i64* @g, align 8
  store i32 %p, i32* @r, align 4
  %pe = zext i32 %p to i64
  %sum = add i64 %v, %pe
  %ret = trunc i64 %sum to i32
  ret i32 %ret
}

; NOFUSE-LABEL: fused:
; NOFUSE-DAG: {{p[0-3]}} = cmp.gtu(
; NOFUSE-DAG: {{r[0-9]+:[0-9]+}} = vminub(
; NOFUSE-NOT: {{r[0-9]+:[0-9]+}},{{p[0-3]}} = vminub(

; FUSE-LABEL: only_vmin:
; FUSE-NOT: cmp.gtu
; FUSE: {{r[0-9]+:[0-9]+}} = vminub(
define i64 @only_vmin(i64 %a, i64 %b) {
  %v = tail call i64 @llvm.hexagon.A2.vminub(i64 %a, i64 %b)
  ret i64 %v
}

; FUSE-LABEL: only_cmp:
; FUSE: {{p[0-3]}} = cmp.gtu(
define i32 @only_cmp(i64 %a, i64 %b) {
  %p = tail call i32 @llvm.hexagon.C2.cmpgtup(i64 %a, i64 %b)
  ret i32 %p
}

; FUSE-LABEL: mismatch:
; FUSE-NOT: {{r[0-9]+:[0-9]+}},{{p[0-3]}} = vminub(
; FUSE-DAG: {{r[0-9]+:[0-9]+}} = vminub(
; FUSE-DAG: {{p[0-3]}} = cmp.gtu(
define i32 @mismatch(i64 %a, i64 %b) {
  %p = tail call i32 @llvm.hexagon.C2.cmpgtup(i64 %a, i64 %b)
  %v = tail call i64 @llvm.hexagon.A2.vminub(i64 %b, i64 %a)
  %pe = zext i32 %p to i64
  %sum = add i64 %v, %pe
  %ret = trunc i64 %sum to i32
  ret i32 %ret
}

declare i64 @llvm.hexagon.A2.vminub(i64, i64)
declare i32 @llvm.hexagon.C2.cmpgtup(i64, i64)
