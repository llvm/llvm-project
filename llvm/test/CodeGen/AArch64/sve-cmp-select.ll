; RUN: llc -mtriple=aarch64-linux-unknown -mattr=+sve -o - < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix="WARN" --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 16 x i8> @vselect_cmp_ne(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
  ; CHECK-LABEL: vselect_cmp_ne
  ; CHECK:       // %bb.0:
	; CHECK-NEXT:    ptrue	p0.b
	; CHECK-NEXT:    cmpne	p0.b, p0/z, z0.b, z1.b
	; CHECK-NEXT:    sel	z0.b, p0, z1.b, z2.b
	; CHECK-NEXT:    ret
  %cmp = icmp ne <vscale x 16 x i8> %a, %b
  %d = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c
  ret <vscale x 16 x i8> %d
}

define <vscale x 16 x i8> @vselect_cmp_sgt(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
  ; CHECK-LABEL: vselect_cmp_sgt
  ; CHECK:       // %bb.0:
  ; CHECK-NEXT: 	ptrue	p0.b
  ; CHECK-NEXT: 	cmpgt	p0.b, p0/z, z0.b, z1.b
  ; CHECK-NEXT: 	sel	z0.b, p0, z1.b, z2.b
  ; CHECK-NEXT: 	ret
  %cmp = icmp sgt <vscale x 16 x i8> %a, %b
  %d = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c
  ret <vscale x 16 x i8> %d
}

define <vscale x 16 x i8> @vselect_cmp_ugt(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
  ; CHECK-LABEL: vselect_cmp_ugt
  ; CHECK:       // %bb.0:
  ; CHECK-NEXT: 	ptrue	p0.b
  ; CHECK-NEXT: 	cmphi	p0.b, p0/z, z0.b, z1.b
  ; CHECK-NEXT: 	sel	z0.b, p0, z1.b, z2.b
  ; CHECK-NEXT: 	ret
  %cmp = icmp ugt <vscale x 16 x i8> %a, %b
  %d = select <vscale x 16 x i1> %cmp, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c
  ret <vscale x 16 x i8> %d
}
