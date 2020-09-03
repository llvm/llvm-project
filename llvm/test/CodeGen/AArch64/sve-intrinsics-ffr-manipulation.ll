; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; RDFFR
;

define <vscale x 16 x i1> @rdffr() {
; CHECK-LABEL: rdffr:
; CHECK: rdffr p0.b
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.rdffr()
  ret <vscale x 16 x i1> %out
}

define <vscale x 16 x i1> @rdffr_z(<vscale x 16 x i1> %pg) {
; CHECK-LABEL: rdffr_z:
; CHECK: rdffr p0.b, p0/z
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i1> @llvm.aarch64.sve.rdffr.z(<vscale x 16 x i1> %pg)
  ret <vscale x 16 x i1> %out
}

;
; SETFFR
;

define void @set_ffr() {
; CHECK-LABEL: set_ffr:
; CHECK: setffr
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.setffr()
  ret void
}

;
; WRFFR
;

define void @wrffr(<vscale x 16 x i1> %a) {
; CHECK-LABEL: wrffr:
; CHECK: wrffr p0.b
; CHECK-NEXT: ret
  call void @llvm.aarch64.sve.wrffr(<vscale x 16 x i1> %a)
  ret void
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.rdffr()
declare <vscale x 16 x i1> @llvm.aarch64.sve.rdffr.z(<vscale x 16 x i1>)
declare void @llvm.aarch64.sve.setffr()
declare void @llvm.aarch64.sve.wrffr(<vscale x 16 x i1>)
