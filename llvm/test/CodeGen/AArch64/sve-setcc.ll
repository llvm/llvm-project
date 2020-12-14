; RUN: llc -mtriple=aarch64--linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

; Ensure we use the inverted CC result of SVE compare instructions when branching.
define void @sve_cmplt_setcc_inverted(<vscale x 8 x i16>* %out, <vscale x 8 x i16> %in, <vscale x 8 x i1> %pg) {
; CHECK-LABEL: @sve_cmplt_setcc_inverted
; CHECK: cmplt p1.h, p0/z, z0.h, #0
; CHECK-NEXT: b.ne
entry:
  %0 = tail call <vscale x 8 x i1> @llvm.aarch64.sve.cmplt.wide.nxv8i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %in, <vscale x 2 x i64> zeroinitializer)
  %1 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv8i1(<vscale x 8 x i1> %pg, <vscale x 8 x i1> %0)
  br i1 %1, label %if.end, label %if.then

if.then:
  tail call void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16> %in, <vscale x 8 x i16>* %out, i32 2, <vscale x 8 x i1> %pg)
  br label %if.end

if.end:
  ret void
}

declare i1 @llvm.aarch64.sve.ptest.any.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)

declare <vscale x 8 x i1> @llvm.aarch64.sve.cmplt.wide.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x i16>, <vscale x 2 x i64>)

declare void @llvm.masked.store.nxv8i16.p0nxv8i16(<vscale x 8 x i16>, <vscale x 8 x i16>*, i32, <vscale x 8 x i1>)
