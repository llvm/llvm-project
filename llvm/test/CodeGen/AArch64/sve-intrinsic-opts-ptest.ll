; RUN: opt -S -sve-intrinsic-opts -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck --check-prefix OPT %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

define i1 @ptest_any1(<vscale x 2 x i1> %a) {
; OPT-LABEL: ptest_any1
; OPT: %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 0)
; OPT-NOT: convert
; OPT-NEXT: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.any.nxv2i1(<vscale x 2 x i1> %mask, <vscale x 2 x i1> %a)
; OPT-NEXT: ret i1 %[[OUT]]
  %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 0)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

; No transform because the ptest is using differently sized operands.
define i1 @ptest_any2(<vscale x 4 x i1> %a) {
; OPT-LABEL: ptest_any2
; OPT: %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
; OPT-NEXT: %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %mask)
; OPT-NEXT: %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
; OPT-NEXT: %out = call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  %mask = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

define i1 @ptest_first(<vscale x 4 x i1> %a) {
; OPT-LABEL: ptest_first
; OPT: %mask = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 0)
; OPT-NOT: convert
; OPT-NEXT: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.first.nxv4i1(<vscale x 4 x i1> %mask, <vscale x 4 x i1> %a)
; OPT-NEXT: ret i1 %[[OUT]]
  %mask = tail call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 0)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.first.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

define i1 @ptest_last(<vscale x 8 x i1> %a) {
; OPT-LABEL: ptest_last
; OPT: %mask = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 0)
; OPT-NOT: convert
; OPT-NEXT: %[[OUT:.*]] = call i1 @llvm.aarch64.sve.ptest.last.nxv8i1(<vscale x 8 x i1> %mask, <vscale x 8 x i1> %a)
; OPT-NEXT: ret i1 %[[OUT]]
  %mask = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 0)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %mask)
  %2 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %a)
  %out = call i1 @llvm.aarch64.sve.ptest.last.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %2)
  ret i1 %out
}

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)

declare i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare i1 @llvm.aarch64.sve.ptest.first.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare i1 @llvm.aarch64.sve.ptest.last.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1>)
