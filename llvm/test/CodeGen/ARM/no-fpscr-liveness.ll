; RUN: llc -o - %s | FileCheck %s
; Make sure we do not try to compute liveness for FPSCR which in this case
; is read before being written to (this is fine because becase FPSCR is
; reserved).
target triple = "thumbv7s-apple-ios"

%struct.wibble = type { double }

@global = common global i32 0, align 4
@global.1 = common global i32 0, align 4

; CHECK-LABEL: eggs:
; CHECK: sub sp, #8
; VMRS instruction comes before any other instruction writing FPSCR:
; CHECK-NOT: vcmp
; CHECK: vmrs {{r[0-9]}}, fpscr
; CHECK; vcmp
; ...
; CHECK: add sp, #8
; CHECK: bx lr
define i32 @eggs(double* nocapture readnone %arg) {
bb:
  %tmp = alloca %struct.wibble, align 4
  %tmp1 = bitcast %struct.wibble* %tmp to i8*
  %tmp2 = tail call i32 @llvm.flt.rounds()
  %tmp3 = ptrtoint %struct.wibble* %tmp to i32
  %tmp4 = sitofp i32 %tmp3 to double
  %tmp5 = fmul double %tmp4, 0x0123456789ABCDEF
  %tmp6 = fptosi double %tmp5 to i32
  %tmp7 = fcmp une double %tmp5, 0.000000e+00
  %tmp8 = sitofp i32 %tmp6 to double
  %tmp9 = fcmp une double %tmp5, %tmp8
  %tmp10 = and i1 %tmp7, %tmp9
  %tmp11 = sext i1 %tmp10 to i32
  %tmp12 = add nsw i32 %tmp11, %tmp6
  store i32 %tmp12, i32* @global, align 4
  %tmp13 = icmp ne i32 %tmp12, 0
  %tmp14 = icmp ne i32 %tmp2, 0
  %tmp15 = and i1 %tmp14, %tmp13
  br i1 %tmp15, label %bb16, label %bb18

bb16:                                             ; preds = %bb
  %tmp17 = load i32, i32* @global.1, align 4
  br label %bb18

bb18:                                             ; preds = %bb16, %bb
  ret i32 undef
}

declare i32 @llvm.flt.rounds()
declare i32 @zot(...)
