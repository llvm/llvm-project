; RUN:  llc -O3 -mtriple=armv7l-unknown-linux-gnueabihf -mcpu=generic %s -o - | FileCheck %s
; Check that we respect the existing chain between loads and stores when we
; legalize unaligned loads.
; Test case from PR24669.

; Make sure the loads happen before the stores.
; CHECK-LABEL: get_set_complex:
; CHECK-NOT: str
; CHECK: ldr
; CHECK-NOT: str
; CHECK: ldr
; CHECK: str
; CHECK: {{bx|pop.*pc}}
define i32 @get_set_complex(ptr noalias nocapture %retptr,
                            ptr noalias nocapture readnone %excinfo,
                            ptr noalias nocapture readnone %env,
                            ptr nocapture %arg.rec,
                            float %arg.val.0, float %arg.val.1)
{
entry:
  %inserted.real = insertvalue { float, float } undef, float %arg.val.0, 0
  %inserted.imag = insertvalue { float, float } %inserted.real, float %arg.val.1, 1
  %.15 = getelementptr inbounds [38 x i8], ptr %arg.rec, i32 0, i32 10
  %.18 = load float, ptr %.15, align 1
  %.19 = getelementptr inbounds [38 x i8], ptr %arg.rec, i32 0, i32 14
  %.20 = load float, ptr %.19, align 1
  %inserted.real.1 = insertvalue { float, float } undef, float %.18, 0
  %inserted.imag.1 = insertvalue { float, float } %inserted.real.1, float %.20, 1
  store { float, float } %inserted.imag, ptr %.15, align 1
  store { float, float } %inserted.imag.1, ptr %retptr, align 4
  ret i32 0
}
