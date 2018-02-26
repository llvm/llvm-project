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
define i32 @get_set_complex({ float, float }* noalias nocapture %retptr,
                            { i8*, i32 }** noalias nocapture readnone %excinfo,
                            i8* noalias nocapture readnone %env,
                            [38 x i8]* nocapture %arg.rec,
                            float %arg.val.0, float %arg.val.1)
{
entry:
  %inserted.real = insertvalue { float, float } undef, float %arg.val.0, 0
  %inserted.imag = insertvalue { float, float } %inserted.real, float %arg.val.1, 1
  %.15 = getelementptr inbounds [38 x i8], [38 x i8]* %arg.rec, i32 0, i32 10
  %.16 = bitcast i8* %.15 to { float, float }*
  %.17 = bitcast i8* %.15 to float*
  %.18 = load float, float* %.17, align 1
  %.19 = getelementptr inbounds [38 x i8], [38 x i8]* %arg.rec, i32 0, i32 14
  %tmp = bitcast i8* %.19 to float*
  %.20 = load float, float* %tmp, align 1
  %inserted.real.1 = insertvalue { float, float } undef, float %.18, 0
  %inserted.imag.1 = insertvalue { float, float } %inserted.real.1, float %.20, 1
  store { float, float } %inserted.imag, { float, float }* %.16, align 1
  store { float, float } %inserted.imag.1, { float, float }* %retptr, align 4
  ret i32 0
}
