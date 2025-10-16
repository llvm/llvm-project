; RUN: llc < %s -mtriple=arm64-apple-darwin

; Can't copy or spill / restore CPSR.
; rdar://9105206

define fastcc void @t(i1 %arg) ssp align 2 {
entry:
  br i1 %arg, label %bb3.i, label %bb2.i

bb2.i:                                            ; preds = %entry
  br label %bb3.i

bb3.i:                                            ; preds = %bb2.i, %entry
  br i1 %arg, label %_ZN12gjkepa2_impl3EPA6appendERNS0_5sListEPNS0_5sFaceE.exit71, label %bb.i69

bb.i69:                                           ; preds = %bb3.i
  br label %_ZN12gjkepa2_impl3EPA6appendERNS0_5sListEPNS0_5sFaceE.exit71

_ZN12gjkepa2_impl3EPA6appendERNS0_5sListEPNS0_5sFaceE.exit71: ; preds = %bb.i69, %bb3.i
  %0 = select i1 undef, float 0.000000e+00, float undef
  %1 = fdiv float %0, undef
  %2 = fcmp ult float %1, 0xBF847AE140000000
  %storemerge9 = select i1 %2, float %1, float 0.000000e+00
  store float %storemerge9, ptr undef, align 4
  br i1 %arg, label %bb42, label %bb47

bb42:                                             ; preds = %_ZN12gjkepa2_impl3EPA6appendERNS0_5sListEPNS0_5sFaceE.exit71
  br i1 %arg, label %bb46, label %bb53

bb46:                                             ; preds = %bb42
  br label %bb48

bb47:                                             ; preds = %_ZN12gjkepa2_impl3EPA6appendERNS0_5sListEPNS0_5sFaceE.exit71
  br label %bb48

bb48:                                             ; preds = %bb47, %bb46
  br i1 %arg, label %bb1.i14, label %bb.i13

bb.i13:                                           ; preds = %bb48
  br label %bb1.i14

bb1.i14:                                          ; preds = %bb.i13, %bb48
  br label %bb53

bb53:                                             ; preds = %bb1.i14, %bb42
  ret void
}
