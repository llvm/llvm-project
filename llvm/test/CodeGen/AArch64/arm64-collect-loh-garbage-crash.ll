; RUN: llc -mtriple=arm64-apple-ios -O3 -aarch64-enable-collect-loh -aarch64-collect-loh-bb-only=true -aarch64-collect-loh-pre-collect-register=false < %s -o - | FileCheck %s
; Check that the LOH analysis does not crash when the analysed chained
; contains instructions that are filtered out.
;
; Before the fix for <rdar://problem/16041712>, these cases were removed
; from the main container. Now, the deterministic container does not allow
; to remove arbitrary values, so we have to live with garbage values.
; <rdar://problem/16041712>

%"class.H4ISP::H4ISPDevice" = type { i32 (%"class.H4ISP::H4ISPDevice"*, i32, i8*, i8*)*, i8*, i32*, %"class.H4ISP::H4ISPCameraManager"* }

%"class.H4ISP::H4ISPCameraManager" = type opaque

declare i32 @_ZN5H4ISP11H4ISPDevice32ISP_SelectBestMIPIFrequencyIndexEjPj(%"class.H4ISP::H4ISPDevice"*)

@pH4ISPDevice = hidden global %"class.H4ISP::H4ISPDevice"* null, align 8

; CHECK-LABEL: _foo:
; CHECK: ret
; CHECK-NOT: .loh AdrpLdrGotLdr
define void @foo() {
entry:
  br label %if.then83
if.then83:                                        ; preds = %if.end81
  %tmp = load %"class.H4ISP::H4ISPDevice"*, %"class.H4ISP::H4ISPDevice"** @pH4ISPDevice, align 8
  %call84 = call i32 @_ZN5H4ISP11H4ISPDevice32ISP_SelectBestMIPIFrequencyIndexEjPj(%"class.H4ISP::H4ISPDevice"* %tmp) #19
  tail call void asm sideeffect "", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27}"()
  %tmp2 = load %"class.H4ISP::H4ISPDevice"*, %"class.H4ISP::H4ISPDevice"** @pH4ISPDevice, align 8
  tail call void asm sideeffect "", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x28}"()
  %pCameraManager.i268 = getelementptr inbounds %"class.H4ISP::H4ISPDevice", %"class.H4ISP::H4ISPDevice"* %tmp2, i64 0, i32 3
  %tmp3 = load %"class.H4ISP::H4ISPCameraManager"*, %"class.H4ISP::H4ISPCameraManager"** %pCameraManager.i268, align 8
  %tobool.i269 = icmp eq %"class.H4ISP::H4ISPCameraManager"* %tmp3, null
  br i1 %tobool.i269, label %if.then83, label %end
end:
  ret void
}

