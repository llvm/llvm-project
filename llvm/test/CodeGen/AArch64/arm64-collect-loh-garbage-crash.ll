; RUN: llc -o - %s -mtriple=arm64-apple-ios -O3 -aarch64-enable-collect-loh | FileCheck %s
; RUN: llc -o - %s -mtriple=arm64_32-apple-watchos -O3 -aarch64-enable-collect-loh | FileCheck %s
; Check that the LOH analysis does not crash when the analysed chained
; contains instructions that are filtered out.
;
; Before the fix for <rdar://problem/16041712>, these cases were removed
; from the main container. Now, the deterministic container does not allow
; to remove arbitrary values, so we have to live with garbage values.
; <rdar://problem/16041712>

%"class.H4ISP::H4ISPDevice" = type { ptr, ptr, ptr, ptr }

%"class.H4ISP::H4ISPCameraManager" = type opaque

declare i32 @_ZN5H4ISP11H4ISPDevice32ISP_SelectBestMIPIFrequencyIndexEjPj(ptr)

@pH4ISPDevice = hidden global ptr null, align 8

; CHECK-LABEL: _foo:
; CHECK: ret
; CHECK-NOT: .loh AdrpLdrGotLdr
define void @foo() {
entry:
  br label %if.then83
if.then83:                                        ; preds = %if.end81
  %tmp = load ptr, ptr @pH4ISPDevice, align 8
  %call84 = call i32 @_ZN5H4ISP11H4ISPDevice32ISP_SelectBestMIPIFrequencyIndexEjPj(ptr %tmp) #19
  tail call void asm sideeffect "", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27}"()
  %tmp2 = load ptr, ptr @pH4ISPDevice, align 8
  tail call void asm sideeffect "", "~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x28}"()
  %pCameraManager.i268 = getelementptr inbounds %"class.H4ISP::H4ISPDevice", ptr %tmp2, i64 0, i32 3
  %tmp3 = load ptr, ptr %pCameraManager.i268, align 8
  %tobool.i269 = icmp eq ptr %tmp3, null
  br i1 %tobool.i269, label %if.then83, label %end
end:
  ret void
}

