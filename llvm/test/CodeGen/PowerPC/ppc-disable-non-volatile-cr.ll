; Note: Test option to disable use of non-volatile CR to avoid CR spilling in prologue.
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -ppc-disable-non-volatile-cr\
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck --check-prefix=CHECK-DISABLE %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu\
; RUN:     -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck --check-prefix=CHECK-ENABLE %s

; Function Attrs: nounwind
define dso_local signext i32 @DisableNonVolatileCR(i32 signext %a, i32 signext %b) {
; CHECK-DISABLE-LABEL: DisableNonVolatileCR:
; CHECK-DISABLE:       # %bb.0: # %entry
; CHECK-DISABLE-NOT:    mfocrf {{r[0-9]+}}
; CHECK-DISABLE-NOT:    stw {{r[0-9]+}}
; CHECK-DISABLE:        stdu r1
; CHECK-DISABLE-DAG:    mfocrf [[REG2:r[0-9]+]]
; CHECK-DISABLE-DAG:    stw [[REG2]]
; CHECK-DISABLE:        # %bb.1: # %if.then
;
; CHECK-ENABLE-LABEL: DisableNonVolatileCR:
; CHECK-ENABLE:       # %bb.0: # %entry
; CHECK-ENABLE-DAG:    mfocrf [[REG1:r[0-9]+]]
; CHECK-ENABLE-DAG:    stw [[REG1]]
; CHECK-ENABLE:        stdu r1
; CHECK-ENABLE-NOT:    mfocrf {{r[0-9]+}}
; CHECK-ENABLE-NOT:    stw {{r[0-9]+}}
; CHECK-ENABLE:        # %bb.1: # %if.then

entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void @fa()
  br label %if.end

if.else:                                          ; preds = %entry
  tail call void @fb()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %conv = zext i1 %cmp to i32
  %call = tail call signext i32 @callee(i32 signext %conv)
  ret i32 %call
}

declare void @fa(...)
declare void @fb(...)
declare signext i32 @callee(i32 signext)
