; RUN: opt -instcombine -S < %s | FileCheck %s


; CHECK-LABEL: define i8* @simplifyNullLaunder()
define i8* @simplifyNullLaunder() {
; CHECK-NEXT: ret i8* null
  %b2 = call i8* @llvm.launder.invariant.group.p0i8(i8* null)
  ret i8* %b2
}

; CHECK-LABEL: define i8 addrspace(42)* @dontsimplifyNullLaunderForDifferentAddrspace()
define i8 addrspace(42)* @dontsimplifyNullLaunderForDifferentAddrspace() {
; CHECK: %b2 = call i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)* null)
; CHECK: ret i8 addrspace(42)* %b2
  %b2 = call i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)* null)
  ret i8 addrspace(42)* %b2
}

; CHECK-LABEL: define i8* @simplifyUndefLaunder()
define i8* @simplifyUndefLaunder() {
; CHECK-NEXT: ret i8* undef
  %b2 = call i8* @llvm.launder.invariant.group.p0i8(i8* undef)
  ret i8* %b2
}

; CHECK-LABEL: define i8 addrspace(42)* @simplifyUndefLaunder2()
define i8 addrspace(42)* @simplifyUndefLaunder2() {
; CHECK-NEXT: ret i8 addrspace(42)* undef
  %b2 = call i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)* undef)
  ret i8 addrspace(42)* %b2
}

declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)*)


; CHECK-LABEL: define i8* @simplifyNullStrip()
define i8* @simplifyNullStrip() {
; CHECK-NEXT: ret i8* null
  %b2 = call i8* @llvm.strip.invariant.group.p0i8(i8* null)
  ret i8* %b2
}

; CHECK-LABEL: define i8 addrspace(42)* @dontsimplifyNullStripForDifferentAddrspace()
define i8 addrspace(42)* @dontsimplifyNullStripForDifferentAddrspace() {
; CHECK: %b2 = call i8 addrspace(42)* @llvm.strip.invariant.group.p42i8(i8 addrspace(42)* null)
; CHECK: ret i8 addrspace(42)* %b2
  %b2 = call i8 addrspace(42)* @llvm.strip.invariant.group.p42i8(i8 addrspace(42)* null)
  ret i8 addrspace(42)* %b2
}

; CHECK-LABEL: define i8* @simplifyUndefStrip()
define i8* @simplifyUndefStrip() {
; CHECK-NEXT: ret i8* undef
  %b2 = call i8* @llvm.strip.invariant.group.p0i8(i8* undef)
  ret i8* %b2
}

; CHECK-LABEL: define i8 addrspace(42)* @simplifyUndefStrip2()
define i8 addrspace(42)* @simplifyUndefStrip2() {
; CHECK-NEXT: ret i8 addrspace(42)* undef
  %b2 = call i8 addrspace(42)* @llvm.strip.invariant.group.p42i8(i8 addrspace(42)* undef)
  ret i8 addrspace(42)* %b2
}

declare i8* @llvm.strip.invariant.group.p0i8(i8*)
declare i8 addrspace(42)* @llvm.strip.invariant.group.p42i8(i8 addrspace(42)*)

