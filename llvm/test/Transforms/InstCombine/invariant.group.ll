; RUN: opt -instcombine -early-cse -earlycse-debug-hash -S < %s | FileCheck %s


; CHECK-LABEL: define i8* @simplifyNullLaunder()
define i8* @simplifyNullLaunder() {
; CHECK-NEXT: ret i8* null
  %b2 = call i8* @llvm.launder.invariant.group.p0i8(i8* null)
  ret i8* %b2
}

; CHECK-LABEL: define i8* @dontSimplifyNullLaunderNoNullOpt()
define i8* @dontSimplifyNullLaunderNoNullOpt() #0 {
; CHECK-NEXT: call i8* @llvm.launder.invariant.group.p0i8(i8* null)
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

; CHECK-LABEL: define i8* @simplifyNullStrip()
define i8* @simplifyNullStrip() {
; CHECK-NEXT: ret i8* null
  %b2 = call i8* @llvm.strip.invariant.group.p0i8(i8* null)
  ret i8* %b2
}

; CHECK-LABEL: define i8* @dontSimplifyNullStripNonNullOpt()
define i8* @dontSimplifyNullStripNonNullOpt() #0 {
; CHECK-NEXT: call i8* @llvm.strip.invariant.group.p0i8(i8* null)
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

; CHECK-LABEL: define i8* @simplifyLaunderOfLaunder(
define i8* @simplifyLaunderOfLaunder(i8* %a) {
; CHECK:   call i8* @llvm.launder.invariant.group.p0i8(i8* %a)
; CHECK-NOT: llvm.launder.invariant.group
  %a2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a)
  %a3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a2)
  ret i8* %a3
}

; CHECK-LABEL: define i8* @simplifyStripOfLaunder(
define i8* @simplifyStripOfLaunder(i8* %a) {
; CHECK-NOT: llvm.launder.invariant.group
; CHECK:   call i8* @llvm.strip.invariant.group.p0i8(i8* %a)
  %a2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a)
  %a3 = call i8* @llvm.strip.invariant.group.p0i8(i8* %a2)
  ret i8* %a3
}

; CHECK-LABEL: define i1 @simplifyForCompare(
define i1 @simplifyForCompare(i8* %a) {
  %a2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a)

  %a3 = call i8* @llvm.strip.invariant.group.p0i8(i8* %a2)
  %b2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %a)
  %c = icmp eq i8* %a3, %b2
; CHECK: ret i1 true
  ret i1 %c
}

; CHECK-LABEL: define i16* @skipWithDifferentTypes(
define i16* @skipWithDifferentTypes(i8* %a) {
  %a2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a)
  %c1 = bitcast i8* %a2 to i16*

  ; CHECK: %[[b:.*]] = call i8* @llvm.strip.invariant.group.p0i8(i8* %a)
  %a3 = call i16* @llvm.strip.invariant.group.p0i16(i16* %c1)
  ; CHECK-NEXT: %[[r:.*]] = bitcast i8* %[[b]] to i16*
  ; CHECK-NEXT: ret i16* %[[r]]
  ret i16* %a3
}

; CHECK-LABEL: define i16 addrspace(42)* @skipWithDifferentTypesAddrspace(
define i16 addrspace(42)* @skipWithDifferentTypesAddrspace(i8 addrspace(42)* %a) {
  %a2 = call i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)* %a)
  %c1 = bitcast i8 addrspace(42)* %a2 to i16 addrspace(42)*

  ; CHECK: %[[b:.*]] = call i8 addrspace(42)* @llvm.strip.invariant.group.p42i8(i8 addrspace(42)* %a)
  %a3 = call i16 addrspace(42)* @llvm.strip.invariant.group.p42i16(i16 addrspace(42)* %c1)
  ; CHECK-NEXT: %[[r:.*]] = bitcast i8 addrspace(42)* %[[b]] to i16 addrspace(42)*
  ; CHECK-NEXT: ret i16 addrspace(42)* %[[r]]
  ret i16 addrspace(42)* %a3
}

; CHECK-LABEL: define i16 addrspace(42)* @skipWithDifferentTypesDifferentAddrspace(
define i16 addrspace(42)* @skipWithDifferentTypesDifferentAddrspace(i8* %a) {
  %cast = addrspacecast i8* %a to i8 addrspace(42)*
  %a2 = call i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)* %cast)
  %c1 = bitcast i8 addrspace(42)* %a2 to i16 addrspace(42)*

  ; CHECK: %[[b:.*]] = call i8* @llvm.strip.invariant.group.p0i8(i8* %a)
  %a3 = call i16 addrspace(42)* @llvm.strip.invariant.group.p42i16(i16 addrspace(42)* %c1)
  ; CHECK-NEXT: %[[r:.*]] = bitcast i8* %[[b]] to i16*
  ; CHECK-NEXT: %[[r2:.*]] = addrspacecast i16* %[[r]] to i16 addrspace(42)*
  ; CHECK-NEXT: ret i16 addrspace(42)* %[[r2]]
  ret i16 addrspace(42)* %a3
}

declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8 addrspace(42)* @llvm.launder.invariant.group.p42i8(i8 addrspace(42)*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)
declare i8 addrspace(42)* @llvm.strip.invariant.group.p42i8(i8 addrspace(42)*)
declare i16* @llvm.strip.invariant.group.p0i16(i16* %c1)
declare i16 addrspace(42)* @llvm.strip.invariant.group.p42i16(i16 addrspace(42)* %c1)

attributes #0 = { null_pointer_is_valid }
