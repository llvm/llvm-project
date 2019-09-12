; RUN: llc -o - %s -mtriple=arm64-apple-ios -O2 | FileCheck %s
; RUN: llc -o - %s -mtriple=arm64_32-apple-watchos -O2 | FileCheck %s
; RUN: llc -o - %s -mtriple=arm64-linux-gnu -O2 | FileCheck %s --check-prefix=CHECK-ELF

; CHECK-ELF-NOT: .loh
; CHECK-ELF-NOT: AdrpAdrp
; CHECK-ELF-NOT: AdrpAdd
; CHECK-ELF-NOT: AdrpLdrGot

@a = internal unnamed_addr global i32 0, align 4
@b = external global i32

; Function Attrs: noinline nounwind ssp
define void @foo(i32 %t) {
entry:
  %tmp = load i32, i32* @a, align 4
  %add = add nsw i32 %tmp, %t
  store i32 %add, i32* @a, align 4
  ret void
}

; Function Attrs: nounwind ssp
; Testcase for <rdar://problem/15438605>, AdrpAdrp reuse is valid only when the first adrp
; dominates the second.
; The first adrp comes from the loading of 'a' and the second the loading of 'b'.
; 'a' is loaded in if.then, 'b' in if.end4, if.then does not dominates if.end4.
; CHECK-LABEL: _test
; CHECK: ret
; CHECK-NOT: .loh AdrpAdrp
define i32 @test(i32 %t) {
entry:
  %cmp = icmp sgt i32 %t, 5
  br i1 %cmp, label %if.then, label %if.end4

if.then:                                          ; preds = %entry
  %tmp = load i32, i32* @a, align 4
  %add = add nsw i32 %tmp, %t
  %cmp1 = icmp sgt i32 %add, 12
  br i1 %cmp1, label %if.then2, label %if.end4

if.then2:                                         ; preds = %if.then
  tail call void @foo(i32 %add)
  %tmp1 = load i32, i32* @a, align 4
  br label %if.end4

if.end4:                                          ; preds = %if.then2, %if.then, %entry
  %t.addr.0 = phi i32 [ %tmp1, %if.then2 ], [ %t, %if.then ], [ %t, %entry ]
  %tmp2 = load i32, i32* @b, align 4
  %add5 = add nsw i32 %tmp2, %t.addr.0
  tail call void @foo(i32 %add5)
  %tmp3 = load i32, i32* @b, align 4
  %add6 = add nsw i32 %tmp3, %t.addr.0
  ret i32 %add6
}

@C = common global i32 0, align 4

; Check that we catch AdrpLdrGotLdr case when we have a simple chain:
; adrp -> ldrgot -> ldr.
; CHECK-LABEL: _getC
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _C@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _C@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i32 @getC() {
  %res = load i32, i32* @C, align 4
  ret i32 %res
}

; LDRSW supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExtC
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _C@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _C@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsw x0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i64 @getSExtC() {
  %res = load i32, i32* @C, align 4
  %sextres = sext i32 %res to i64
  ret i64 %sextres
}

; It may not be safe to fold the literal in the load if the address is
; used several times.
; Make sure we emit AdrpLdrGot for those.
; CHECK-LABEL: _getSeveralC
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _C@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _C@GOTPAGEOFF]
; CHECK-NEXT: ldr [[LOAD:w[0-9]+]], [x[[LDRGOT_REG]]]
; CHECK-NEXT: add [[ADD:w[0-9]+]], [[LOAD]], w0
; CHECK-NEXT: str [[ADD]], [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGot [[ADRP_LABEL]], [[LDRGOT_LABEL]]
define void @getSeveralC(i32 %t) {
entry:
  %tmp = load i32, i32* @C, align 4
  %add = add nsw i32 %tmp, %t
  store i32 %add, i32* @C, align 4
  ret void
}

; Make sure we catch that:
; adrp -> ldrgot -> str.
; CHECK-LABEL: _setC
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _C@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _C@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define void @setC(i32 %t) {
entry:
  store i32 %t, i32* @C, align 4
  ret void
}

; Perform the same tests for internal global and a displacement
; in the addressing mode.
; Indeed we will get an ADD for those instead of LOADGot.
@InternalC = internal global i32 0, align 4

; Check that we catch AdrpAddLdr case when we have a simple chain:
; adrp -> add -> ldr.
; CHECK-LABEL: _getInternalCPlus4
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: [[ADDGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: add [[ADDGOT_REG:x[0-9]+]], [[ADRP_REG]], _InternalC@PAGEOFF
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr w0, {{\[}}[[ADDGOT_REG]], #16]
; CHECK-NEXT: ret
; CHECK: .loh AdrpAddLdr [[ADRP_LABEL]], [[ADDGOT_LABEL]], [[LDR_LABEL]]
define i32 @getInternalCPlus4() {
  %addr = getelementptr inbounds i32, i32* @InternalC, i32 4
  %res = load i32, i32* %addr, align 4
  ret i32 %res
}

; LDRSW supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExtInternalCPlus4
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: [[ADDGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: add [[ADDGOT_REG:x[0-9]+]], [[ADRP_REG]], _InternalC@PAGEOFF
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsw x0, {{\[}}[[ADDGOT_REG]], #16]
; CHECK-NEXT: ret
; CHECK: .loh AdrpAddLdr [[ADRP_LABEL]], [[ADDGOT_LABEL]], [[LDR_LABEL]]
define i64 @getSExtInternalCPlus4() {
  %addr = getelementptr inbounds i32, i32* @InternalC, i32 4
  %res = load i32, i32* %addr, align 4
  %sextres = sext i32 %res to i64
  ret i64 %sextres
}

; It may not be safe to fold the literal in the load if the address is
; used several times.
; Make sure we emit AdrpAdd for those.
; CHECK-LABEL: _getSeveralInternalCPlus4
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: [[ADDGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: add [[ADDGOT_REG:x[0-9]+]], [[ADRP_REG]], _InternalC@PAGEOFF
; CHECK-NEXT: ldr [[LOAD:w[0-9]+]], {{\[}}[[ADDGOT_REG]], #16]
; CHECK-NEXT: add [[ADD:w[0-9]+]], [[LOAD]], w0
; CHECK-NEXT: str [[ADD]], {{\[}}[[ADDGOT_REG]], #16]
; CHECK-NEXT: ret
; CHECK: .loh AdrpAdd [[ADRP_LABEL]], [[ADDGOT_LABEL]]
define void @getSeveralInternalCPlus4(i32 %t) {
entry:
  %addr = getelementptr inbounds i32, i32* @InternalC, i32 4
  %tmp = load i32, i32* %addr, align 4
  %add = add nsw i32 %tmp, %t
  store i32 %add, i32* %addr, align 4
  ret void
}

; Make sure we catch that:
; adrp -> add -> str.
; CHECK-LABEL: _setInternalCPlus4
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: [[ADDGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: add [[ADDGOT_REG:x[0-9]+]], [[ADRP_REG]], _InternalC@PAGEOFF
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str w0, {{\[}}[[ADDGOT_REG]], #16]
; CHECK-NEXT: ret
; CHECK: .loh AdrpAddStr [[ADRP_LABEL]], [[ADDGOT_LABEL]], [[LDR_LABEL]]
define void @setInternalCPlus4(i32 %t) {
entry:
  %addr = getelementptr inbounds i32, i32* @InternalC, i32 4
  store i32 %t, i32* %addr, align 4
  ret void
}

; Check that we catch AdrpAddLdr case when we have a simple chain:
; adrp -> ldr.
; CHECK-LABEL: _getInternalC
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr w0, {{\[}}[[ADRP_REG]], _InternalC@PAGEOFF]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdr [[ADRP_LABEL]], [[LDR_LABEL]]
define i32 @getInternalC() {
  %res = load i32, i32* @InternalC, align 4
  ret i32 %res
}

; LDRSW supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExtInternalC
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsw x0, {{\[}}[[ADRP_REG]], _InternalC@PAGEOFF]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdr [[ADRP_LABEL]], [[LDR_LABEL]]
define i64 @getSExtInternalC() {
  %res = load i32, i32* @InternalC, align 4
  %sextres = sext i32 %res to i64
  ret i64 %sextres
}

; It may not be safe to fold the literal in the load if the address is
; used several times.
; Make sure we do not catch anything here. We have a adrp alone,
; there is not much we can do about it.
; CHECK-LABEL: _getSeveralInternalC
; CHECK: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: ldr [[LOAD:w[0-9]+]], {{\[}}[[ADRP_REG]], _InternalC@PAGEOFF]
; CHECK-NEXT: add [[ADD:w[0-9]+]], [[LOAD]], w0
; CHECK-NEXT: str [[ADD]], {{\[}}[[ADRP_REG]], _InternalC@PAGEOFF]
; CHECK-NEXT: ret
define void @getSeveralInternalC(i32 %t) {
entry:
  %tmp = load i32, i32* @InternalC, align 4
  %add = add nsw i32 %tmp, %t
  store i32 %add, i32* @InternalC, align 4
  ret void
}

; Make sure we do not catch anything when:
; adrp -> str.
; We cannot fold anything in the str at this point.
; Indeed, strs do not support litterals.
; CHECK-LABEL: _setInternalC
; CHECK: adrp [[ADRP_REG:x[0-9]+]], _InternalC@PAGE
; CHECK-NEXT: str w0, {{\[}}[[ADRP_REG]], _InternalC@PAGEOFF]
; CHECK-NEXT: ret
define void @setInternalC(i32 %t) {
entry:
  store i32 %t, i32* @InternalC, align 4
  ret void
}

; Now check other variant of loads/stores.

@D = common global i8 0, align 4

; LDRB does not support loading from a literal.
; Make sure we emit AdrpLdrGot and not AdrpLdrGotLdr for those.
; CHECK-LABEL: _getD
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _D@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _D@GOTPAGEOFF]
; CHECK-NEXT: ldrb w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGot [[ADRP_LABEL]], [[LDRGOT_LABEL]]
define i8 @getD() {
  %res = load i8, i8* @D, align 4
  ret i8 %res
}

; CHECK-LABEL: _setD
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _D@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _D@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: strb w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setD(i8 %t) {
  store i8 %t, i8* @D, align 4
  ret void
}

; LDRSB supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExtD
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _D@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _D@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsb w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i32 @getSExtD() {
  %res = load i8, i8* @D, align 4
  %sextres = sext i8 %res to i32
  ret i32 %sextres
}

; LDRSB supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExt64D
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _D@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _D@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsb x0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i64 @getSExt64D() {
  %res = load i8, i8* @D, align 4
  %sextres = sext i8 %res to i64
  ret i64 %sextres
}

@E = common global i16 0, align 4

; LDRH does not support loading from a literal.
; Make sure we emit AdrpLdrGot and not AdrpLdrGotLdr for those.
; CHECK-LABEL: _getE
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _E@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _E@GOTPAGEOFF]
; CHECK-NEXT: ldrh w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGot [[ADRP_LABEL]], [[LDRGOT_LABEL]]
define i16 @getE() {
  %res = load i16, i16* @E, align 4
  ret i16 %res
}

; LDRSH supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExtE
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _E@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _E@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsh w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i32 @getSExtE() {
  %res = load i16, i16* @E, align 4
  %sextres = sext i16 %res to i32
  ret i32 %sextres
}

; CHECK-LABEL: _setE
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _E@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _E@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: strh w0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setE(i16 %t) {
  store i16 %t, i16* @E, align 4
  ret void
}

; LDRSH supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getSExt64E
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _E@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _E@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldrsh x0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i64 @getSExt64E() {
  %res = load i16, i16* @E, align 4
  %sextres = sext i16 %res to i64
  ret i64 %sextres
}

@F = common global i64 0, align 4

; LDR supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getF
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _F@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _F@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr x0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define i64 @getF() {
  %res = load i64, i64* @F, align 4
  ret i64 %res
}

; CHECK-LABEL: _setF
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _F@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _F@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str x0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setF(i64 %t) {
  store i64 %t, i64* @F, align 4
  ret void
}

@G = common global float 0.0, align 4

; LDR float supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getG
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _G@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _G@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr s0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define float @getG() {
  %res = load float, float* @G, align 4
  ret float %res
}

; CHECK-LABEL: _setG
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _G@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _G@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str s0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setG(float %t) {
  store float %t, float* @G, align 4
  ret void
}

@H = common global half 0.0, align 4

; LDR half supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getH
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _H@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _H@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr h0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define half @getH() {
  %res = load half, half* @H, align 4
  ret half %res
}

; CHECK-LABEL: _setH
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _H@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _H@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str h0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setH(half %t) {
  store half %t, half* @H, align 4
  ret void
}

@I = common global double 0.0, align 4

; LDR double supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getI
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _I@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _I@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr d0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define double @getI() {
  %res = load double, double* @I, align 4
  ret double %res
}

; CHECK-LABEL: _setI
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _I@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _I@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str d0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setI(double %t) {
  store double %t, double* @I, align 4
  ret void
}

@J = common global <2 x i32> <i32 0, i32 0>, align 4

; LDR 64-bit vector supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getJ
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _J@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _J@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr d0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define <2 x i32> @getJ() {
  %res = load <2 x i32>, <2 x i32>* @J, align 4
  ret <2 x i32> %res
}

; CHECK-LABEL: _setJ
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _J@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _J@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str d0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setJ(<2 x i32> %t) {
  store <2 x i32> %t, <2 x i32>* @J, align 4
  ret void
}

@K = common global <4 x i32> <i32 0, i32 0, i32 0, i32 0>, align 4

; LDR 128-bit vector supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getK
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _K@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _K@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr q0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define <4 x i32> @getK() {
  %res = load <4 x i32>, <4 x i32>* @K, align 4
  ret <4 x i32> %res
}

; CHECK-LABEL: _setK
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _K@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _K@GOTPAGEOFF]
; CHECK-NEXT: [[STR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: str q0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotStr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[STR_LABEL]]
define void @setK(<4 x i32> %t) {
  store <4 x i32> %t, <4 x i32>* @K, align 4
  ret void
}

@L = common global <1 x i8> <i8 0>, align 4

; LDR 8-bit vector supports loading from a literal.
; Make sure we emit AdrpLdrGotLdr for those.
; CHECK-LABEL: _getL
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _L@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _L@GOTPAGEOFF]
; CHECK-NEXT: [[LDR_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr b0, [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGotLdr [[ADRP_LABEL]], [[LDRGOT_LABEL]], [[LDR_LABEL]]
define <1 x i8> @getL() {
  %res = load <1 x i8>, <1 x i8>* @L, align 4
  ret <1 x i8> %res
}

; CHECK-LABEL: _setL
; CHECK: [[ADRP_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: adrp [[ADRP_REG:x[0-9]+]], _L@GOTPAGE
; CHECK-NEXT: [[LDRGOT_LABEL:Lloh[0-9]+]]:
; CHECK-NEXT: ldr {{[xw]}}[[LDRGOT_REG:[0-9]+]], {{\[}}[[ADRP_REG]], _L@GOTPAGEOFF]
; CHECK-NEXT: ; kill
; Ultimately we should generate str b0, but right now, we match the vector
; variant which does not allow to fold the immediate into the store.
; CHECK-NEXT: st1.b { v0 }[0], [x[[LDRGOT_REG]]]
; CHECK-NEXT: ret
; CHECK: .loh AdrpLdrGot [[ADRP_LABEL]], [[LDRGOT_LABEL]]
define void @setL(<1 x i8> %t) {
  store <1 x i8> %t, <1 x i8>* @L, align 4
  ret void
}

; Make sure we do not assert when we do not track
; all the aliases of a tuple register.
; Indeed the tuple register can be tracked because of
; one of its element, but the other elements of the tuple
; do not need to be tracked and we used to assert on that.
; Note: The test case is fragile in the sense that we need
; a tuple register to appear in the lowering. Thus, the target
; cpu is required to have the problem reproduced.
; CHECK-LABEL: _uninterestingSub
; CHECK: [[LOH_LABEL0:Lloh[0-9]+]]:
; CHECK: adrp [[ADRP_REG:x[0-9]+]], [[CONSTPOOL:lCPI[0-9]+_[0-9]+]]@PAGE
; CHECK: [[LOH_LABEL1:Lloh[0-9]+]]:
; CHECK: ldr q[[IDX:[0-9]+]], {{\[}}[[ADRP_REG]], [[CONSTPOOL]]@PAGEOFF]
; The tuple comes from the next instruction.
; CHECK: ext.16b v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, #1
; CHECK: ret
; CHECK: .loh AdrpLdr [[LOH_LABEL0]], [[LOH_LABEL1]]
define void @uninterestingSub(i8* nocapture %row) #0 {
  %tmp = bitcast i8* %row to <16 x i8>*
  %tmp1 = load <16 x i8>, <16 x i8>* %tmp, align 16
  %vext43 = shufflevector <16 x i8> <i8 undef, i8 16, i8 15, i8 14, i8 13, i8 12, i8 11, i8 10, i8 9, i8 8, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2>, <16 x i8> %tmp1, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  %add.i.414 = add <16 x i8> zeroinitializer, %vext43
  store <16 x i8> %add.i.414, <16 x i8>* %tmp, align 16
  %add.ptr51 = getelementptr inbounds i8, i8* %row, i64 16
  %tmp2 = bitcast i8* %add.ptr51 to <16 x i8>*
  %tmp3 = load <16 x i8>, <16 x i8>* %tmp2, align 16
  %tmp4 = bitcast i8* undef to <16 x i8>*
  %tmp5 = load <16 x i8>, <16 x i8>* %tmp4, align 16
  %vext157 = shufflevector <16 x i8> %tmp3, <16 x i8> %tmp5, <16 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16>
  %add.i.402 = add <16 x i8> zeroinitializer, %vext157
  store <16 x i8> %add.i.402, <16 x i8>* %tmp4, align 16
  ret void
}

@.str.89 = external unnamed_addr constant [12 x i8], align 1
@.str.90 = external unnamed_addr constant [5 x i8], align 1
; CHECK-LABEL: test_r274582
define void @test_r274582(double %x) {
entry:
  br i1 undef, label %if.then.i, label %if.end.i
if.then.i:
  ret void
if.end.i:
; CHECK: .loh AdrpLdrGot
; CHECK: .loh AdrpLdrGot
; CHECK: .loh AdrpAdrp
; CHECK: .loh AdrpLdr
  %mul = fmul double %x, 1.000000e-06
  %add = fadd double %mul, %mul
  %sub = fsub double %add, %add
  call void (i8*, ...) @callee(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.89, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.90, i64 0, i64 0), double %sub)
  unreachable
}
declare void @callee(i8* nocapture readonly, ...) 

attributes #0 = { "target-cpu"="cyclone" }
