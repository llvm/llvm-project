; RUN: llc -O0 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=CONST
; RUN: llc -O2 -mtriple=loongarch32 -verify-machineinstrs < %s | FileCheck %s --check-prefix=CONST
; RUN: llc -O1 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O2 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O3 -mtriple=loongarch64 -verify-machineinstrs < %s | FileCheck %s

; ANDN combines for XOR and masked SETCC patterns.
; CONST checks constant masks at -O0 on LA64 and -O2 on LA32.
; CHECK covers the LA64 -O1, -O2, and -O3 code generation paths.

; XOR/AND patterns, including operand commutations and i32 values.
define i64 @and_xor_0(i64 %x, i64 %m) {
; CHECK-LABEL: and_xor_0:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i64 %x, %m
  %r = xor i64 %a, %m
  ret i64 %r
}

define i64 @and_xor_1(i64 %x, i64 %m) {
; CHECK-LABEL: and_xor_1:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i64 %m, %x
  %r = xor i64 %a, %m
  ret i64 %r
}

define i64 @and_xor_2(i64 %x, i64 %m) {
; CHECK-LABEL: and_xor_2:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i64 %x, %m
  %r = xor i64 %m, %a
  ret i64 %r
}

define i64 @and_xor_3(i64 %x, i64 %m) {
; CHECK-LABEL: and_xor_3:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i64 %m, %x
  %r = xor i64 %m, %a
  ret i64 %r
}

define i64 @and_xor_different_masks(i64 %x, i64 %m1, i64 %m2) {
; CHECK-LABEL: and_xor_different_masks:
; CHECK:       {{[[:space:]]and[[:space:]]}}
; CHECK:       {{[[:space:]]xor[[:space:]]}}
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
  %a = and i64 %x, %m1
  %r = xor i64 %a, %m2
  ret i64 %r
}

define i64 @and_xor_multiuse(i64 %x, i64 %m, ptr %sink) {
; CHECK-LABEL: and_xor_multiuse:
; CHECK:       {{[[:space:]]and[[:space:]]}}
; CHECK:       {{[[:space:]]xor[[:space:]]}}
; CHECK:       st.d
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
  %a = and i64 %x, %m
  store i64 %a, ptr %sink, align 8
  %r = xor i64 %a, %m
  ret i64 %r
}

; LA64 legalizes these i32 operations into GPR-width operations. Match all
; commutations before legalization so unsigned 32-bit bit tests also expose
; ANDN to instruction selection.
define i32 @and_xor_i32_0(i32 %x, i32 %m) {
; CHECK-LABEL: and_xor_i32_0:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i32 %x, %m
  %r = xor i32 %a, %m
  ret i32 %r
}

define i32 @and_xor_i32_1(i32 %x, i32 %m) {
; CHECK-LABEL: and_xor_i32_1:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i32 %m, %x
  %r = xor i32 %a, %m
  ret i32 %r
}

define i32 @and_xor_i32_2(i32 %x, i32 %m) {
; CHECK-LABEL: and_xor_i32_2:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i32 %x, %m
  %r = xor i32 %m, %a
  ret i32 %r
}

define i32 @and_xor_i32_3(i32 %x, i32 %m) {
; CHECK-LABEL: and_xor_i32_3:
; CHECK:       andn
; CHECK-NEXT:  ret
  %a = and i32 %m, %x
  %r = xor i32 %m, %a
  ret i32 %r
}

define i1 @and_xor_i32_eq_zero(i32 %x, i32 %m) {
; CHECK-LABEL: and_xor_i32_eq_zero:
; CHECK:       andn
; CHECK:       sltui
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
  %a = and i32 %x, %m
  %r = xor i32 %a, %m
  %cmp = icmp eq i32 %r, 0
  ret i1 %cmp
}

; This is the shape produced by source-level (x & mask) == mask. There is no
; XOR node in the input DAG; the generic equality pattern would otherwise
; introduce XOR only while selecting machine instructions.
define i1 @and_eq_mask(i64 %x, i64 %m) {
; CHECK-LABEL: and_eq_mask:
; CHECK:       andn
; CHECK-NEXT:  sltui
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
  %a = and i64 %x, %m
  %cmp = icmp eq i64 %a, %m
  ret i1 %cmp
}

define i1 @mask_eq_commuted_and_i32(i32 %x, i32 %m) {
; CHECK-LABEL: mask_eq_commuted_and_i32:
; CHECK:       andn
; CHECK:       sltui
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
  %a = and i32 %m, %x
  %cmp = icmp eq i32 %m, %a
  ret i1 %cmp
}

; Masked equality and inequality patterns.
; Equality and inequality must both fold for a variable mask.
define i1 @and_eq_variable_mask(i64 %x, i64 %m) {
; CHECK-LABEL: and_eq_variable_mask:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]], $a1, $a0
; CHECK-NEXT:  sltui {{\$[a-z0-9]+}}, [[MISSING]], 1
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
  %a = and i64 %x, %m
  %cmp = icmp eq i64 %a, %m
  ret i1 %cmp
}

; SETNE must expose ANDN before instruction selection introduces a machine XOR.
define i1 @and_ne_mask(i64 %x, i64 %m) {
; CHECK-LABEL: and_ne_mask:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]], $a1, $a0
; CHECK-NEXT:  sltu {{\$[a-z0-9]+}}, $zero, [[MISSING]]
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
  %a = and i64 %x, %m
  %cmp = icmp ne i64 %a, %m
  ret i1 %cmp
}

; Commute both the AND and the comparison, including LA64's promoted i32 case.
define i1 @mask_ne_commuted_and_i32(i32 signext %x, i32 signext %m) {
; CHECK-LABEL: mask_ne_commuted_and_i32:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]],
; CHECK-NEXT:  sltu {{\$[a-z0-9]+}}, $zero, [[MISSING]]
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
  %a = and i32 %m, %x
  %cmp = icmp ne i32 %m, %a
  ret i1 %cmp
}

; Keep the equality counterpart of the class-test mask covered at O0 too.
define i32 @and_eq_mask_16385_zext(i32 %x) {
; CONST-LABEL: and_eq_mask_16385_zext:
; CONST:       andn [[EQ_MISSING:\$[a-z0-9]+]],
; CONST:       sltui {{\$[a-z0-9]+}}, [[EQ_MISSING]], 1
; CONST:       ret
; CHECK-LABEL: and_eq_mask_16385_zext:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]],
; CHECK-NEXT:  sltui {{\$[a-z0-9]+}}, [[MISSING]], 1
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
  %a = and i32 %x, 16385
  %cmp = icmp eq i32 %a, 16385
  %result = zext i1 %cmp to i32
  ret i32 %result
}

; Unlike the variable-mask cases, this noncontiguous constant mask previously
; survived as AND + XOR + SLTU. Keep the zero-extension used by class tests.
define i32 @and_ne_mask_16385_zext(i32 %x) {
; CONST-LABEL: and_ne_mask_16385_zext:
; CONST:       andn [[NE_MISSING:\$[a-z0-9]+]],
; CONST:       sltu {{\$[a-z0-9]+}}, $zero, [[NE_MISSING]]
; CONST:       ret
; CHECK-LABEL: and_ne_mask_16385_zext:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]],
; CHECK-NEXT:  sltu {{\$[a-z0-9]+}}, $zero, [[MISSING]]
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
  %a = and i32 %x, 16385
  %cmp = icmp ne i32 %a, 16385
  %result = zext i1 %cmp to i32
  ret i32 %result
}

; The zext SETNE result is also compared against an existing state value before
; branching, rather than returned directly.
define void @and_ne_mask_16385_state(i32 %x, i32 %state, ptr %sink) {
; CHECK-LABEL: and_ne_mask_16385_state:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]],
; CHECK-NEXT:  sltu {{\$[a-z0-9]+}}, $zero, [[MISSING]]
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       {{b(eq|ne)}}
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
  %a = and i32 %x, 16385
  %cmp = icmp ne i32 %a, 16385
  %result = zext i1 %cmp to i32
  %same = icmp eq i32 %state, %result
  br i1 %same, label %hit, label %exit

hit:
  store i8 1, ptr %sink, align 1
  br label %exit

exit:
  ret void
}

; A reduced character-class scan: load the table entry, test the same mask, and
; compare with loop-carried state. The state update may use XORI, but the class
; test itself needs no XOR instruction.
define ptr @find_class_16385(ptr %start, ptr %end, ptr %table, i32 %initial_state) {
; CHECK-LABEL: find_class_16385:
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ld.bu
; CHECK:       {{[[:space:]]ldx?\.w[[:space:]]}}
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       andn [[MISSING:\$[a-z0-9]+]],
; CHECK-NEXT:  sltu {{\$[a-z0-9]+}}, $zero, [[MISSING]]
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       {{b(eq|ne)}}
; CHECK-NOT:   {{[[:space:]]xor[[:space:]]}}
; CHECK:       ret
entry:
  %nonempty = icmp ult ptr %start, %end
  br i1 %nonempty, label %loop, label %exit

loop:
  %current = phi ptr [ %start, %entry ], [ %next, %advance ]
  %state = phi i32 [ %initial_state, %entry ], [ %next_state, %advance ]
  %char = load i8, ptr %current, align 1
  %index = zext i8 %char to i64
  %entry_addr = getelementptr inbounds i32, ptr %table, i64 %index
  %bits = load i32, ptr %entry_addr, align 4
  %masked = and i32 %bits, 16385
  %missing = icmp ne i32 %masked, 16385
  %missing32 = zext i1 %missing to i32
  %same = icmp eq i32 %state, %missing32
  br i1 %same, label %hit, label %advance

advance:
  %next_state = xor i32 %state, 1
  %next = getelementptr inbounds i8, ptr %current, i64 1
  %more = icmp ult ptr %next, %end
  br i1 %more, label %loop, label %exit

hit:
  ret ptr %current

exit:
  ret ptr %end
}

; Do not discard another use of the masked value while rewriting SETNE.
define i1 @and_ne_mask_multiuse(i64 %x, i64 %m, ptr %sink) {
; CHECK-LABEL: and_ne_mask_multiuse:
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK-DAG:   {{[[:space:]]and[[:space:]]}}
; CHECK-DAG:   {{[[:space:]]xor[[:space:]]}}
; CHECK-DAG:   st.d
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       ret
  %a = and i64 %x, %m
  store i64 %a, ptr %sink, align 8
  %cmp = icmp ne i64 %a, %m
  ret i1 %cmp
}

define i1 @and_ne_different_masks(i64 %x, i64 %m1, i64 %m2) {
; CHECK-LABEL: and_ne_different_masks:
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       {{[[:space:]]and[[:space:]]}}
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       {{[[:space:]]xor[[:space:]]}}
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       ret
  %a = and i64 %x, %m1
  %cmp = icmp ne i64 %a, %m2
  ret i1 %cmp
}

; Signed order comparisons do not have the equality/inequality identity.
define i1 @and_slt_mask(i64 %x, i64 %m) {
; CHECK-LABEL: and_slt_mask:
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       {{[[:space:]]and[[:space:]]}}
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       {{[[:space:]]slt[[:space:]]}}
; CHECK-NOT:   {{[[:space:]]andn[[:space:]]}}
; CHECK:       ret
  %a = and i64 %x, %m
  %cmp = icmp slt i64 %a, %m
  ret i1 %cmp
}
