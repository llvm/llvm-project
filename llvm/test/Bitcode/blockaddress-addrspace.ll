; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/global-use-good.ll -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/global-use-bad.ll -o /dev/null 2>&1 | FileCheck %t/global-use-bad.ll
; RUN: llvm-as %t/global-fwddecl-good.ll -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/global-fwddecl-bad.ll -o /dev/null 2>&1 | FileCheck %t/global-fwddecl-bad.ll
; RUN: llvm-as %t/return-fwddecl-good.ll  -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/return-fwddecl-bad.ll -o /dev/null 2>&1 | FileCheck %t/return-fwddecl-bad.ll
; RUN: llvm-as %t/return-self-good.ll  -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/return-self-bad.ll -o /dev/null 2>&1 | FileCheck %t/return-self-bad.ll
; RUN: not llvm-as %t/return-self-bad-2.ll -o /dev/null 2>&1 | FileCheck %t/return-self-bad-2.ll
; RUN: not llvm-as %t/return-unknown-fn-bad.ll -o /dev/null 2>&1 | FileCheck %t/return-unknown-fn-bad.ll
; RUN: llvm-as %t/call-fwddecl-good.ll  -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/call-fwddecl-bad.ll -o /dev/null 2>&1 | FileCheck %t/call-fwddecl-bad.ll
; RUN: llvm-as %t/phi-good.ll  -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/phi-bad.ll -o /dev/null 2>&1 | FileCheck %t/phi-bad.ll
; RUN: llvm-as %t/fwddecl-phi-good.ll  -o - | llvm-dis -o /dev/null
; RUN: not llvm-as %t/fwddecl-phi-bad.ll -o /dev/null 2>&1 | FileCheck %t/fwddecl-phi-bad.ll
; RUN: not llvm-as %t/bad-type-not-ptr.ll -o /dev/null 2>&1 | FileCheck %t/bad-type-not-ptr.ll
; RUN: not llvm-as %t/bad-type-not-i8-ptr.ll -o /dev/null 2>&1 | FileCheck %t/bad-type-not-i8-ptr.ll


;--- global-use-good.ll
target datalayout = "P2"
define void @fn_in_prog_as_implicit() {
  unreachable
bb:
  ret void
}
define void @fn_in_prog_as_explicit() addrspace(2) {
  unreachable
bb:
  ret void
}
define void @fn_in_other_as() addrspace(1) {
  unreachable
bb:
  ret void
}
@global1 = constant i8 addrspace(2)* blockaddress(@fn_in_prog_as_implicit, %bb)
@global2 = constant i8 addrspace(2)* blockaddress(@fn_in_prog_as_explicit, %bb)
@global3 = constant i8 addrspace(1)* blockaddress(@fn_in_other_as, %bb)

;--- global-use-bad.ll
define void @fn() addrspace(1) {
  unreachable
bb:
  ret void
}
@global1 = constant i8 addrspace(2)* blockaddress(@fn, %bb)
; CHECK: [[#@LINE-1]]:38: error: constant expression type mismatch: got type 'i8 addrspace(1)*' but expected 'i8 addrspace(2)*'

; Check that a global blockaddress of a forward-declared function
; uses the type of the global variable address space for the forward declaration
;--- global-fwddecl-good.ll
@global = constant i8 addrspace(2)* blockaddress(@fwddecl_in_prog_as, %bb)
define void @fwddecl_in_prog_as() addrspace(2) {
  unreachable
bb:
  ret void
}

;--- global-fwddecl-bad.ll
; This forward declaration does not match the actual function type so we should get an error:
@global = constant i8 addrspace(2)* blockaddress(@fwddecl_in_unexpected_as, %bb)
; CHECK: [[#@LINE-1]]:77: error: 'bb' defined with type 'i8 addrspace(1)*' but expected 'i8 addrspace(2)*'
define void @fwddecl_in_unexpected_as() addrspace(1) {
  unreachable
bb:
  ret void
}


; When returning blockaddresses of forward-declared functions we
; can also use the type of the variable.
;--- return-fwddecl-good.ll
define i8 addrspace(2)* @take_as2() {
  ret i8 addrspace(2)* blockaddress(@fwddecl_as2, %bb)
}
define i8 addrspace(1)* @take_as1() {
  ret i8 addrspace(1)* blockaddress(@fwddecl_as1, %bb)
}
define void @fwddecl_as1() addrspace(1) {
  unreachable
bb:
  ret void
}
define void @fwddecl_as2() addrspace(2) {
  unreachable
bb:
  ret void
}

;--- return-fwddecl-bad.ll
define i8 addrspace(2)* @take_bad() {
  ret i8 addrspace(2)* blockaddress(@fwddecl_as1, %bb)
  ; CHECK: [[#@LINE-1]]:51: error: 'bb' defined with type 'i8 addrspace(1)*' but expected 'i8 addrspace(2)*'
}
define void @fwddecl_as1() addrspace(1) {
  unreachable
bb:
  ret void
}

;--- return-self-good.ll
target datalayout = "P2"
define i8 addrspace(0)* @take_self_as0() addrspace(0) {
L1:
  br label %L2
L2:
  ret i8 addrspace(0)* blockaddress(@take_self_as0, %L3)
L3:
  unreachable
}
define i8 addrspace(2)* @take_self_prog_as() {
L1:
  br label %L2
L2:
  ret i8 addrspace(2)* blockaddress(@take_self_prog_as, %L3)
L3:
  unreachable
}
define i8 addrspace(1)* @take_self_as1() addrspace(1) {
L1:
  br label %L2
L2:
  ret i8 addrspace(1)* blockaddress(@take_self_as1, %L3)
L3:
  unreachable
}
define i8 addrspace(2)* @take_self_as2() addrspace(2) {
L1:
  br label %L2
L2:
  ret i8 addrspace(2)* blockaddress(@take_self_as2, %L3)
L3:
  unreachable
}

;--- return-self-bad.ll
target datalayout = "P2"
define i8 addrspace(2)* @take_self_bad() addrspace(1) {
L1:
  br label %L2
L2:
  ret i8 addrspace(2)* blockaddress(@take_self_bad, %L3)
  ; CHECK: [[#@LINE-1]]:24: error: constant expression type mismatch: got type 'i8 addrspace(1)*' but expected 'i8 addrspace(2)*'
L3:
  unreachable
}
;--- return-self-bad-2.ll
target datalayout = "P2"
define i8* @take_self_bad_prog_as() {
L1:
  br label %L2
L2:
  ret i8* blockaddress(@take_self_bad_prog_as, %L3)
  ; CHECK: [[#@LINE-1]]:11: error: constant expression type mismatch: got type 'i8 addrspace(2)*' but expected 'i8*'
L3:
  unreachable
}

;--- return-unknown-fn-bad.ll
target datalayout = "P2"
define i8 addrspace(1)* @return_unknown_fn() addrspace(1) {
  ret i8 addrspace(1)* blockaddress(@undefined, %bb)
  ; CHECK: [[#@LINE-1]]:37: error: expected function name in blockaddress
}


;--- call-fwddecl-good.ll
target datalayout = "P2"
define void @call_from_fn_in_as2() addrspace(2) {
  call addrspace(2) void bitcast (i8 addrspace(2)* blockaddress(@fwddecl_as2, %bb) to void () addrspace(2)*)()
  ret void
}
define void @call_from_fn_in_as1() addrspace(1) {
  call addrspace(1) void bitcast (i8 addrspace(1)* blockaddress(@fwddecl_as1, %bb) to void () addrspace(1)*)()
  ret void
}
define void @fwddecl_as2() addrspace(2) {
  unreachable
bb:
  ret void
}
define void @fwddecl_as1() addrspace(1) {
  unreachable
bb:
  ret void
}

;--- call-fwddecl-bad.ll
target datalayout = "P2"
define void @call_from_fn_in_as2_explicit() addrspace(2) {
  call addrspace(2) void bitcast (i8 addrspace(2)* blockaddress(@fwddecl_as1, %bb) to void () addrspace(2)*)()
  ; CHECK: [[#@LINE-1]]:79: error: 'bb' defined with type 'i8 addrspace(1)*' but expected 'i8 addrspace(2)*'
  ret void
}
define void @fwddecl_as1() addrspace(1) {
  unreachable
bb:
  ret void
}

;--- phi-good.ll
target datalayout = "P2"
define i8 addrspace(1)* @f1() addrspace(1) {
L1:
  br label %L3
L2:
  br label %L3
L3:
  %p = phi i8 addrspace(1)* [ blockaddress(@f1, %L4), %L2 ], [ null, %L1 ]
  ret i8 addrspace(1)* %p
L4:
  unreachable
}
define i8 addrspace(2)* @f2() {
L1:
  br label %L3
L2:
  br label %L3
L3:
  %p = phi i8 addrspace(2)* [ blockaddress(@f2, %L4), %L2 ], [ null, %L1 ]
  ret i8 addrspace(2)* %p
L4:
  unreachable
}

;--- phi-bad.ll
target datalayout = "P2"
define i8* @f() {
L1:
  br label %L3
L2:
  br label %L3
L3:
  %p = phi i8* [ blockaddress(@f, %L4), %L2 ], [ null, %L1 ]
  ; CHECK: [[#@LINE-1]]:18: error: constant expression type mismatch: got type 'i8 addrspace(2)*' but expected 'i8*'
  ret i8* %p
}

; A blockaddress function forward-declaration used in a phi node should
; create the forward declaration in the same address space as the current function
;--- fwddecl-phi-good.ll
define i8 addrspace(1)* @f() addrspace(1) {
L1:
  br label %L3
L2:
  br label %L3
L3:
  %p = phi i8 addrspace(1)* [ blockaddress(@fwddecl_as1, %bb), %L2 ], [ null, %L1 ]
  ret i8 addrspace(1)* %p
L4:
  unreachable
}
define void @fwddecl_as1() addrspace(1) {
  unreachable
bb:
  ret void
}

;--- fwddecl-phi-bad.ll
define i8 addrspace(2)* @f() addrspace(2) {
L1:
  br label %L3
L2:
  br label %L3
L3:
  %p = phi i8 addrspace(2)* [ blockaddress(@fwddecl_as1, %bb), %L2 ], [ null, %L1 ]
  ; CHECK: [[#@LINE-1]]:58: error: 'bb' defined with type 'i8 addrspace(1)*' but expected 'i8 addrspace(2)*'
  ret i8 addrspace(2)* %p
L4:
  unreachable
}
define void @fwddecl_as1() addrspace(1) {
  unreachable
bb:
  ret void
}

;--- bad-type-not-ptr.ll
@global = constant i8 blockaddress(@unknown_fn, %bb)
; CHECK: [[#@LINE-1]]:23: error: type of blockaddress must be a pointer and not 'i8'
;--- bad-type-not-i8-ptr.ll
@global = constant i32* blockaddress(@unknown_fn, %bb)
; CHECK: [[#@LINE-1]]:25: error: constant expression type mismatch: got type 'i8*' but expected 'i32*'
