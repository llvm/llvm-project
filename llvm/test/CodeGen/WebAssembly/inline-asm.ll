; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers -no-integrated-as -verify-machineinstrs | FileCheck %s

; Test basic inline assembly. Pass -no-integrated-as since these aren't
; actually valid assembly syntax.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: foo:
; CHECK-NEXT: .functype foo (i32) -> (i32){{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # 0 = aaa(0){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: local.get $push0=, 0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @foo(i32 %r) {
entry:
  %0 = tail call i32 asm sideeffect "# $0 = aaa($1)", "=r,r"(i32 %r) #0, !srcloc !0
  ret i32 %0
}

; CHECK-LABEL: imm:
; CHECK-NEXT: .functype imm () -> (i32){{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # 0 = ccc(42){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: local.get $push0=, 0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i32 @imm() {
entry:
  %0 = tail call i32 asm sideeffect "# $0 = ccc($1)", "=r,i"(i32 42) #0, !srcloc !2
  ret i32 %0
}

; CHECK-LABEL: foo_i64:
; CHECK-NEXT: .functype foo_i64 (i64) -> (i64){{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # 0 = aaa(0){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: local.get $push0=, 0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define i64 @foo_i64(i64 %r) {
entry:
  %0 = tail call i64 asm sideeffect "# $0 = aaa($1)", "=r,r"(i64 %r) #0, !srcloc !0
  ret i64 %0
}

; CHECK-LABEL: X_i16:
; CHECK: foo 1{{$}}
; CHECK: local.get $push[[S0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[S1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.store16 0($pop[[S0]]), $pop[[S1]]{{$}}
define void @X_i16(i16 * %t) {
  call void asm sideeffect "foo $0", "=*X,~{dirflag},~{fpsr},~{flags},~{memory}"(i16* %t)
  ret void
}

; CHECK-LABEL: X_ptr:
; CHECK: foo 1{{$}}
; CHECK: local.get $push[[S0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[S1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i32.store 0($pop[[S0]]), $pop[[S1]]{{$}}
define void @X_ptr(i16 ** %t) {
  call void asm sideeffect "foo $0", "=*X,~{dirflag},~{fpsr},~{flags},~{memory}"(i16** %t)
  ret void
}

; CHECK-LABEL: funcname:
; CHECK: foo funcname{{$}}
define void @funcname() {
  tail call void asm sideeffect "foo $0", "i"(void ()* nonnull @funcname) #0, !srcloc !0
  ret void
}

; CHECK-LABEL: varname:
; CHECK: foo gv+37{{$}}
@gv = global [0 x i8] zeroinitializer
define void @varname() {
  tail call void asm sideeffect "foo $0", "i"(i8* getelementptr inbounds ([0 x i8], [0 x i8]* @gv, i64 0, i64 37)) #0, !srcloc !0
  ret void
}

; CHECK-LABEL: r_constraint
; CHECK:      i32.const $push[[S0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.set [[L0:[0-9]+]], $pop[[S0]]{{$}}
; CHECK-NEXT: i32.const $push[[S1:[0-9]+]]=, 37{{$}}
; CHECK-NEXT: local.set [[L1:[0-9]+]], $pop[[S1]]{{$}}
; CHECK:      foo [[L2:[0-9]+]], 1, [[L0]], [[L1]]{{$}}
; CHECK:      local.get $push{{[0-9]+}}=, [[L2]]{{$}}
define hidden i32 @r_constraint(i32 %a, i32 %y) {
entry:
  %z = bitcast i32 0 to i32
  %t0 = tail call i32 asm "foo $0, $1, $2, $3", "=r,r,r,r"(i32 %y, i32 %z, i32 37) #0, !srcloc !0
  ret i32 %t0
}

; CHECK-LABEL: tied_operands
; CHECK: local.get  $push0=, 0
; CHECK: return    $pop0
define i32 @tied_operands(i32 %var) {
entry:
  %ret = call i32 asm "", "=r,0"(i32 %var)
  ret i32 %ret
}

attributes #0 = { nounwind }

!0 = !{i32 47}
!1 = !{i32 145}
!2 = !{i32 231}
