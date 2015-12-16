; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test basic inline assembly.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: foo:
; CHECK-NEXT: .param i32{{$}}
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # $0 = aaa($0){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: return $0{{$}}
define i32 @foo(i32 %r) {
entry:
  %0 = tail call i32 asm sideeffect "# $0 = aaa($1)", "=r,r"(i32 %r) #0, !srcloc !0
  ret i32 %0
}

; CHECK-LABEL: bar:
; CHECK-NEXT: .param i32, i32{{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # 0($1) = bbb(0($0)){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: return{{$}}
define void @bar(i32* %r, i32* %s) {
entry:
  tail call void asm sideeffect "# $0 = bbb($1)", "=*m,*m"(i32* %s, i32* %r) #0, !srcloc !1
  ret void
}

; CHECK-LABEL: imm:
; CHECK-NEXT: .result i32{{$}}
; CHECK-NEXT: .local i32{{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # $0 = ccc(42){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: return $0{{$}}
define i32 @imm() {
entry:
  %0 = tail call i32 asm sideeffect "# $0 = ccc($1)", "=r,i"(i32 42) #0, !srcloc !2
  ret i32 %0
}

; CHECK-LABEL: foo_i64:
; CHECK-NEXT: .param i64{{$}}
; CHECK-NEXT: .result i64{{$}}
; CHECK-NEXT: #APP{{$}}
; CHECK-NEXT: # $0 = aaa($0){{$}}
; CHECK-NEXT: #NO_APP{{$}}
; CHECK-NEXT: return $0{{$}}
define i64 @foo_i64(i64 %r) {
entry:
  %0 = tail call i64 asm sideeffect "# $0 = aaa($1)", "=r,r"(i64 %r) #0, !srcloc !0
  ret i64 %0
}

; CHECK-LABEL: X_i16:
; CHECK: foo $1{{$}}
; CHECK: i32.store16 $discard=, 0($0), $1{{$}}
define void @X_i16(i16 * %t) {
  call void asm sideeffect "foo $0", "=*X,~{dirflag},~{fpsr},~{flags},~{memory}"(i16* %t)
  ret void
}

; CHECK-LABEL: X_ptr:
; CHECK: foo $1{{$}}
; CHECK: i32.store $discard=, 0($0), $1{{$}}
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

attributes #0 = { nounwind }

!0 = !{i32 47}
!1 = !{i32 145}
!2 = !{i32 231}
