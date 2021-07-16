; RUN: not llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -no-integrated-as

; Test basic inline assembly "m" operands, which are unsupported. Pass
; -no-integrated-as since these aren't actually valid assembly syntax.

target triple = "wasm32-unknown-unknown"

define void @bar(i32* %r, i32* %s) {
entry:
  tail call void asm sideeffect "# $0 = bbb($1)", "=*m,*m"(i32* %s, i32* %r) #0, !srcloc !1
  ret void
}
