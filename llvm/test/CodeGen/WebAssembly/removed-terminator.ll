; RUN: llc -O0 < %s

target triple = "wasm32-unknown-unknown"

define void @test(i1 %x) {
  %y = xor i1 %x, true
  ; We now do a limited amount of register stackification in RegStackify even in
  ; -O0, so its operand (%y) is stackified. But this terminator will be removed
  ; in CFGSort after that. We need to make sure we unstackify %y so that it can
  ; be dropped in ExplicitLocals.
  br i1 %y, label %exit, label %exit

exit:
  ret void
}
