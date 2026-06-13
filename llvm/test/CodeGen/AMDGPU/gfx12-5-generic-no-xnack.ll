; RUN: not llc -mtriple=amdgcn -mcpu=gfx12-5-generic -mattr=+xnack -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX125-PLUS-XNACK %s
; RUN: not llc -mtriple=amdgcn -mcpu=gfx12-5-generic -mattr=-xnack -filetype=null %s 2>&1 | FileCheck --check-prefix=GFX125-MINUS-XNACK %s

; GFX125-PLUS-XNACK: target only supports xnack 'Any'; '+/-xnack' is not allowed
; GFX125-MINUS-XNACK: target only supports xnack 'Any'; '+/-xnack' is not allowed

define void @foo() {
  ret void
}
