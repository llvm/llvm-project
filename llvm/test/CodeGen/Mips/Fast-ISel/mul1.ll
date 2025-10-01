; RUN: llc < %s -mtriple=mipsel -mcpu=mips32 -O0 -relocation-model=pic
; RUN: llc < %s -mtriple=mipsel -mcpu=mips32r2 -O0 -relocation-model=pic

; The test is just to make sure it is able to allocate
; registers for this example. There was an issue with allocating AC0
; after a mul instruction.

declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)

define i32 @foo(i32 %a, i32 %b)  {
entry:
  %0 = mul i32 %a, %b
  %1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %0, i32 %b)
  %2 = extractvalue { i32, i1 } %1, 0
  ret i32 %2
}
