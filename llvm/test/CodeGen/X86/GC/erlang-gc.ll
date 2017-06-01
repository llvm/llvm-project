; RUN: llc -mtriple=x86_64-linux-gnu < %s | FileCheck %s --check-prefix=CHECK64
; RUN: llc -mtriple=i686-linux-gnu < %s | FileCheck %s --check-prefix=CHECK32

define i32 @main(i32 %x) nounwind gc "erlang" {
  %puts = tail call i32 @foo(i32 %x)
  ret i32 0

; CHECK64:      .section  .note.gc,"",@progbits
; CHECK64-NEXT: .p2align 3
; CHECK64-NEXT: .short 1      # safe point count
; CHECK64-NEXT: .long  .Ltmp0 # safe point address
; CHECK64-NEXT: .short 1      # stack frame size (in words)
; CHECK64-NEXT: .short 0      # stack arity
; CHECK64-NEXT: .short 0      # live root count

; CHECK32:      .section  .note.gc,"",@progbits
; CHECK32-NEXT: .p2align 2
; CHECK32-NEXT: .short 1      # safe point count
; CHECK32-NEXT: .long  .Ltmp0 # safe point address
; CHECK32-NEXT: .short 3      # stack frame size (in words)
; CHECK32-NEXT: .short 0      # stack arity
; CHECK32-NEXT: .short 0      # live root count
}

declare i32 @foo(i32)
