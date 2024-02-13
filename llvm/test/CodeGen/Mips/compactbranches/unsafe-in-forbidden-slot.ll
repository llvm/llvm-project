; RUN: llc -march=mips64el -O0 -mcpu=mips64r6 < %s | FileCheck %s
; RUN: llc -march=mips64 -O0 -mcpu=mips64r6 < %s | FileCheck %s

@boo = global i32 0, align 4

; Function Attrs: nounwind
define void @_Z3foov() #0 {
entry:
  %0 = load volatile i32, ptr @boo, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  store volatile i32 1, ptr @boo, align 4
  br label %sw.epilog
; CHECK: beqzc
; CHECK-NEXT: nop
; CHECK-NEXT: # %bb.1
; CHECK-NEXT: j

sw.bb1:                                           ; preds = %entry, %entry
  store volatile i32 2, ptr @boo, align 4
  br label %sw.epilog
; CHECK: bnezc
; CHECK-NEXT: nop
; CHECK-NEXT: # %bb.3
; CHECK-NEXT: j

sw.epilog:                                        ; preds = %entry, %sw.bb1, %sw.bb
  ret void
}
