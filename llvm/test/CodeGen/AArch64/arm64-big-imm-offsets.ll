; RUN: llc -mtriple=arm64-eabi < %s


; Make sure large offsets aren't mistaken for valid immediate offsets.
; <rdar://problem/13190511>
define void @f(ptr nocapture %p) {
entry:
  %a = ptrtoint ptr %p to i64
  %ao = add i64 %a, 25769803792
  %b = inttoptr i64 %ao to ptr
  store volatile i32 0, ptr %b, align 4
  store volatile i32 0, ptr %b, align 4
  ret void
}
