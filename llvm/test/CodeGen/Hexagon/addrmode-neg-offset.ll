; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK: r0 = add(r0,#4)
; CHECK: memd(r0+#-16) =
define void @neg_gep_store(ptr %ptr) {
  %gep = getelementptr i8, ptr %ptr, i32 -12
  store i64 0, ptr %gep, align 8
  ret void
}
