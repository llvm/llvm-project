; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-LABEL: foo:
; CHECK: memh(r0+#0){{.*}}={{.*}}#-2
; Don't use memh(r0+#0)=##65534.

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; Function Attrs: nounwind
define void @foo(ptr nocapture %s) #0 {
entry:
  store i8 -2, ptr %s, align 2
  %add.ptr = getelementptr inbounds i8, ptr %s, i32 1
  store i8 -1, ptr %add.ptr, align 1
  ret void
}

attributes #0 = { nounwind }
