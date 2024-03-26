; RUN: llc < %s -mtriple=lanai-unknown-unknown | FileCheck %s

; Test scheduling of subwords.

%struct.X = type { i16, i16 }

define void @f(ptr inreg nocapture %c) #0 {
entry:
  %a = getelementptr inbounds %struct.X, ptr %c, i32 0, i32 0
  %0 = load i16, ptr %a, align 2
  %inc = add i16 %0, 1
  store i16 %inc, ptr %a, align 2
  %b = getelementptr inbounds %struct.X, ptr %c, i32 0, i32 1
  %1 = load i16, ptr %b, align 2
  %dec = add i16 %1, -1
  store i16 %dec, ptr %b, align 2
  ret void
}

; Verify that the two loads occur before the stores. Without memory
; disambiguation and subword schedule, the resultant code was a per subword
; load-modify-store sequence instead of the more optimal schedule where all
; loads occurred before modification and storage.
; CHECK:      uld.h
; CHECK-NEXT: uld.h
; CHECK-NEXT: add
; CHECK-NEXT: st.h
; CHECK-NEXT: sub
; CHECK-NEXT: st.h
