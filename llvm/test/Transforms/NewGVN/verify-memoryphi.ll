; Skip dead MemoryPhis when performing memory congruency verification
; in NewGVN.
; RUN: opt -S -passes=newgvn %s | FileCheck %s
; REQUIRES: asserts

; CHECK: define void @tinkywinky() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br i1 false, label %body, label %end
; CHECK:      body:
; CHECK-NEXT:   store i8 poison, ptr null
; CHECK-NEXT:   br label %end
; CHECK:      end:
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)

define void @tinkywinky() {
entry:
  call void @llvm.lifetime.start.p0(i64 4, ptr undef)
  br i1 false, label %body, label %end

body:
  call void @llvm.lifetime.start.p0(i64 4, ptr undef)
  br label %end

end:
  ret void
}
