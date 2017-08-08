; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK: Function: t1
; CHECK: NoAlias: i32* %gep1, i32* %gep2
define void @t1([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %add
  ret void
}

; CHECK: Function: t2
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t2([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 1, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 0, i32 %add
  ret void
}

; CHECK: Function: t3
; CHECK: MustAlias: i32* %gep1, i32* %gep2
define void @t3([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 0, i32 %add
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 0, i32 %add
  ret void
}

; CHECK: Function: t4
; CHECK: MayAlias: i32* %gep1, i32* %gep2
define void @t4([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 1, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 %add, i32 %add
  ret void
}

; CHECK: Function: t5
; CHECK: MayAlias: i32* %gep2, i64* %bc
define void @t5([8 x i32]* %p, i32 %addend, i32* %q) {
  %knownnonzero = load i32, i32* %q, !range !0
  %add = add nsw nuw i32 %addend, %knownnonzero
  %gep1 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %addend
  %gep2 = getelementptr [8 x i32], [8 x i32]* %p, i32 2, i32 %add
  %bc = bitcast i32* %gep1 to i64*
  ret void
}

!0 = !{ i32 1, i32 5 }
