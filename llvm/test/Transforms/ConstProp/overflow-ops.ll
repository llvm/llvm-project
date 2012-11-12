; RUN: opt < %s -constprop -S | FileCheck %s


declare {i8, i1} @llvm.uadd.with.overflow.i8(i8, i8)
declare {i8, i1} @llvm.usub.with.overflow.i8(i8, i8)
declare {i8, i1} @llvm.umul.with.overflow.i8(i8, i8)

declare {i8, i1} @llvm.sadd.with.overflow.i8(i8, i8)
declare {i8, i1} @llvm.ssub.with.overflow.i8(i8, i8)
declare {i8, i1} @llvm.smul.with.overflow.i8(i8, i8)

;;-----------------------------
;; uadd
;;-----------------------------

define {i8, i1} @uadd_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 42, i8 100)
  ret {i8, i1} %t

; CHECK: @uadd_1
; CHECK: ret { i8, i1 } { i8 -114, i1 false }
}

define {i8, i1} @uadd_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 142, i8 120)
  ret {i8, i1} %t

; CHECK: @uadd_2
; CHECK: ret { i8, i1 } { i8 6, i1 true }
}

;;-----------------------------
;; usub
;;-----------------------------

define {i8, i1} @usub_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.usub.with.overflow.i8(i8 4, i8 2)
  ret {i8, i1} %t

; CHECK: @usub_1
; CHECK: ret { i8, i1 } { i8 2, i1 false }
}

define {i8, i1} @usub_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.usub.with.overflow.i8(i8 4, i8 6)
  ret {i8, i1} %t

; CHECK: @usub_2
; CHECK: ret { i8, i1 } { i8 -2, i1 true }
}

;;-----------------------------
;; umul
;;-----------------------------

define {i8, i1} @umul_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.umul.with.overflow.i8(i8 100, i8 3)
  ret {i8, i1} %t

; CHECK: @umul_1
; CHECK: ret { i8, i1 } { i8 44, i1 true }
}

define {i8, i1} @umul_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.umul.with.overflow.i8(i8 100, i8 2)
  ret {i8, i1} %t

; CHECK: @umul_2
; CHECK: ret { i8, i1 } { i8 -56, i1 false }
}

;;-----------------------------
;; sadd
;;-----------------------------

define {i8, i1} @sadd_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 42, i8 2)
  ret {i8, i1} %t

; CHECK: @sadd_1
; CHECK: ret { i8, i1 } { i8 44, i1 false }
}

define {i8, i1} @sadd_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 120, i8 10)
  ret {i8, i1} %t

; CHECK: @sadd_2
; CHECK: ret { i8, i1 } { i8 -126, i1 true }
}

define {i8, i1} @sadd_3() nounwind {
entry:
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 -120, i8 10)
  ret {i8, i1} %t

; CHECK: @sadd_3
; CHECK: ret { i8, i1 } { i8 -110, i1 false }
}

define {i8, i1} @sadd_4() nounwind {
entry:
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 -120, i8 -10)
  ret {i8, i1} %t

; CHECK: @sadd_4
; CHECK: ret { i8, i1 } { i8 126, i1 true }
}

define {i8, i1} @sadd_5() nounwind {
entry:
  %t = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 2, i8 -10)
  ret {i8, i1} %t

; CHECK: @sadd_5
; CHECK: ret { i8, i1 } { i8 -8, i1 false }
}


;;-----------------------------
;; ssub
;;-----------------------------

define {i8, i1} @ssub_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 4, i8 2)
  ret {i8, i1} %t

; CHECK: @ssub_1
; CHECK: ret { i8, i1 } { i8 2, i1 false }
}

define {i8, i1} @ssub_2() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 4, i8 6)
  ret {i8, i1} %t

; CHECK: @ssub_2
; CHECK: ret { i8, i1 } { i8 -2, i1 false }
}

define {i8, i1} @ssub_3() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 -10, i8 120)
  ret {i8, i1} %t

; CHECK: @ssub_3
; CHECK: ret { i8, i1 } { i8 126, i1 true }
}

define {i8, i1} @ssub_3b() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 -10, i8 10)
  ret {i8, i1} %t

; CHECK: @ssub_3b
; CHECK: ret { i8, i1 } { i8 -20, i1 false }
}

define {i8, i1} @ssub_4() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 120, i8 -10)
  ret {i8, i1} %t

; CHECK: @ssub_4
; CHECK: ret { i8, i1 } { i8 -126, i1 true }
}

define {i8, i1} @ssub_4b() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 20, i8 -10)
  ret {i8, i1} %t

; CHECK: @ssub_4b
; CHECK: ret { i8, i1 } { i8 30, i1 false }
}

define {i8, i1} @ssub_5() nounwind {
entry:
  %t = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 -20, i8 -10)
  ret {i8, i1} %t

; CHECK: @ssub_5
; CHECK: ret { i8, i1 } { i8 -10, i1 false }
}

;;-----------------------------
;; smul
;;-----------------------------

; rdar://8501501
define {i8, i1} @smul_1() nounwind {
entry:
  %t = call {i8, i1} @llvm.smul.with.overflow.i8(i8 -20, i8 -10)
  ret {i8, i1} %t

; CHECK: @smul_1
; CHECK: ret { i8, i1 } { i8 -56, i1 true }
}
