; RUN: llc -mtriple=i686-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK32 --check-prefix=CHECK
; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK64 --check-prefix=CHECK

define void @zero_optsize(i32* %p) optsize {
entry:
  store i32 0, i32* %p
  ret void

; CHECK-LABEL: zero_optsize:
; CHECK: movl $0
; CHECK: ret
}

define void @minus_one_optsize(i32* %p) optsize {
entry:
  store i32 -1, i32* %p
  ret void

; CHECK-LABEL: minus_one_optsize:
; CHECK: movl $-1
; CHECK: ret
}


define void @zero_64(i64* %p) minsize {
entry:
  store i64 0, i64* %p
  ret void

; CHECK-LABEL: zero_64:
; CHECK32: andl $0
; CHECK32: andl $0
; CHECK64: andq $0
; CHECK: ret
}

define void @zero_32(i32* %p) minsize {
entry:
  store i32 0, i32* %p
  ret void

; CHECK-LABEL: zero_32:
; CHECK: andl $0
; CHECK: ret
}

define void @zero_16(i16* %p) minsize {
entry:
  store i16 0, i16* %p
  ret void

; CHECK-LABEL: zero_16:
; CHECK: andw $0
; CHECK: ret
}


define void @minus_one_64(i64* %p) minsize {
entry:
  store i64 -1, i64* %p
  ret void

; CHECK-LABEL: minus_one_64:
; CHECK32: orl $-1
; CHECK32: orl $-1
; CHECK64: orq $-1
; CHECK: ret
}

define void @minus_one_32(i32* %p) minsize {
entry:
  store i32 -1, i32* %p
  ret void

; CHECK-LABEL: minus_one_32:
; CHECK: orl $-1
; CHECK: ret
}

define void @minus_one_16(i16* %p) minsize {
entry:
  store i16 -1, i16* %p
  ret void

; CHECK-LABEL: minus_one_16:
; CHECK: orw $-1
; CHECK: ret
}
