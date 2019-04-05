; RUN: llvm-extract -func=a --recursive %s -S | FileCheck --check-prefix=CHECK-AB %s
; RUN: llvm-extract -func=a --recursive --delete %s -S | FileCheck --check-prefix=CHECK-CD %s
; RUN: llvm-extract -func=d --recursive %s -S | FileCheck --check-prefix=CHECK-CD %s
; RUN: llvm-extract -func=e --recursive %s -S | FileCheck --check-prefix=CHECK-CD %s

; CHECK-AB: define void @a
; CHECK-AB: define void @b
; CHECK-AB-NOT: define void @c
; CHECK-AB-NOT: define void @d

; CHECK-CD-NOT: define void @a
; CHECK-CD-NOT: define void @b
; CHECK-CD: define void @c
; CHECK-CD: define void @d

define void @a() {
  call void @b()
  ret void
}

define void @b() {
  ret void
}

define void @c() {
  call void @d()
  ret void
}

define void @d() {
  call void @c()
  ret void
}

define void @e() {
  invoke void @c()
  to label %L unwind label %L
L:
  ret void
}