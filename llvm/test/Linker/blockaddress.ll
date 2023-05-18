; RUN: llvm-as %s -o %t.bc
; RUN: llvm-link %t.bc -S | FileCheck %s

declare void @f(ptr)

; Test that a blockaddress in @y referring to %label in @x can be moved when @y
; appears after @x.
define void @x() {
  br label %label

label:
  call void @y()
  ret void
}

define void @y() {
; CHECK: define void @y() {
; CHECK-NEXT: call void @f(ptr blockaddress(@x, %label))
  call void @f(ptr blockaddress(@x, %label))
  ret void
}

; Test that a blockaddress in @a referring to %label in @b can be moved when @a
; appears before @b.
define void @a() {
; CHECK: define void @a() {
; CHECK-NEXT: call void @f(ptr blockaddress(@b, %label))
  call void @f(ptr blockaddress(@b, %label))
  ret void
}

define void @b() {
  br label %label

label:
  call void @a()
  ret void
}

; Test that @c and @d can both have blockaddress Constants that refer to one
; another.

define void @c() {
; CHECK: define void @c() {
; CHECK-NEXT:  br label %label
; CHECK-EMPTY:
; CHECK-NEXT: label:
; CHECK-NEXT: call void @f(ptr blockaddress(@d, %label))
  br label %label

label:
  call void @f(ptr blockaddress(@d, %label))
  ret void
}

define void @d() {
; CHECK: define void @d() {
; CHECK-NEXT:  br label %label
; CHECK-EMPTY:
; CHECK-NEXT: label:
; CHECK-NEXT: call void @f(ptr blockaddress(@c, %label))
  br label %label

label:
  call void @f(ptr blockaddress(@c, %label))
  ret void
}

; Test that Functions added to IRLinker's Worklist member lazily (linkonce_odr)
; aren't susceptible to the the same issues as @x/@y above.
define void @parsed() {
  br label %label

label:
  ret void
}

define linkonce_odr void @lazy() {
; CHECK: define linkonce_odr void @lazy() {
; CHECK-NEXT: br label %label
; CHECK-EMPTY:
; CHECK-NEXT: label:
; CHECK-NEXT: call void @f(ptr blockaddress(@parsed, %label))
  br label %label

label:
  call void @f(ptr blockaddress(@parsed, %label))
  ret void
}

define void @parsed2() {
  call void @lazy()
  ret void
}

; Same test as @lazy, just with one more level of lazy parsed functions.
define void @parsed3() {
  br label %label

label:
  ret void
}

define linkonce_odr void @lazy1() {
; CHECK: define linkonce_odr void @lazy1() {
; CHECK-NEXT: br label %label
; CHECK-EMPTY:
; CHECK-NEXT: label:
; CHECK-NEXT: call void @f(ptr blockaddress(@parsed3, %label))
  br label %label

label:
  call void @f(ptr blockaddress(@parsed3, %label))
  ret void
}

define linkonce_odr void @lazy2() {
  call void @lazy1()
  ret void
}

define void @parsed4() {
  call void @lazy2()
  ret void
}
