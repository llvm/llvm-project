; RUN: llc < %s -mtriple=i686-unknown-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s


@x = common global i32 0, align 4

define zeroext i1 @unsigned_i1() {
entry:
  %0 = load i32, i32* @x
  %cmp = icmp eq i32 %0, 42
  ret i1 %cmp

; Unsigned i1 return values are not extended.
; CHECK-LABEL: unsigned_i1:
; CHECK:			 cmp
; CHECK-NEXT:  sete
; CHECK-NEXT:  ret
}

define zeroext i8 @unsigned_i8() {
entry:
  %0 = load i32, i32* @x
  %cmp = icmp eq i32 %0, 42
  %retval = zext i1 %cmp to i8
  ret i8 %retval

; Unsigned i8 return values are not extended.
; CHECK-LABEL: unsigned_i8:
; CHECK:			 cmp
; CHECK-NEXT:  sete
; CHECK-NEXT:  ret
}

define signext i8 @signed_i8() {
entry:
  %0 = load i32, i32* @x
  %cmp = icmp eq i32 %0, 42
  %retval = zext i1 %cmp to i8
  ret i8 %retval

; Signed i8 return values are not extended.
; CHECK-LABEL: signed_i8:
; CHECK:			 cmp
; CHECK-NEXT:  sete
; CHECK-NEXT:  ret
}

@a = common global i16 0
@b = common global i16 0
define zeroext i16 @unsigned_i16() {
entry:
  %0 = load i16, i16* @a
  %1 = load i16, i16* @b
  %add = add i16 %1, %0
  ret i16 %add

; i16 return values are not extended.
; CHECK-LABEL: unsigned_i16:
; CHECK:			 movw
; CHECK-NEXT:  addw
; CHECK-NEXT:  ret
}


define i32 @use_i1() {
entry:
  %0 = call i1 @unsigned_i1();
  %1 = zext i1 %0 to i32
  ret i32 %1

; The high 24 bits of %eax from a function returning i1 are undefined.
; CHECK-LABEL: use_i1:
; CHECK: call
; CHECK-NEXT: movzbl
; CHECK-NEXT: {{pop|add}}
; CHECK-NEXT: ret
}

define i32 @use_i8() {
entry:
  %0 = call i8 @unsigned_i8();
  %1 = zext i8 %0 to i32
  ret i32 %1

; The high 24 bits of %eax from a function returning i8 are undefined.
; CHECK-LABEL: use_i8:
; CHECK: call
; CHECK-NEXT: movzbl
; CHECK-NEXT: {{pop|add}}
; CHECK-NEXT: ret
}

define i32 @use_i16() {
entry:
  %0 = call i16 @unsigned_i16();
  %1 = zext i16 %0 to i32
  ret i32 %1

; The high 16 bits of %eax from a function returning i16 are undefined.
; CHECK-LABEL: use_i16:
; CHECK: call
; CHECK-NEXT: movzwl
; CHECK-NEXT: {{pop|add}}
; CHECK-NEXT: ret
}
