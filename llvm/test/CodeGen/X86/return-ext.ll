; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -fixup-byte-word-insts=0 | \
; RUN:   FileCheck -check-prefix=CHECK -check-prefix=BWOFF %s
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -fixup-byte-word-insts=1 | \
; RUN:   FileCheck -check-prefix=CHECK -check-prefix=BWON %s
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -fixup-byte-word-insts=0 | \
; RUN:   FileCheck -check-prefix=CHECK -check-prefix=BWOFF %s
; RUN: llc < %s -mtriple=i686-unknown-linux-gnu -fixup-byte-word-insts=1 | \
; RUN:   FileCheck -check-prefix=CHECK -check-prefix=BWON %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -fixup-byte-word-insts=0 | \
; RUN:   FileCheck -check-prefix=DARWIN -check-prefix=DARWIN-BWOFF %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -fixup-byte-word-insts=1 | \
; RUN:   FileCheck -check-prefix=DARWIN -check-prefix=DARWIN-BWON %s


@x = common global i32 0, align 4

define zeroext i1 @unsigned_i1() {
entry:
  %0 = load i32, ptr @x
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
  %0 = load i32, ptr @x
  %cmp = icmp eq i32 %0, 42
  %retval = zext i1 %cmp to i8
  ret i8 %retval

; Unsigned i8 return values are not extended.
; CHECK-LABEL: unsigned_i8:
; CHECK:			 cmp
; CHECK-NEXT:  sete
; CHECK-NEXT:  ret

; Except on Darwin, for legacy reasons.
; DARWIN-LABEL: unsigned_i8:
; DARWIN:       xorl
; DARWIN-NEXT:  cmp
; DARWIN-NEXT:  sete
; DARWIN-NEXT:  ret
}

define signext i8 @signed_i8() {
entry:
  %0 = load i32, ptr @x
  %cmp = icmp eq i32 %0, 42
  %retval = zext i1 %cmp to i8
  ret i8 %retval

; Signed i8 return values are not extended.
; CHECK-LABEL: signed_i8:
; CHECK:			 cmp
; CHECK-NEXT:  sete
; CHECK-NEXT:  ret

; Except on Darwin, for legacy reasons.
; DARWIN-LABEL: signed_i8:
; DARWIN:       xorl
; DARWIN-NEXT:  cmp
; DARWIN-NEXT:  sete
; DARWIN-NEXT:  ret
}

@a = common global i16 0
@b = common global i16 0
define zeroext i16 @unsigned_i16() {
entry:
  %0 = load i16, ptr @a
  %1 = load i16, ptr @b
  %add = add i16 %1, %0
  ret i16 %add

; i16 return values are not extended.
; CHECK-LABEL: unsigned_i16:
; BWOFF:       movw
; BWON:        movzwl
; CHECK-NEXT:  addw
; CHECK-NEXT:  ret

; Except on Darwin, for legacy reasons.
; DARWIN-LABEL: unsigned_i16:
; DARWIN-BWOFF: movw
; DARWIN-BWON:  movzwl
; DARWIN-NEXT:  addw
; DARWIN-NEXT:  movzwl
; DARWIN-NEXT:  ret
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
; CHECK-NEXT: .cfi_def_cfa_offset {{4|8}}
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
; CHECK-NEXT: .cfi_def_cfa_offset {{4|8}}
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
; CHECK-NEXT: .cfi_def_cfa_offset {{4|8}}
; CHECK-NEXT: ret
}
