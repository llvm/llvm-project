; RUN: llc < %s -march=x86-64 -mtriple=x86_64-pc-linux-gnu -show-mc-encoding | FileCheck %s

; PR8365
; CHECK: andl	$-64, %edi              # encoding: [0x83,0xe7,0xc0]

define i64 @bra(i32 %zed) nounwind {
 %t1 = zext i32 %zed to i64
 %t2 = and i64  %t1, 4294967232
 ret i64 %t2
}

; CHECK:  orq     $2, %rdi                # encoding: [0x48,0x83,0xcf,0x02]

define void @foo(i64 %zed, i64* %x) nounwind {
  %t1 = and i64 %zed, -4
  %t2 = or i64 %t1, 2
  store i64 %t2, i64* %x, align 8
  ret void
}

define i64 @bar(i64 %zed) nounwind {
; CHECK:  andl     $42, %edi               # encoding: [0x83,0xe7,0x2a]
  %t1 = and i64 %zed, 42
  ret i64 %t1
}

define i64 @baz(i64 %zed) nounwind {
; CHECK:  andl $2147483647, %edi      # encoding: [0x81,0xe7,0xff,0xff,0xff,0x7f]
  %t1 = and i64 %zed, 2147483647
  ret i64 %t1
}
