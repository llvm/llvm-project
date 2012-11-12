; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s --check-prefix=CHECK-64
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s --check-prefix=CHECK-64
; RUN: llc < %s -march=x86 | FileCheck %s --check-prefix=CHECK-32

; CHECK-64: g64xh:
; CHECK-64:   testb $8, {{%ah|%ch}}
; CHECK-64:   ret
; CHECK-32: g64xh:
; CHECK-32:   testb $8, %ah
; CHECK-32:   ret
define void @g64xh(i64 inreg %x) nounwind {
  %t = and i64 %x, 2048
  %s = icmp eq i64 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g64xl:
; CHECK-64:   testb $8, [[A0L:%dil|%cl]]
; CHECK-64:   ret
; CHECK-32: g64xl:
; CHECK-32:   testb $8, %al
; CHECK-32:   ret
define void @g64xl(i64 inreg %x) nounwind {
  %t = and i64 %x, 8
  %s = icmp eq i64 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g32xh:
; CHECK-64:   testb $8, {{%ah|%ch}}
; CHECK-64:   ret
; CHECK-32: g32xh:
; CHECK-32:   testb $8, %ah
; CHECK-32:   ret
define void @g32xh(i32 inreg %x) nounwind {
  %t = and i32 %x, 2048
  %s = icmp eq i32 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g32xl:
; CHECK-64:   testb $8, [[A0L]]
; CHECK-64:   ret
; CHECK-32: g32xl:
; CHECK-32:   testb $8, %al
; CHECK-32:   ret
define void @g32xl(i32 inreg %x) nounwind {
  %t = and i32 %x, 8
  %s = icmp eq i32 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g16xh:
; CHECK-64:   testb $8, {{%ah|%ch}}
; CHECK-64:   ret
; CHECK-32: g16xh:
; CHECK-32:   testb $8, %ah
; CHECK-32:   ret
define void @g16xh(i16 inreg %x) nounwind {
  %t = and i16 %x, 2048
  %s = icmp eq i16 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g16xl:
; CHECK-64:   testb $8, [[A0L]]
; CHECK-64:   ret
; CHECK-32: g16xl:
; CHECK-32:   testb $8, %al
; CHECK-32:   ret
define void @g16xl(i16 inreg %x) nounwind {
  %t = and i16 %x, 8
  %s = icmp eq i16 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g64x16:
; CHECK-64:   testw $-32640, %[[A0W:di|cx]]
; CHECK-64:   ret
; CHECK-32: g64x16:
; CHECK-32:   testw $-32640, %ax
; CHECK-32:   ret
define void @g64x16(i64 inreg %x) nounwind {
  %t = and i64 %x, 32896
  %s = icmp eq i64 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g32x16:
; CHECK-64:   testw $-32640, %[[A0W]]
; CHECK-64:   ret
; CHECK-32: g32x16:
; CHECK-32:   testw $-32640, %ax
; CHECK-32:   ret
define void @g32x16(i32 inreg %x) nounwind {
  %t = and i32 %x, 32896
  %s = icmp eq i32 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}
; CHECK-64: g64x32:
; CHECK-64:   testl $268468352, %e[[A0W]]
; CHECK-64:   ret
; CHECK-32: g64x32:
; CHECK-32:   testl $268468352, %eax
; CHECK-32:   ret
define void @g64x32(i64 inreg %x) nounwind {
  %t = and i64 %x, 268468352
  %s = icmp eq i64 %t, 0
  br i1 %s, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}

declare void @bar()
