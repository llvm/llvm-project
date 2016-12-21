; RUN: llc -relocation-model=static -verify-machineinstrs -mtriple powerpc64-linux < %s | FileCheck %s
; RUN: llc -relocation-model=static -verify-machineinstrs -O0 -mtriple powerpc64-linux < %s | FileCheck %s

define void @foo()  {
  ret void
}
declare i32 @bar(i8*)

; CHECK-LABEL: {{^}}zed:
; CHECK:        addis 3, 2, foo@toc@ha
; CHECK-NEXT:   addi 3, 3, foo@toc@l
; CHECK-NEXT:   bl bar

define  void @zed() {
  call i32 @bar(i8* bitcast (void ()* @foo to i8*))
  ret void
}
