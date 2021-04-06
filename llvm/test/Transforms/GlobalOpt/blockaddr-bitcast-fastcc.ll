; RUN: opt < %s -globalopt -S | FileCheck %s

; Check if fastcc is set on function "g"
; as its address is not taken and other
; conditions are met to tag "g" as fastcc.

@x = internal global i8* zeroinitializer

; CHECK: define internal fastcc void @g() unnamed_addr {
define internal void @g() {
entry:
  br label %here

here:
  ret void
}

define void @f() {
  store i8* blockaddress(@g, %here), i8** @x, align 8
; CHECK: call fastcc i32 bitcast (void ()* @g to i32 ()*)()
  call i32 bitcast (void ()* @g to i32 ()*)()
  ret void
}

