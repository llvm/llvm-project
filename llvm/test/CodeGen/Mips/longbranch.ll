; RUN: llc -march=mipsel -force-mips-long-branch < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=n64  -force-mips-long-branch < %s | FileCheck %s -check-prefix=N64

@g0 = external global i32

define void @foo1(i32 %s) nounwind {
entry:
; O32: bal
; O32: lui $1, 0
; O32: addiu $1, $1, {{[0-9]+}} 
; N64: lui $1, 0
; N64: daddiu $1, $1, 0
; N64: dsll $1, $1, 16
; N64: daddiu $1, $1, 0
; N64: bal
; N64: dsll $1, $1, 16
; N64: daddiu $1, $1, {{[0-9]+}}  

  %tobool = icmp eq i32 %s, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32* @g0, align 4
  %add = add nsw i32 %0, 12
  store i32 %add, i32* @g0, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

