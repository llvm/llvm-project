; RUN: opt < %s -simplifycfg -S | FileCheck %s

; In PR17073 ( http://llvm.org/pr17073 ), we illegally hoisted an operation that can trap.
; The first test confirms that we don't do that when the trapping op is reached by the current BB (block1).
; The second test confirms that we don't do that when the trapping op is reached by the previous BB (entry).
; The third test confirms that we can still do this optimization for an operation (add) that doesn't trap.
; The tests must be complicated enough to prevent previous SimplifyCFG actions from optimizing away
; the instructions that we're checking for.

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.9.0"

@a = common global i32 0, align 4
@b = common global i8 0, align 1

; CHECK-LABEL: can_trap1 
; CHECK-NOT: or i1 %tobool, icmp eq (i32* bitcast (i8* @b to i32*), i32* @a)
; CHECK-NOT: select i1 %tobool, i32* null, i32* select (i1 icmp eq (i64 urem (i64 2, i64 zext (i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a) to i64)), i64 0), i32* null, i32* @a) 
define i32* @can_trap1() {
entry:
  %0 = load i32, i32* @a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %exit, label %block1

block1:
  br i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a), label %exit, label %block2

block2:
  br label %exit

exit:
  %storemerge = phi i32* [ null, %entry ],[ null, %block2 ], [ select (i1 icmp eq (i64 urem (i64 2, i64 zext (i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a) to i64)), i64 0), i32* null, i32* @a), %block1 ]
  ret i32* %storemerge
}

; CHECK-LABEL: can_trap2 
; CHECK-NOT: or i1 %tobool, icmp eq (i32* bitcast (i8* @b to i32*), i32* @a)
; CHECK-NOT: select i1 %tobool, i32* select (i1 icmp eq (i64 urem (i64 2, i64 zext (i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a) to i64)), i64 0), i32* null, i32* @a), i32* null
define i32* @can_trap2() {
entry:
  %0 = load i32, i32* @a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %exit, label %block1

block1:
  br i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a), label %exit, label %block2

block2:
  br label %exit

exit:
  %storemerge = phi i32* [ select (i1 icmp eq (i64 urem (i64 2, i64 zext (i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a) to i64)), i64 0), i32* null, i32* @a), %entry ],[ null, %block2 ], [ null, %block1 ]
  ret i32* %storemerge
}

; CHECK-LABEL: cannot_trap 
; CHECK: select i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a), i32* select (i1 icmp eq (i64 add (i64 zext (i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a) to i64), i64 2), i64 0), i32* null, i32* @a), i32* null
define i32* @cannot_trap() {
entry:
  %0 = load i32, i32* @a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %exit, label %block1

block1:
  br i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a), label %exit, label %block2

block2:
  br label %exit

exit:
  %storemerge = phi i32* [ null, %entry ],[ null, %block2 ], [ select (i1 icmp eq (i64 add (i64 2, i64 zext (i1 icmp eq (i32* bitcast (i8* @b to i32*), i32* @a) to i64)), i64 0), i32* null, i32* @a), %block1 ]
  ret i32* %storemerge
}
