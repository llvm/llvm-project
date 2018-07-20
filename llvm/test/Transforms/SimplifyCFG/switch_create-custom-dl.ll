; RUN: opt -S -simplifycfg < %s | FileCheck %s
target datalayout="p:40:64:64:32"

declare void @foo1()

declare void @foo2()

define void @test1(i32 %V) {
        %C1 = icmp eq i32 %V, 4         ; <i1> [#uses=1]
        %C2 = icmp eq i32 %V, 17                ; <i1> [#uses=1]
        %CN = or i1 %C1, %C2            ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
; CHECK-LABEL: @test1(
; CHECK:  switch i32 %V, label %F [
; CHECK:    i32 17, label %T
; CHECK:    i32 4, label %T
; CHECK:  ]
}

define void @test1_ptr(i32* %V) {
        %C1 = icmp eq i32* %V, inttoptr (i32 4 to i32*)
        %C2 = icmp eq i32* %V, inttoptr (i32 17 to i32*)
        %CN = or i1 %C1, %C2            ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
; CHECK-LABEL: @test1_ptr(
; DL:  %magicptr = ptrtoint i32* %V to i32
; DL:  switch i32 %magicptr, label %F [
; DL:    i32 17, label %T
; DL:    i32 4, label %T
; DL:  ]
}

define void @test1_ptr_as1(i32 addrspace(1)* %V) {
        %C1 = icmp eq i32 addrspace(1)* %V, inttoptr (i32 4 to i32 addrspace(1)*)
        %C2 = icmp eq i32 addrspace(1)* %V, inttoptr (i32 17 to i32 addrspace(1)*)
        %CN = or i1 %C1, %C2            ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
; CHECK-LABEL: @test1_ptr_as1(
; DL:  %magicptr = ptrtoint i32 addrspace(1)* %V to i16
; DL:  switch i16 %magicptr, label %F [
; DL:    i16 17, label %T
; DL:    i16 4, label %T
; DL:  ]
}

define void @test2(i32 %V) {
        %C1 = icmp ne i32 %V, 4         ; <i1> [#uses=1]
        %C2 = icmp ne i32 %V, 17                ; <i1> [#uses=1]
        %CN = and i1 %C1, %C2           ; <i1> [#uses=1]
        br i1 %CN, label %T, label %F
T:              ; preds = %0
        call void @foo1( )
        ret void
F:              ; preds = %0
        call void @foo2( )
        ret void
; CHECK-LABEL: @test2(
; CHECK:  switch i32 %V, label %T [
; CHECK:    i32 17, label %F
; CHECK:    i32 4, label %F
; CHECK:  ]
}

define void @test3(i32 %V) {
        %C1 = icmp eq i32 %V, 4         ; <i1> [#uses=1]
        br i1 %C1, label %T, label %N
N:              ; preds = %0
        %C2 = icmp eq i32 %V, 17                ; <i1> [#uses=1]
        br i1 %C2, label %T, label %F
T:              ; preds = %N, %0
        call void @foo1( )
        ret void
F:              ; preds = %N
        call void @foo2( )
        ret void

; CHECK-LABEL: @test3(
; CHECK: switch i32 %V, label %F [
; CHECK:     i32 4, label %T
; CHECK:     i32 17, label %T
; CHECK:   ]
}



define i32 @test4(i8 zeroext %c) nounwind ssp noredzone {
entry:
  %cmp = icmp eq i8 %c, 62
  br i1 %cmp, label %lor.end, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp4 = icmp eq i8 %c, 34
  br i1 %cmp4, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %lor.lhs.false
  %cmp8 = icmp eq i8 %c, 92
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %lor.lhs.false, %entry
  %0 = phi i1 [ true, %lor.lhs.false ], [ true, %entry ], [ %cmp8, %lor.rhs ]
  %lor.ext = zext i1 %0 to i32
  ret i32 %lor.ext

; CHECK-LABEL: @test4(
; CHECK:  switch i8 %c, label %lor.rhs [
; CHECK:    i8 62, label %lor.end
; CHECK:    i8 34, label %lor.end
; CHECK:    i8 92, label %lor.end
; CHECK:  ]
}

define i32 @test5(i8 zeroext %c) nounwind ssp noredzone {
entry:
  switch i8 %c, label %lor.rhs [
    i8 62, label %lor.end
    i8 34, label %lor.end
    i8 92, label %lor.end
  ]

lor.rhs:                                          ; preds = %entry
  %V = icmp eq i8 %c, 92
  br label %lor.end

lor.end:                                          ; preds = %entry, %entry, %entry, %lor.rhs
  %0 = phi i1 [ true, %entry ], [ %V, %lor.rhs ], [ true, %entry ], [ true, %entry ]
  %lor.ext = zext i1 %0 to i32
  ret i32 %lor.ext
; CHECK-LABEL: @test5(
; CHECK:  switch i8 %c, label %lor.rhs [
; CHECK:    i8 62, label %lor.end
; CHECK:    i8 34, label %lor.end
; CHECK:    i8 92, label %lor.end
; CHECK:  ]
}


define i1 @test6({ i32, i32 }* %I) {
entry:
        %tmp.1.i = getelementptr { i32, i32 }, { i32, i32 }* %I, i64 0, i32 1         ; <i32*> [#uses=1]
        %tmp.2.i = load i32, i32* %tmp.1.i           ; <i32> [#uses=6]
        %tmp.2 = icmp eq i32 %tmp.2.i, 14               ; <i1> [#uses=1]
        br i1 %tmp.2, label %shortcirc_done.4, label %shortcirc_next.0
shortcirc_next.0:               ; preds = %entry
        %tmp.6 = icmp eq i32 %tmp.2.i, 15               ; <i1> [#uses=1]
        br i1 %tmp.6, label %shortcirc_done.4, label %shortcirc_next.1
shortcirc_next.1:               ; preds = %shortcirc_next.0
        %tmp.11 = icmp eq i32 %tmp.2.i, 16              ; <i1> [#uses=1]
        br i1 %tmp.11, label %shortcirc_done.4, label %shortcirc_next.2
shortcirc_next.2:               ; preds = %shortcirc_next.1
        %tmp.16 = icmp eq i32 %tmp.2.i, 17              ; <i1> [#uses=1]
        br i1 %tmp.16, label %shortcirc_done.4, label %shortcirc_next.3
shortcirc_next.3:               ; preds = %shortcirc_next.2
        %tmp.21 = icmp eq i32 %tmp.2.i, 18              ; <i1> [#uses=1]
        br i1 %tmp.21, label %shortcirc_done.4, label %shortcirc_next.4
shortcirc_next.4:               ; preds = %shortcirc_next.3
        %tmp.26 = icmp eq i32 %tmp.2.i, 19              ; <i1> [#uses=1]
        br label %UnifiedReturnBlock
shortcirc_done.4:               ; preds = %shortcirc_next.3, %shortcirc_next.2, %shortcirc_next.1, %shortcirc_next.0, %entry
        br label %UnifiedReturnBlock
UnifiedReturnBlock:             ; preds = %shortcirc_done.4, %shortcirc_next.4
        %UnifiedRetVal = phi i1 [ %tmp.26, %shortcirc_next.4 ], [ true, %shortcirc_done.4 ]             ; <i1> [#uses=1]
        ret i1 %UnifiedRetVal

; CHECK-LABEL: @test6(
; CHECK: %tmp.2.i.off = add i32 %tmp.2.i, -14
; CHECK: %switch = icmp ult i32 %tmp.2.i.off, 6
}

define void @test7(i8 zeroext %c, i32 %x) nounwind ssp noredzone {
entry:
  %cmp = icmp ult i32 %x, 32
  %cmp4 = icmp eq i8 %c, 97
  %or.cond = or i1 %cmp, %cmp4
  %cmp9 = icmp eq i8 %c, 99
  %or.cond11 = or i1 %or.cond, %cmp9
  br i1 %or.cond11, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @foo1() nounwind noredzone
  ret void

if.end:                                           ; preds = %entry
  ret void

; CHECK-LABEL: @test7(
; CHECK:   %cmp = icmp ult i32 %x, 32
; CHECK:   br i1 %cmp, label %if.then, label %switch.early.test
; CHECK: switch.early.test:
; CHECK:   switch i8 %c, label %if.end [
; CHECK:     i8 99, label %if.then
; CHECK:     i8 97, label %if.then
; CHECK:   ]
}

define i32 @test8(i8 zeroext %c, i32 %x, i1 %C) nounwind ssp noredzone {
entry:
  br i1 %C, label %N, label %if.then
N:
  %cmp = icmp ult i32 %x, 32
  %cmp4 = icmp eq i8 %c, 97
  %or.cond = or i1 %cmp, %cmp4
  %cmp9 = icmp eq i8 %c, 99
  %or.cond11 = or i1 %or.cond, %cmp9
  br i1 %or.cond11, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %A = phi i32 [0, %entry], [42, %N]
  tail call void @foo1() nounwind noredzone
  ret i32 %A

if.end:                                           ; preds = %entry
  ret i32 0

; CHECK-LABEL: @test8(
; CHECK: switch.early.test:
; CHECK:   switch i8 %c, label %if.end [
; CHECK:     i8 99, label %if.then
; CHECK:     i8 97, label %if.then
; CHECK:   ]
; CHECK:   %A = phi i32 [ 0, %entry ], [ 42, %switch.early.test ], [ 42, %N ], [ 42, %switch.early.test ]
}

;; This is "Example 7" from http://blog.regehr.org/archives/320
define i32 @test9(i8 zeroext %c) nounwind ssp noredzone {
entry:
  %cmp = icmp ult i8 %c, 33
  br i1 %cmp, label %lor.end, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %cmp4 = icmp eq i8 %c, 46
  br i1 %cmp4, label %lor.end, label %lor.lhs.false6

lor.lhs.false6:                                   ; preds = %lor.lhs.false
  %cmp9 = icmp eq i8 %c, 44
  br i1 %cmp9, label %lor.end, label %lor.lhs.false11

lor.lhs.false11:                                  ; preds = %lor.lhs.false6
  %cmp14 = icmp eq i8 %c, 58
  br i1 %cmp14, label %lor.end, label %lor.lhs.false16

lor.lhs.false16:                                  ; preds = %lor.lhs.false11
  %cmp19 = icmp eq i8 %c, 59
  br i1 %cmp19, label %lor.end, label %lor.lhs.false21

lor.lhs.false21:                                  ; preds = %lor.lhs.false16
  %cmp24 = icmp eq i8 %c, 60
  br i1 %cmp24, label %lor.end, label %lor.lhs.false26

lor.lhs.false26:                                  ; preds = %lor.lhs.false21
  %cmp29 = icmp eq i8 %c, 62
  br i1 %cmp29, label %lor.end, label %lor.lhs.false31

lor.lhs.false31:                                  ; preds = %lor.lhs.false26
  %cmp34 = icmp eq i8 %c, 34
  br i1 %cmp34, label %lor.end, label %lor.lhs.false36

lor.lhs.false36:                                  ; preds = %lor.lhs.false31
  %cmp39 = icmp eq i8 %c, 92
  br i1 %cmp39, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %lor.lhs.false36
  %cmp43 = icmp eq i8 %c, 39
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %lor.lhs.false36, %lor.lhs.false31, %lor.lhs.false26, %lor.lhs.false21, %lor.lhs.false16, %lor.lhs.false11, %lor.lhs.false6, %lor.lhs.false, %entry
  %0 = phi i1 [ true, %lor.lhs.false36 ], [ true, %lor.lhs.false31 ], [ true, %lor.lhs.false26 ], [ true, %lor.lhs.false21 ], [ true, %lor.lhs.false16 ], [ true, %lor.lhs.false11 ], [ true, %lor.lhs.false6 ], [ true, %lor.lhs.false ], [ true, %entry ], [ %cmp43, %lor.rhs ]
  %conv46 = zext i1 %0 to i32
  ret i32 %conv46

; CHECK-LABEL: @test9(
; CHECK:   %cmp = icmp ult i8 %c, 33
; CHECK:   br i1 %cmp, label %lor.end, label %switch.early.test

; CHECK: switch.early.test:
; CHECK:   switch i8 %c, label %lor.rhs [
; CHECK:     i8 92, label %lor.end
; CHECK:     i8 62, label %lor.end
; CHECK:     i8 60, label %lor.end
; CHECK:     i8 59, label %lor.end
; CHECK:     i8 58, label %lor.end
; CHECK:     i8 46, label %lor.end
; CHECK:     i8 44, label %lor.end
; CHECK:     i8 34, label %lor.end
; CHECK:     i8 39, label %lor.end
; CHECK:   ]
}

define i32 @test10(i32 %mode, i1 %Cond) {
  %A = icmp ne i32 %mode, 0
  %B = icmp ne i32 %mode, 51
  %C = and i1 %A, %B
  %D = and i1 %C, %Cond
  br i1 %D, label %T, label %F
T:
  ret i32 123
F:
  ret i32 324

; CHECK-LABEL: @test10(
; CHECK:  br i1 %Cond, label %switch.early.test, label %F
; CHECK:switch.early.test:
; CHECK:  switch i32 %mode, label %T [
; CHECK:    i32 51, label %F
; CHECK:    i32 0, label %F
; CHECK:  ]
}

; PR8780
define i32 @test11(i32 %bar) nounwind {
entry:
  %cmp = icmp eq i32 %bar, 4
  %cmp2 = icmp eq i32 %bar, 35
  %or.cond = or i1 %cmp, %cmp2
  %cmp5 = icmp eq i32 %bar, 53
  %or.cond1 = or i1 %or.cond, %cmp5
  %cmp8 = icmp eq i32 %bar, 24
  %or.cond2 = or i1 %or.cond1, %cmp8
  %cmp11 = icmp eq i32 %bar, 23
  %or.cond3 = or i1 %or.cond2, %cmp11
  %cmp14 = icmp eq i32 %bar, 55
  %or.cond4 = or i1 %or.cond3, %cmp14
  %cmp17 = icmp eq i32 %bar, 12
  %or.cond5 = or i1 %or.cond4, %cmp17
  %cmp20 = icmp eq i32 %bar, 35
  %or.cond6 = or i1 %or.cond5, %cmp20
  br i1 %or.cond6, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ 1, %if.then ], [ 0, %if.end ]
  ret i32 %retval.0

; CHECK-LABEL: @test11(
; CHECK: switch i32 %bar, label %if.end [
; CHECK:   i32 55, label %return
; CHECK:   i32 53, label %return
; CHECK:   i32 35, label %return
; CHECK:   i32 24, label %return
; CHECK:   i32 23, label %return
; CHECK:   i32 12, label %return
; CHECK:   i32 4, label %return
; CHECK: ]
}

define void @test12() nounwind {
entry:
  br label %bb49.us.us

bb49.us.us:
  %A = icmp eq i32 undef, undef
  br i1 %A, label %bb55.us.us, label %malformed

bb48.us.us:
  %B = icmp ugt i32 undef, undef
  br i1 %B, label %bb55.us.us, label %bb49.us.us

bb55.us.us:
  br label %bb48.us.us

malformed:
  ret void
; CHECK-LABEL: @test12(

}

; test13 - handle switch formation with ult.
define void @test13(i32 %x) nounwind ssp noredzone {
entry:
  %cmp = icmp ult i32 %x, 2
  br i1 %cmp, label %if.then, label %lor.lhs.false3

lor.lhs.false3:                                   ; preds = %lor.lhs.false
  %cmp5 = icmp eq i32 %x, 3
  br i1 %cmp5, label %if.then, label %lor.lhs.false6

lor.lhs.false6:                                   ; preds = %lor.lhs.false3
  %cmp8 = icmp eq i32 %x, 4
  br i1 %cmp8, label %if.then, label %lor.lhs.false9

lor.lhs.false9:                                   ; preds = %lor.lhs.false6
  %cmp11 = icmp eq i32 %x, 6
  br i1 %cmp11, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false9, %lor.lhs.false6, %lor.lhs.false3, %lor.lhs.false, %entry
  call void @foo1() noredzone
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false9
  ret void
; CHECK-LABEL: @test13(
; CHECK:  switch i32 %x, label %if.end [
; CHECK:     i32 6, label %if.then
; CHECK:     i32 4, label %if.then
; CHECK:     i32 3, label %if.then
; CHECK:     i32 1, label %if.then
; CHECK:     i32 0, label %if.then
; CHECK:   ]
}

; test14 - handle switch formation with ult.
define void @test14(i32 %x) nounwind ssp noredzone {
entry:
  %cmp = icmp ugt i32 %x, 2
  br i1 %cmp, label %lor.lhs.false3, label %if.then

lor.lhs.false3:                                   ; preds = %lor.lhs.false
  %cmp5 = icmp ne i32 %x, 3
  br i1 %cmp5, label %lor.lhs.false6, label %if.then

lor.lhs.false6:                                   ; preds = %lor.lhs.false3
  %cmp8 = icmp ne i32 %x, 4
  br i1 %cmp8, label %lor.lhs.false9, label %if.then

lor.lhs.false9:                                   ; preds = %lor.lhs.false6
  %cmp11 = icmp ne i32 %x, 6
  br i1 %cmp11, label %if.end, label %if.then

if.then:                                          ; preds = %lor.lhs.false9, %lor.lhs.false6, %lor.lhs.false3, %lor.lhs.false, %entry
  call void @foo1() noredzone
  br label %if.end

if.end:                                           ; preds = %if.then, %lor.lhs.false9
  ret void
; CHECK-LABEL: @test14(
; CHECK:  switch i32 %x, label %if.end [
; CHECK:     i32 6, label %if.then
; CHECK:     i32 4, label %if.then
; CHECK:     i32 3, label %if.then
; CHECK:     i32 1, label %if.then
; CHECK:     i32 0, label %if.then
; CHECK:   ]
}

; Don't crash on ginormous ranges.
define void @test15(i128 %x) nounwind {
  %cmp = icmp ugt i128 %x, 2
  br i1 %cmp, label %if.end, label %lor.false

lor.false:
  %cmp2 = icmp ne i128 %x, 100000000000000000000
  br i1 %cmp2, label %if.end, label %if.then

if.then:
  call void @foo1() noredzone
  br label %if.end

if.end:
  ret void

; CHECK-LABEL: @test15(
; CHECK-NOT: switch
; CHECK: ret void
}

; PR8675
; rdar://5134905
define zeroext i1 @test16(i32 %x) nounwind {
entry:
; CHECK-LABEL: @test16(
; CHECK: %x.off = add i32 %x, -1
; CHECK: %switch = icmp ult i32 %x.off, 3
  %cmp.i = icmp eq i32 %x, 1
  br i1 %cmp.i, label %lor.end, label %lor.lhs.false

lor.lhs.false:
  %cmp.i2 = icmp eq i32 %x, 2
  br i1 %cmp.i2, label %lor.end, label %lor.rhs

lor.rhs:
  %cmp.i1 = icmp eq i32 %x, 3
  br label %lor.end

lor.end:
  %0 = phi i1 [ true, %lor.lhs.false ], [ true, %entry ], [ %cmp.i1, %lor.rhs ]
  ret i1 %0
}

; Check that we don't turn an icmp into a switch where it's not useful.
define void @test17(i32 %x, i32 %y) {
  %cmp = icmp ult i32 %x, 3
  %switch = icmp ult i32 %y, 2
  %or.cond775 = or i1 %cmp, %switch
  br i1 %or.cond775, label %lor.lhs.false8, label %return

lor.lhs.false8:
  tail call void @foo1()
  ret void

return:
  ret void

; CHECK-LABEL: @test17(
; CHECK-NOT: switch.early.test
; CHECK-NOT: switch i32
; CHECK: ret void
}

define void @test18(i32 %arg) {
bb:
  %tmp = and i32 %arg, -2
  %tmp1 = icmp eq i32 %tmp, 8
  %tmp2 = icmp eq i32 %arg, 10
  %tmp3 = or i1 %tmp1, %tmp2
  %tmp4 = icmp eq i32 %arg, 11
  %tmp5 = or i1 %tmp3, %tmp4
  %tmp6 = icmp eq i32 %arg, 12
  %tmp7 = or i1 %tmp5, %tmp6
  br i1 %tmp7, label %bb19, label %bb8

bb8:                                              ; preds = %bb
  %tmp9 = add i32 %arg, -13
  %tmp10 = icmp ult i32 %tmp9, 2
  %tmp11 = icmp eq i32 %arg, 16
  %tmp12 = or i1 %tmp10, %tmp11
  %tmp13 = icmp eq i32 %arg, 17
  %tmp14 = or i1 %tmp12, %tmp13
  %tmp15 = icmp eq i32 %arg, 18
  %tmp16 = or i1 %tmp14, %tmp15
  %tmp17 = icmp eq i32 %arg, 15
  %tmp18 = or i1 %tmp16, %tmp17
  br i1 %tmp18, label %bb19, label %bb20

bb19:                                             ; preds = %bb8, %bb
  tail call void @foo1()
  br label %bb20

bb20:                                             ; preds = %bb19, %bb8
  ret void

; CHECK-LABEL: @test18(
; CHECK: %arg.off = add i32 %arg, -8
; CHECK: icmp ult i32 %arg.off, 11
}

define void @PR26323(i1 %tobool23, i32 %tmp3) {
entry:
  %tobool5 = icmp ne i32 %tmp3, 0
  %neg14 = and i32 %tmp3, -2
  %cmp17 = icmp ne i32 %neg14, -1
  %or.cond = and i1 %tobool5, %tobool23
  %or.cond1 = and i1 %cmp17, %or.cond
  br i1 %or.cond1, label %if.end29, label %if.then27

if.then27:                                        ; preds = %entry
  call void @foo1()
  unreachable

if.end29:                                         ; preds = %entry
  ret void
}

; CHECK-LABEL: define void @PR26323(
; CHECK:  %tobool5 = icmp ne i32 %tmp3, 0
; CHECK:  %neg14 = and i32 %tmp3, -2
; CHECK:  %cmp17 = icmp ne i32 %neg14, -1
; CHECK:  %or.cond = and i1 %tobool5, %tobool23
; CHECK:  %or.cond1 = and i1 %cmp17, %or.cond
; CHECK:  br i1 %or.cond1, label %if.end29, label %if.then27

; Form a switch when and'ing a negated power of two
; CHECK-LABEL: define void @test19
; CHECK: switch i32 %arg, label %else [
; CHECK: i32 32, label %if
; CHECK: i32 13, label %if
; CHECK: i32 12, label %if
define void @test19(i32 %arg) {
  %and = and i32 %arg, -2
  %cmp1 = icmp eq i32 %and, 12
  %cmp2 = icmp eq i32 %arg, 32
  %pred = or i1 %cmp1, %cmp2
  br i1 %pred, label %if, label %else

if:
  call void @foo1()
  ret void

else:
  ret void
}

; Since %cmp1 is always false, a switch is never formed
; CHECK-LABEL: define void @test20
; CHECK-NOT: switch
; CHECK: ret void
define void @test20(i32 %arg) {
  %and = and i32 %arg, -2
  %cmp1 = icmp eq i32 %and, 13
  %cmp2 = icmp eq i32 %arg, 32
  %pred = or i1 %cmp1, %cmp2
  br i1 %pred, label %if, label %else

if:
  call void @foo1()
  ret void

else:
  ret void
}

; Form a switch when or'ing a power of two
; CHECK-LABEL: define void @test21
; CHECK: i32 32, label %else
; CHECK: i32 13, label %else
; CHECK: i32 12, label %else
define void @test21(i32 %arg) {
  %and = or i32 %arg, 1
  %cmp1 = icmp ne i32 %and, 13
  %cmp2 = icmp ne i32 %arg, 32
  %pred = and i1 %cmp1, %cmp2
  br i1 %pred, label %if, label %else

if:
  call void @foo1()
  ret void

else:
  ret void
}

; Since %cmp1 is always false, a switch is never formed
; CHECK-LABEL: define void @test22
; CHECK-NOT: switch
; CHECK: ret void
define void @test22(i32 %arg) {
  %and = or i32 %arg, 1
  %cmp1 = icmp ne i32 %and, 12
  %cmp2 = icmp ne i32 %arg, 32
  %pred = and i1 %cmp1, %cmp2
  br i1 %pred, label %if, label %else

if:
  call void @foo1()
  ret void

else:
  ret void
}