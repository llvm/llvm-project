; Test that simplifycfg can create switch instructions from constant pointers.
;
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

@.str = private constant [5 x i8] c"null\00"      ; <[5 x i8]*> [#uses=2]
@.str1 = private constant [4 x i8] c"one\00"      ; <[4 x i8]*> [#uses=2]
@.str2 = private constant [4 x i8] c"two\00"      ; <[4 x i8]*> [#uses=2]
@.str3 = private constant [5 x i8] c"four\00"     ; <[5 x i8]*> [#uses=2]

@.str_as1 = private addrspace(1) constant [5 x i8] c"null\00"      ; <[5 x i8]*> [#uses=2]
@.str1_as1 = private addrspace(1) constant [4 x i8] c"one\00"      ; <[4 x i8]*> [#uses=2]
@.str2_as1 = private addrspace(1) constant [4 x i8] c"two\00"      ; <[4 x i8]*> [#uses=2]
@.str3_as1 = private addrspace(1) constant [5 x i8] c"four\00"     ; <[5 x i8]*> [#uses=2]

declare i32 @puts(i8*)
declare i32 @puts_as1(i8 addrspace(1)*)

define void @f(i8* %x) nounwind ssp {
; CHECK-LABEL: @f(
; CHECK: switch i64 %magicptr
; CHECK: i64 0, label
; CHECK: i64 1, label
; CHECK: i64 2, label
; CHECK: i64 3, label
; CHECK: i64 4, label
; CHECK: }

entry:
  %tobool = icmp eq i8* %x, null                  ; <i1> [#uses=1]
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = call i32 @puts(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end21

if.else:                                          ; preds = %entry
  %cmp = icmp eq i8* %x, inttoptr (i64 1 to i8*)  ; <i1> [#uses=1]
  br i1 %cmp, label %if.then2, label %if.else4

if.then2:                                         ; preds = %if.else
  %call3 = call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end20

if.else4:                                         ; preds = %if.else
  %cmp6 = icmp eq i8* %x, inttoptr (i64 2 to i8*) ; <i1> [#uses=1]
  br i1 %cmp6, label %if.then9, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.else4
  %cmp8 = icmp eq i8* %x, inttoptr (i64 3 to i8*) ; <i1> [#uses=1]
  br i1 %cmp8, label %if.then9, label %if.else11

if.then9:                                         ; preds = %lor.lhs.false, %if.else4
  %call10 = call i32 @puts(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str2, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end19

if.else11:                                        ; preds = %lor.lhs.false
  %cmp13 = icmp eq i8* %x, inttoptr (i64 4 to i8*) ; <i1> [#uses=1]
  br i1 %cmp13, label %if.then14, label %if.else16

if.then14:                                        ; preds = %if.else11
  %call15 = call i32 @puts(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str3, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end

if.else16:                                        ; preds = %if.else11
  %call18 = call i32 @puts(i8* %x) nounwind       ; <i32> [#uses=0]
  br label %if.end

if.end:                                           ; preds = %if.else16, %if.then14
  br label %if.end19

if.end19:                                         ; preds = %if.end, %if.then9
  br label %if.end20

if.end20:                                         ; preds = %if.end19, %if.then2
  br label %if.end21

if.end21:                                         ; preds = %if.end20, %if.then
  ret void
}

; Is it useful to test a version where the ptrtoints are to the same
; size?
define void @f_as1(i8 addrspace(1)* %x) nounwind ssp {
; CHECK-LABEL: @f_as1(
; CHECK: ptrtoint i8 addrspace(1)* %x to i16
; CHECK: switch i16 %magicptr
; CHECK: i16 0, label
; CHECK: i16 1, label
; CHECK: i16 2, label
; CHECK: i16 3, label
; CHECK: i16 4, label
; CHECK: }

entry:
  %tobool = icmp eq i8 addrspace(1)* %x, null                  ; <i1> [#uses=1]
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = call i32 @puts_as1(i8 addrspace(1)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(1)* @.str_as1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end21

if.else:                                          ; preds = %entry
  %cmp = icmp eq i8 addrspace(1)* %x, inttoptr (i64 1 to i8 addrspace(1)*)  ; <i1> [#uses=1]
  br i1 %cmp, label %if.then2, label %if.else4

if.then2:                                         ; preds = %if.else
  %call3 = call i32 @puts_as1(i8 addrspace(1)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(1)* @.str1_as1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end20

if.else4:                                         ; preds = %if.else
  %cmp6 = icmp eq i8 addrspace(1)* %x, inttoptr (i64 2 to i8 addrspace(1)*) ; <i1> [#uses=1]
  br i1 %cmp6, label %if.then9, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.else4
  %cmp8 = icmp eq i8 addrspace(1)* %x, inttoptr (i64 3 to i8 addrspace(1)*) ; <i1> [#uses=1]
  br i1 %cmp8, label %if.then9, label %if.else11

if.then9:                                         ; preds = %lor.lhs.false, %if.else4
  %call10 = call i32 @puts_as1(i8 addrspace(1)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(1)* @.str2_as1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end19

if.else11:                                        ; preds = %lor.lhs.false
  %cmp13 = icmp eq i8 addrspace(1)* %x, inttoptr (i64 4 to i8 addrspace(1)*) ; <i1> [#uses=1]
  br i1 %cmp13, label %if.then14, label %if.else16

if.then14:                                        ; preds = %if.else11
  %call15 = call i32 @puts_as1(i8 addrspace(1)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(1)* @.str3_as1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  br label %if.end

if.else16:                                        ; preds = %if.else11
  %call18 = call i32 @puts_as1(i8 addrspace(1)* %x) nounwind       ; <i32> [#uses=0]
  br label %if.end

if.end:                                           ; preds = %if.else16, %if.then14
  br label %if.end19

if.end19:                                         ; preds = %if.end, %if.then9
  br label %if.end20

if.end20:                                         ; preds = %if.end19, %if.then2
  br label %if.end21

if.end21:                                         ; preds = %if.end20, %if.then
  ret void
}

