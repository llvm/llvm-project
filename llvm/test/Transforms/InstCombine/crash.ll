; RUN: opt < %s -passes=instcombine -S
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin10.0"

define i32 @test0(i8 %tmp2) ssp {
entry:
  %tmp3 = zext i8 %tmp2 to i32
  %tmp8 = lshr i32 %tmp3, 6 
  %tmp9 = lshr i32 %tmp3, 7 
  %tmp10 = xor i32 %tmp9, 67108858
  %tmp11 = xor i32 %tmp10, %tmp8 
  %tmp12 = xor i32 %tmp11, 0     
  ret i32 %tmp12
}

; PR4905
define <2 x i64> @test1(<2 x i64> %x, <2 x i64> %y) nounwind {
entry:
  %conv.i94 = bitcast <2 x i64> %y to <4 x i32>   ; <<4 x i32>> [#uses=1]
  %sub.i97 = sub <4 x i32> %conv.i94, poison       ; <<4 x i32>> [#uses=1]
  %conv3.i98 = bitcast <4 x i32> %sub.i97 to <2 x i64> ; <<2 x i64>> [#uses=2]
  %conv2.i86 = bitcast <2 x i64> %conv3.i98 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %cmp.i87 = icmp sgt <4 x i32> poison, %conv2.i86 ; <<4 x i1>> [#uses=1]
  %sext.i88 = sext <4 x i1> %cmp.i87 to <4 x i32> ; <<4 x i32>> [#uses=1]
  %conv3.i89 = bitcast <4 x i32> %sext.i88 to <2 x i64> ; <<2 x i64>> [#uses=1]
  %and.i = and <2 x i64> %conv3.i89, %conv3.i98   ; <<2 x i64>> [#uses=1]
  %or.i = or <2 x i64> zeroinitializer, %and.i    ; <<2 x i64>> [#uses=1]
  %conv2.i43 = bitcast <2 x i64> %or.i to <4 x i32> ; <<4 x i32>> [#uses=1]
  %sub.i = sub <4 x i32> zeroinitializer, %conv2.i43 ; <<4 x i32>> [#uses=1]
  %conv3.i44 = bitcast <4 x i32> %sub.i to <2 x i64> ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %conv3.i44
}


; PR4908
define void @test2(ptr nocapture %b, ptr nocapture %c) nounwind ssp {
entry:
  %arrayidx = getelementptr inbounds <1 x i16>, ptr %b, i64 0 ; <ptr>
  %tmp2 = load <1 x i16>, ptr %arrayidx               ; <<1 x i16>> [#uses=1]
  %tmp6 = bitcast <1 x i16> %tmp2 to i16          ; <i16> [#uses=1]
  %tmp7 = zext i16 %tmp6 to i32                   ; <i32> [#uses=1]
  %ins = or i32 0, %tmp7                          ; <i32> [#uses=1]
  %arrayidx20 = getelementptr inbounds i32, ptr %c, i64 0 ; <ptr> [#uses=1]
  store i32 %ins, ptr %arrayidx20
  ret void
}

; PR5262
@tmp2 = global i64 0                              ; <ptr> [#uses=1]

declare void @use(i64) nounwind

define void @foo(i1) nounwind align 2 {
; <label>:1
  br i1 %0, label %2, label %3

; <label>:2                                       ; preds = %1
  br label %3

; <label>:3                                       ; preds = %2, %1
  %4 = phi i8 [ 1, %2 ], [ 0, %1 ]                ; <i8> [#uses=1]
  %5 = icmp eq i8 %4, 0                           ; <i1> [#uses=1]
  %6 = load i64, ptr @tmp2, align 8                   ; <i64> [#uses=1]
  %7 = select i1 %5, i64 0, i64 %6                ; <i64> [#uses=1]
  br label %8

; <label>:8                                       ; preds = %3
  call void @use(i64 %7)
  ret void
}

%t0 = type { i32, i32 }
%t1 = type { i32, i32, i32, i32, ptr }

declare ptr @bar2(i64)

define void @bar3(i1, i1) nounwind align 2 {
; <label>:2
  br i1 %1, label %10, label %3

; <label>:3                                       ; preds = %2
  %4 = getelementptr inbounds %t0, ptr null, i64 0, i32 1 ; <ptr> [#uses=0]
  %5 = getelementptr inbounds %t1, ptr null, i64 0, i32 4 ; <ptr> [#uses=1]
  %6 = load ptr, ptr %5, align 8                     ; <ptr> [#uses=1]
  %7 = icmp ne ptr %6, null                      ; <i1> [#uses=1]
  %8 = zext i1 %7 to i32                          ; <i32> [#uses=1]
  %9 = add i32 %8, 0                              ; <i32> [#uses=1]
  br label %10

; <label>:10                                      ; preds = %3, %2
  %11 = phi i32 [ %9, %3 ], [ 0, %2 ]             ; <i32> [#uses=1]
  br i1 %1, label %12, label %13

; <label>:12                                      ; preds = %10
  br label %13

; <label>:13                                      ; preds = %12, %10
  %14 = zext i32 %11 to i64                       ; <i64> [#uses=1]
  %15 = tail call ptr @bar2(i64 %14) nounwind      ; <ptr> [#uses=0]
  ret void
}




; PR5262
; Make sure the PHI node gets put in a place where all of its operands dominate
; it.
define i64 @test4(i1 %c, ptr %P) nounwind align 2 {
BB0:
  br i1 %c, label %BB1, label %BB2

BB1:
  br label %BB2

BB2:
  %v5_ = phi i1 [ true, %BB0], [false, %BB1]
  %v6 = load i64, ptr %P
  br label %l8

l8:
  br label %l10
  
l10:
  %v11 = select i1 %v5_, i64 0, i64 %v6
  ret i64 %v11
}

; PR5471
define i32 @test5a() {
       ret i32 0
}

define void @test5(ptr %ptr) personality ptr @__gxx_personality_v0 {
  store i1 true, ptr %ptr
  %r = invoke i32 @test5a() to label %exit unwind label %unwind
unwind:
  %exn = landingpad {ptr, i32}
          cleanup
  br label %exit
exit:
  ret void
}


; PR5673

@test6g = external global ptr  

define arm_aapcs_vfpcc i32 @test6(i32 %argc, ptr %argv) nounwind {
entry:
  store ptr getelementptr (i32, ptr @test6, i32 -2048), ptr @test6g, align 4
  unreachable
}


; PR5827

%class.RuleBasedBreakIterator = type { ptr }
%class.UStack = type { ptr }

define i32 @_ZN22RuleBasedBreakIterator15checkDictionaryEi(ptr %this, i32 %x) align 2 personality ptr @__gxx_personality_v0 {
entry:
  %breaks = alloca %class.UStack, align 4         ; <ptr> [#uses=3]
  call void @_ZN6UStackC1Ei(ptr %breaks, i32 0)
  %tobool = icmp ne i32 %x, 0                     ; <i1> [#uses=1]
  br i1 %tobool, label %cond.end, label %cond.false

terminate.handler:                                ; preds = %ehcleanup
  %exc = landingpad { ptr, i32 }
           cleanup
  call void @_ZSt9terminatev() noreturn nounwind
  unreachable

ehcleanup:                                        ; preds = %cond.false
  %exc1 = landingpad { ptr, i32 }
           catch ptr null
  invoke void @_ZN6UStackD1Ev(ptr %breaks)
          to label %cont unwind label %terminate.handler

cont:                                             ; preds = %ehcleanup
  resume { ptr, i32 } %exc1

cond.false:                                       ; preds = %entry
  %tmp4 = getelementptr inbounds %class.RuleBasedBreakIterator, ptr %this, i32 0, i32 0 ; <ptr> [#uses=1]
  %tmp5 = load ptr, ptr %tmp4                     ; <ptr> [#uses=1]
  %call = invoke i64 %tmp5()
          to label %cond.end unwind label %ehcleanup ; <i64> [#uses=1]

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i64 [ 0, %entry ], [ %call, %cond.false ] ; <i64> [#uses=1]
  %conv = trunc i64 %cond to i32                  ; <i32> [#uses=1]
  call void @_ZN6UStackD1Ev(ptr %breaks)
  ret i32 %conv
}

declare void @_ZN6UStackC1Ei(ptr, i32)

declare void @_ZN6UStackD1Ev(ptr)

declare i32 @__gxx_personality_v0(...)

declare void @_ZSt9terminatev()

declare void @_Unwind_Resume_or_Rethrow(ptr)



; rdar://7590304
define ptr @test10(ptr %self, ptr %tmp3, ptr %ptr1, ptr %ptr2) personality ptr @__gxx_personality_v0 {
entry:
  store i1 true, ptr %ptr1
  store i1 true, ptr %ptr2
  invoke void @test10a()
          to label %invoke.cont unwind label %try.handler ; <ptr> [#uses=0]

invoke.cont:                                      ; preds = %entry
  unreachable

try.handler:                                      ; preds = %entry
  %exn = landingpad {ptr, i32}
           catch ptr null
  ret ptr %self
}

define void @test10a() {
  ret void
}


; PR6193
define i32 @test11(i32 %aMaskWidth, i8 %aStride) nounwind {
entry:
  %conv41 = sext i8 %aStride to i32
  %neg = xor i32 %conv41, -1
  %and42 = and i32 %aMaskWidth, %neg
  %and47 = and i32 130, %conv41
  %or = or i32 %and42, %and47
  ret i32 %or
}

; PR6503
define void @test12(ptr %A) nounwind {
entry:
  %tmp1 = load i32, ptr %A
  %cmp = icmp ugt i32 1, %tmp1                    ; <i1> [#uses=1]
  %conv = zext i1 %cmp to i32                     ; <i32> [#uses=1]
  %tmp2 = load i32, ptr %A
  %cmp3 = icmp ne i32 %tmp2, 0                    ; <i1> [#uses=1]
  %conv4 = zext i1 %cmp3 to i32                   ; <i32> [#uses=1]
  %or = or i32 %conv, %conv4                      ; <i32> [#uses=1]
  %cmp5 = icmp ugt i32 0, %or                 ; <i1> [#uses=1]
  %conv6 = zext i1 %cmp5 to i32                   ; <i32> [#uses=0]
  ret void
}

%s1 = type { %s2, %s2, [6 x %s2], i32, i32, i32, [1 x i32], [0 x i8] }
%s2 = type { i64 }
define void @test13(ptr %ptr1, ptr %ptr2, ptr %ptr3) nounwind {
entry:
  %0 = getelementptr inbounds %s1, ptr null, i64 0, i32 2, i64 0, i32 0
  %1 = getelementptr inbounds %s1, ptr null, i64 0, i32 2, i64 1, i32 0
  %.pre = load i32, ptr %0, align 8
  %2 = lshr i32 %.pre, 19
  %brmerge = or i1 1, 0
  %3 = and i32 %2, 3
  %4 = add nsw i32 %3, 1
  %5 = shl i32 %4, 19
  %6 = add i32 %5, 1572864
  %7 = and i32 %6, 1572864
  %8 = load i64, ptr %1, align 8
  %trunc156 = trunc i64 %8 to i32
  %9 = and i32 %trunc156, -1537
  %10 = and i32 %9, -6145
  %11 = or i32 %10, 2048
  %12 = and i32 %11, -24577
  %13 = or i32 %12, 16384
  %14 = or i32 %13, 98304
  store i32 %14, ptr %ptr1, align 8
  %15 = and i32 %14, -1572865
  %16 = or i32 %15, %7
  store i32 %16, ptr %ptr2, align 8
  %17 = and i32 %16, -449
  %18 = or i32 %17, 64
  store i32 %18, ptr %ptr3, align 8
  unreachable
}


; PR8807
declare i32 @test14f(ptr) nounwind

define void @test14(ptr %ptr) nounwind readnone {
entry:
  %call10 = call i32 @test14f(ptr byval(i32) %ptr)
  ret void
}


; PR8896
@g_54 = external global [7 x i16]

define void @test15(ptr %p_92, i1 %c1) nounwind {
entry:
%0 = load i32, ptr %p_92, align 4
%1 = icmp ne i32 %0, 0
%2 = zext i1 %1 to i32
%3 = call i32 @func_14() nounwind
%4 = trunc i32 %3 to i16
%5 = sext i16 %4 to i32
%6 = trunc i32 %5 to i16
br i1 %c1, label %"3", label %"5"

"3":                                              ; preds = %entry
%7 = sext i16 %6 to i32
%8 = ashr i32 %7, -1649554541
%9 = trunc i32 %8 to i16
br label %"5"

"5":                                              ; preds = %"3", %entry
%10 = phi i16 [ %9, %"3" ], [ %6, %entry ]
%11 = sext i16 %10 to i32
%12 = xor i32 %2, %11
%13 = sext i32 %12 to i64
%14 = icmp ne i64 %13, 0
br i1 %14, label %return, label %"7"

"7":                                              ; preds = %"5"
ret void

return:                                           ; preds = %"5"
ret void
}

declare i32 @func_14()


define double @test16(i32 %a) nounwind {
  %cmp = icmp slt i32 %a, 2
  %select = select i1 %cmp, double 2.000000e+00, double 3.141592e+00
  ret double %select
}


; PR8983
%struct.basic_ios = type { i8 }

define ptr@test17() ssp {
entry:
  ret ptr null
}

; PR9013
define void @test18() nounwind ssp {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %l_197.0 = phi i32 [ 0, %entry ], [ %sub.i, %for.inc ]
  br label %for.inc

for.inc:                                          ; preds = %for.cond
  %conv = and i32 %l_197.0, 255
  %sub.i = add nsw i32 %conv, -1
  br label %for.cond

return:                                           ; No predecessors!
  ret void
}

; PR11275
declare void @test18b() noreturn
declare void @test18foo(ptr)
declare void @test18a() noreturn
define fastcc void @test18x(ptr %t0, i1 %b) uwtable align 2 personality ptr @__gxx_personality_v0 {
entry:
  br i1 %b, label %e1, label %e2
e1:
  invoke void @test18b() noreturn
          to label %u unwind label %lpad
e2:
  invoke void @test18a() noreturn
          to label %u unwind label %lpad
lpad:
  %t5 = phi ptr [ %t0, %e1 ], [ %t0, %e2 ]
  %lpad.nonloopexit262 = landingpad { ptr, i32 }
          cleanup
  call void @test18foo(ptr %t5)
  unreachable
u:
  unreachable
}
