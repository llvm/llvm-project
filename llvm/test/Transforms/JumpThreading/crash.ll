; RUN: opt < %s -jump-threading -S | FileCheck %s
; PR2285
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
	%struct.system__secondary_stack__mark_id = type { i64, i64 }

define void @_ada_c35507b() {
entry:
	br label %bb

bb:		; preds = %bb13, %entry
	%ch.0 = phi i8 [ 0, %entry ], [ 0, %bb13 ]		; <i8> [#uses=1]
	%tmp11 = icmp ugt i8 %ch.0, 31		; <i1> [#uses=1]
	%tmp120 = call %struct.system__secondary_stack__mark_id @system__secondary_stack__ss_mark( )		; <%struct.system__secondary_stack__mark_id> [#uses=1]
	br i1 %tmp11, label %bb110, label %bb13

bb13:		; preds = %bb
	br label %bb

bb110:		; preds = %bb
	%mrv_gr124 = extractvalue %struct.system__secondary_stack__mark_id %tmp120, 1		; <i64> [#uses=0]
	unreachable
}

declare %struct.system__secondary_stack__mark_id @system__secondary_stack__ss_mark()



define fastcc void @findratio(double* nocapture %res1, double* nocapture %res2) nounwind ssp {
entry:
  br label %bb12

bb6.us:                                        
  %tmp = icmp eq i32 undef, undef              
  %tmp1 = fsub double undef, undef             
  %tmp2 = fcmp ult double %tmp1, 0.000000e+00  
  br i1 %tmp, label %bb6.us, label %bb13


bb12:                                            
  %tmp3 = fcmp ult double undef, 0.000000e+00  
  br label %bb13

bb13:                                            
  %.lcssa31 = phi double [ undef, %bb12 ], [ %tmp1, %bb6.us ]
  %.lcssa30 = phi i1 [ %tmp3, %bb12 ], [ %tmp2, %bb6.us ] 
  br i1 %.lcssa30, label %bb15, label %bb61

bb15:                                            
  %tmp4 = fsub double -0.000000e+00, %.lcssa31   
  ret void


bb61:                                            
  ret void
}


; PR5258
define i32 @test(i1 %cond, i1 %cond2, i32 %a) {
A:
  br i1 %cond, label %F, label %A1
F:
  br label %A1

A1:  
  %d = phi i1 [false, %A], [true, %F]
  %e = add i32 %a, %a
  br i1 %d, label %B, label %G
  
G:
  br i1 %cond2, label %B, label %D
  
B:
  %f = phi i32 [%e, %G], [%e, %A1]
  %b = add i32 0, 0
  switch i32 %a, label %C [
    i32 7, label %D
    i32 8, label %D
    i32 9, label %D
  ]

C:
  br label %D
  
D:
  %c = phi i32 [%e, %B], [%e, %B], [%e, %B], [%f, %C], [%e, %G]
  ret i32 %c
E:
  ret i32 412
}


define i32 @test2() nounwind {
entry:
        br i1 true, label %decDivideOp.exit, label %bb7.i

bb7.i:          ; preds = %bb7.i, %entry
        br label %bb7.i

decDivideOp.exit:               ; preds = %entry
        ret i32 undef
}


; PR3298

define i32 @test3(i32 %p_79, i32 %p_80) nounwind {
entry:
	br label %bb7

bb1:		; preds = %bb2
	br label %bb2

bb2:		; preds = %bb7, %bb1
	%l_82.0 = phi i8 [ 0, %bb1 ], [ %l_82.1, %bb7 ]		; <i8> [#uses=3]
	br i1 true, label %bb3, label %bb1

bb3:		; preds = %bb2
	%0 = icmp eq i32 %p_80_addr.1, 0		; <i1> [#uses=1]
	br i1 %0, label %bb7, label %bb6

bb5:		; preds = %bb6
	%1 = icmp eq i8 %l_82.0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1.i, label %bb.i

bb.i:		; preds = %bb5
	br label %safe_div_func_char_s_s.exit

bb1.i:		; preds = %bb5
	br label %safe_div_func_char_s_s.exit

safe_div_func_char_s_s.exit:		; preds = %bb1.i, %bb.i
	br label %bb6

bb6:		; preds = %safe_div_func_char_s_s.exit, %bb3
	%p_80_addr.0 = phi i32 [ %p_80_addr.1, %bb3 ], [ 1, %safe_div_func_char_s_s.exit ]		; <i32> [#uses=2]
	%2 = icmp eq i32 %p_80_addr.0, 0		; <i1> [#uses=1]
	br i1 %2, label %bb7, label %bb5

bb7:		; preds = %bb6, %bb3, %entry
	%l_82.1 = phi i8 [ 1, %entry ], [ %l_82.0, %bb3 ], [ %l_82.0, %bb6 ]		; <i8> [#uses=2]
	%p_80_addr.1 = phi i32 [ 0, %entry ], [ %p_80_addr.1, %bb3 ], [ %p_80_addr.0, %bb6 ]		; <i32> [#uses=4]
	%3 = icmp eq i32 %p_80_addr.1, 0		; <i1> [#uses=1]
	br i1 %3, label %bb8, label %bb2

bb8:		; preds = %bb7
	%4 = sext i8 %l_82.1 to i32		; <i32> [#uses=0]
	ret i32 0
}


; PR3353

define i32 @test4(i8 %X) {
entry:
        %Y = add i8 %X, 1
        %Z = add i8 %Y, 1
        br label %bb33.i

bb33.i:         ; preds = %bb33.i, %bb32.i
        switch i8 %Y, label %bb32.i [
                i8 39, label %bb35.split.i
                i8 13, label %bb33.i
        ]

bb35.split.i:
        ret i32 5
bb32.i:
        ret i32 1
}


define fastcc void @test5(i1 %tmp, i32 %tmp1) nounwind ssp {
entry:
  br i1 %tmp, label %bb12, label %bb13


bb12:                                            
  br label %bb13

bb13:                                            
  %.lcssa31 = phi i32 [ undef, %bb12 ], [ %tmp1, %entry ]
  %A = and i1 undef, undef
  br i1 %A, label %bb15, label %bb61

bb15:                                            
  ret void


bb61:                                            
  ret void
}


; PR5640
define fastcc void @test6(i1 %tmp, i1 %tmp1) nounwind ssp {
entry:
  br i1 %tmp, label %bb12, label %bb14

bb12:           
  br label %bb14

bb14:           
  %A = phi i1 [ %A, %bb13 ],  [ true, %bb12 ], [%tmp1, %entry]
  br label %bb13

bb13:                                            
  br i1 %A, label %bb14, label %bb61


bb61:                                            
  ret void
}


; PR5698
define void @test7(i32 %x) {
entry:
  br label %tailrecurse

tailrecurse:
  switch i32 %x, label %return [
    i32 2, label %bb2
    i32 3, label %bb
  ]

bb:         
  switch i32 %x, label %return [
    i32 2, label %bb2
    i32 3, label %tailrecurse
  ]

bb2:        
  ret void

return:     
  ret void
}

; PR6119
define i32 @test8(i32 %action) nounwind {
entry:
  switch i32 %action, label %lor.rhs [
    i32 1, label %if.then
    i32 0, label %lor.end
  ]

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef

lor.rhs:                                          ; preds = %entry
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %cmp103 = xor i1 undef, undef                   ; <i1> [#uses=1]
  br i1 %cmp103, label %for.cond, label %if.then

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond
}

; PR6119
define i32 @test9(i32 %action) nounwind {
entry:
  switch i32 %action, label %lor.rhs [
    i32 1, label %if.then
    i32 0, label %lor.end
  ]

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef

lor.rhs:                                          ; preds = %entry
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %0 = phi i1 [ undef, %lor.rhs ], [ true, %entry ] ; <i1> [#uses=1]
  %cmp103 = xor i1 undef, %0                      ; <i1> [#uses=1]
  br i1 %cmp103, label %for.cond, label %if.then

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond
}

; PR6119
define i32 @test10(i32 %action, i32 %type) nounwind {
entry:
  %cmp2 = icmp eq i32 %type, 0                    ; <i1> [#uses=1]
  switch i32 %action, label %lor.rhs [
    i32 1, label %if.then
    i32 0, label %lor.end
  ]

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef

lor.rhs:                                          ; preds = %entry
  %cmp101 = icmp eq i32 %action, 2                ; <i1> [#uses=1]
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %0 = phi i1 [ %cmp101, %lor.rhs ], [ true, %entry ] ; <i1> [#uses=1]
  %cmp103 = xor i1 %cmp2, %0                      ; <i1> [#uses=1]
  br i1 %cmp103, label %for.cond, label %if.then

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body

for.body:                                         ; preds = %for.cond
  br label %for.cond
}


; PR6305
define void @test11() nounwind {
entry:
  br label %A

A:                                             ; preds = %entry
  call void undef(i64 ptrtoint (i8* blockaddress(@test11, %A) to i64)) nounwind
  unreachable
}

; PR6743
define void @test12() nounwind ssp {
entry:
  br label %lbl_51

lbl_51:                                           ; preds = %if.then, %entry
  %tmp3 = phi i1 [ false, %if.then ], [ undef, %entry ] ; <i1> [#uses=2]
  br i1 %tmp3, label %if.end12, label %if.then

if.then:                                          ; preds = %lbl_51
  br i1 %tmp3, label %lbl_51, label %if.end12

if.end12:                                         ; preds = %if.then, %lbl_51
  ret void
}



; PR7356
define i32 @test13(i32* %P, i8* %Ptr) {
entry:
  indirectbr i8* %Ptr, [label %BrBlock, label %B2]
  
B2:
  store i32 4, i32 *%P
  br label %BrBlock

BrBlock:
  %L = load i32, i32* %P
  %C = icmp eq i32 %L, 42
  br i1 %C, label %T, label %F
  
T:
  ret i32 123
F:
  ret i32 1422
}


; PR7498
define void @test14() nounwind {
entry:
  %cmp33 = icmp slt i8 undef, 0                   ; <i1> [#uses=1]
  %tobool = icmp eq i8 undef, 0                   ; <i1> [#uses=1]
  br i1 %tobool, label %land.end69, label %land.rhs

land.rhs:                                         ; preds = %entry
  br label %land.end69

land.end69:                                       ; preds = %land.rhs, %entry
  %0 = phi i1 [ undef, %land.rhs ], [ true, %entry ] ; <i1> [#uses=1]
  %cmp71 = or i1 true, %0                         ; <i1> [#uses=1]
  %cmp73 = xor i1 %cmp33, %cmp71                  ; <i1> [#uses=1]
  br i1 %cmp73, label %if.then, label %if.end

if.then:                                          ; preds = %land.end69
  ret void

if.end:                                           ; preds = %land.end69
  ret void
}

; PR7647
define void @test15() nounwind {
entry:
  ret void
  
if.then237:
  br label %lbl_664

lbl_596:                                          ; preds = %lbl_664, %for.end37
  store volatile i64 undef, i64* undef, align 4
  br label %for.cond111

for.cond111:                                      ; preds = %safe_sub_func_int64_t_s_s.exit, %lbl_596
  %storemerge = phi i8 [ undef, %cond.true.i100 ], [ 22, %lbl_596 ] ; <i8> [#uses=1]
  %l_678.5 = phi i64 [ %l_678.3, %cond.true.i100 ], [ undef, %lbl_596 ] ; <i64> [#uses=2]
  %cmp114 = icmp slt i8 %storemerge, -2           ; <i1> [#uses=1]
  br i1 %cmp114, label %lbl_664, label %if.end949

lbl_664:                                          ; preds = %for.end1058, %if.then237, %for.cond111
  %l_678.3 = phi i64 [ %l_678.5, %for.cond111 ], [ %l_678.2, %for.cond1035 ], [ 5, %if.then237 ] ; <i64> [#uses=1]
  %tobool118 = icmp eq i32 undef, 0               ; <i1> [#uses=1]
  br i1 %tobool118, label %cond.true.i100, label %lbl_596

cond.true.i100:                                   ; preds = %for.inc120
  br label %for.cond111

lbl_709:
  br label %if.end949
  
for.cond603:                                      ; preds = %for.body607, %if.end336
  br i1 undef, label %for.cond603, label %if.end949

if.end949:                                        ; preds = %for.cond603, %lbl_709, %for.cond111
  %l_678.2 = phi i64 [ %l_678.5, %for.cond111 ], [ undef, %lbl_709 ], [ 5, %for.cond603 ] ; <i64> [#uses=1]
  br label %for.body1016

for.body1016:                                     ; preds = %for.cond1012
  br label %for.body1016

for.cond1035:                                     ; preds = %for.inc1055, %if.then1026
  br i1 undef, label %for.cond1040, label %lbl_664

for.cond1040:                                     ; preds = %for.body1044, %for.cond1035
  ret void
}

; PR7755
define void @test16(i1 %c, i1 %c2, i1 %c3, i1 %c4) nounwind ssp {
entry:
  %cmp = icmp sgt i32 undef, 1                    ; <i1> [#uses=1]
  br i1 %c, label %land.end, label %land.rhs

land.rhs:                                         ; preds = %entry
  br i1 %c2, label %lor.lhs.false.i, label %land.end

lor.lhs.false.i:                                  ; preds = %land.rhs
  br i1 %c3, label %land.end, label %land.end

land.end:                            
  %0 = phi i1 [ true, %entry ], [ false, %land.rhs ], [false, %lor.lhs.false.i], [false, %lor.lhs.false.i] ; <i1> [#uses=1]
  %cmp12 = and i1 %cmp, %0 
  %xor1 = xor i1 %cmp12, %c4
  br i1 %xor1, label %if.then, label %if.end

if.then:                      
  ret void

if.end:                       
  ret void
}

define void @test17() {
entry:
  br i1 undef, label %bb269.us.us, label %bb269.us.us.us

bb269.us.us.us:
  %indvar = phi i64 [ %indvar.next, %bb287.us.us.us ], [ 0, %entry ]
  %0 = icmp eq i16 undef, 0
  br i1 %0, label %bb287.us.us.us, label %bb286.us.us.us

bb287.us.us.us:
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 4
  br i1 %exitcond, label %bb288.bb289.loopexit_crit_edge, label %bb269.us.us.us

bb286.us.us.us:
  unreachable

bb269.us.us:
	unreachable

bb288.bb289.loopexit_crit_edge:
  unreachable
}

; PR 8247
%struct.S1 = type { i8, i8 }
@func_89.l_245 = internal constant %struct.S1 { i8 33, i8 6 }, align 1
define void @func_89(i16 zeroext %p_90, %struct.S1* nocapture %p_91, i32* nocapture %p_92) nounwind ssp {
entry:
  store i32 0, i32* %p_92, align 4
  br i1 false, label %lbl_260, label %if.else

if.else:                                          ; preds = %entry
  br label %for.cond

for.cond:                                         ; preds = %lbl_260, %if.else
  %l_245.0 = phi i16 [ %l_245.1, %lbl_260 ], [ 33, %if.else ]
  %l_261.0 = phi i32 [ %and, %lbl_260 ], [ 255, %if.else ]
  %tobool21 = icmp ult i16 %l_245.0, 256
  br i1 %tobool21, label %if.end, label %lbl_260

lbl_260:                                          ; preds = %for.cond, %entry
  %l_245.1 = phi i16 [ 1569, %entry ], [ %l_245.0, %for.cond ]
  %l_261.1 = phi i32 [ 255, %entry ], [ %l_261.0, %for.cond ]
  %and = and i32 %l_261.1, 1
  br label %for.cond

if.end:                                           ; preds = %for.cond
  ret void
}

define void @PR14233(i1 %cmp, i1 %cmp2, i1 %cmp3, i1 %cmp4) {
entry:
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:
  br label %if.end

cond.false:
  br label %if.end

if.end:
  %A = phi i64 [ 0, %cond.true ], [ 1, %cond.false ]
  br i1 %cmp2, label %bb, label %if.end2

bb:
  br label %if.end2

if.end2:
  %B = phi i64 [ ptrtoint (i8* ()* @PR14233.f1 to i64), %bb ], [ %A, %if.end ]
  %cmp.ptr = icmp eq i64 %B, ptrtoint (i8* ()* @PR14233.f2 to i64)
  br i1 %cmp.ptr, label %cond.true2, label %if.end3

cond.true2:
  br i1 %cmp3, label %bb2, label %ur

bb2:
  br i1 %cmp4, label %if.end4, label %if.end3

if.end4:
  unreachable

if.end3:
  %cmp.ptr2 = icmp eq i64 %B, ptrtoint (i8* ()* @PR14233.f2 to i64)
  br i1 %cmp.ptr2, label %ur, label %if.then601

if.then601:
  %C = icmp eq i64 %B, 0
  br i1 %C, label %bb3, label %bb4

bb3:
  unreachable

bb4:
  unreachable

ur:
  unreachable
}

declare i8* @PR14233.f1()

declare i8* @PR14233.f2()

; Make sure the following compiles in a sane amount of time, as opposed
; to taking exponential time.
; (CHECK to make sure the condition doesn't get simplified somehow;
; if it does, the testcase will need to be revised.)
; CHECK-LABEL: define void @almost_infinite_loop
; CHECK: %x39 = or i1 %x38, %x38
; CHECK: br i1 %x39, label %dest1, label %dest2
define void @almost_infinite_loop(i1 %x0) {
entry:
  br label %if.then57.i

if.then57.i:
  %x1 = or i1 %x0, %x0
  %x2 = or i1 %x1, %x1
  %x3 = or i1 %x2, %x2
  %x4 = or i1 %x3, %x3
  %x5 = or i1 %x4, %x4
  %x6 = or i1 %x5, %x5
  %x7 = or i1 %x6, %x6
  %x8 = or i1 %x7, %x7
  %x9 = or i1 %x8, %x8
  %x10 = or i1 %x9, %x9
  %x11 = or i1 %x10, %x10
  %x12 = or i1 %x11, %x11
  %x13 = or i1 %x12, %x12
  %x14 = or i1 %x13, %x13
  %x15 = or i1 %x14, %x14
  %x16 = or i1 %x15, %x15
  %x17 = or i1 %x16, %x16
  %x18 = or i1 %x17, %x17
  %x19 = or i1 %x18, %x18
  %x20 = or i1 %x19, %x19
  %x21 = or i1 %x20, %x20
  %x22 = or i1 %x21, %x21
  %x23 = or i1 %x22, %x22
  %x24 = or i1 %x23, %x23
  %x25 = or i1 %x24, %x24
  %x26 = or i1 %x25, %x25
  %x27 = or i1 %x26, %x26
  %x28 = or i1 %x27, %x27
  %x29 = or i1 %x28, %x28
  %x30 = or i1 %x29, %x29
  %x31 = or i1 %x30, %x30
  %x32 = or i1 %x31, %x31
  %x33 = or i1 %x32, %x32
  %x34 = or i1 %x33, %x33
  %x35 = or i1 %x34, %x34
  %x36 = or i1 %x35, %x35
  %x37 = or i1 %x36, %x36
  %x38 = or i1 %x37, %x37
  %x39 = or i1 %x38, %x38
  br i1 %x39, label %dest1, label %dest2

dest1:
  unreachable

dest2:
  unreachable
}
