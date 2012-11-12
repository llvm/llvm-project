; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64"
%intstruct = type { i32 }
%pair = type { i32, i32 }
%struct.B = type { double }
%struct.A = type { %struct.B, i32, i32 }


@Global = constant [10 x i8] c"helloworld"

; Test noop elimination
define i32* @test1(i32* %I) {
        %A = getelementptr i32* %I, i64 0 
        ret i32* %A
; CHECK: @test1
; CHECK: ret i32* %I
}

; Test noop elimination
define i32* @test2(i32* %I) {
        %A = getelementptr i32* %I
        ret i32* %A
; CHECK: @test2
; CHECK: ret i32* %I
}

; Test that two array indexing geps fold
define i32* @test3(i32* %I) {
        %A = getelementptr i32* %I, i64 17
        %B = getelementptr i32* %A, i64 4
        ret i32* %B
; CHECK: @test3
; CHECK: getelementptr i32* %I, i64 21
}

; Test that two getelementptr insts fold
define i32* @test4({ i32 }* %I) {
        %A = getelementptr { i32 }* %I, i64 1 
        %B = getelementptr { i32 }* %A, i64 0, i32 0
        ret i32* %B
; CHECK: @test4
; CHECK: getelementptr { i32 }* %I, i64 1, i32 0
}

define void @test5(i8 %B) {
        ; This should be turned into a constexpr instead of being an instruction
        %A = getelementptr [10 x i8]* @Global, i64 0, i64 4 
        store i8 %B, i8* %A
        ret void
; CHECK: @test5
; CHECK: store i8 %B, i8* getelementptr inbounds ([10 x i8]* @Global, i64 0, i64 4)
}


define i32* @test7(i32* %I, i64 %C, i64 %D) {
        %A = getelementptr i32* %I, i64 %C 
        %B = getelementptr i32* %A, i64 %D 
        ret i32* %B
; CHECK: @test7
; CHECK: %A.sum = add i64 %C, %D
; CHECK: getelementptr i32* %I, i64 %A.sum
}

define i8* @test8([10 x i32]* %X) {
        ;; Fold into the cast.
        %A = getelementptr [10 x i32]* %X, i64 0, i64 0 
        %B = bitcast i32* %A to i8*     
        ret i8* %B
; CHECK: @test8
; CHECK: bitcast [10 x i32]* %X to i8*
}

define i32 @test9() {
        %A = getelementptr { i32, double }* null, i32 0, i32 1
        %B = ptrtoint double* %A to i32        
        ret i32 %B
; CHECK: @test9
; CHECK: ret i32 8
}

define i1 @test10({ i32, i32 }* %x, { i32, i32 }* %y) {
        %tmp.1 = getelementptr { i32, i32 }* %x, i32 0, i32 1
        %tmp.3 = getelementptr { i32, i32 }* %y, i32 0, i32 1
        ;; seteq x, y
        %tmp.4 = icmp eq i32* %tmp.1, %tmp.3       
        ret i1 %tmp.4
; CHECK: @test10
; CHECK: icmp eq { i32, i32 }* %x, %y
}

define i1 @test11({ i32, i32 }* %X) {
        %P = getelementptr { i32, i32 }* %X, i32 0, i32 0 
        %Q = icmp eq i32* %P, null             
        ret i1 %Q
; CHECK: @test11
; CHECK: icmp eq { i32, i32 }* %X, null
}


; PR4748
define i32 @test12(%struct.A* %a) {
entry:
  %g3 = getelementptr %struct.A* %a, i32 0, i32 1
  store i32 10, i32* %g3, align 4

  %g4 = getelementptr %struct.A* %a, i32 0, i32 0
  
  %new_a = bitcast %struct.B* %g4 to %struct.A*

  %g5 = getelementptr %struct.A* %new_a, i32 0, i32 1	
  %a_a = load i32* %g5, align 4	
  ret i32 %a_a
; CHECK:      @test12
; CHECK:      getelementptr %struct.A* %a, i64 0, i32 1
; CHECK-NEXT: store i32 10, i32* %g3
; CHECK-NEXT: ret i32 10
}


; PR2235
%S = type { i32, [ 100 x i32] }
define i1 @test13(i64 %X, %S* %P) {
        %A = getelementptr inbounds %S* %P, i32 0, i32 1, i64 %X
        %B = getelementptr inbounds %S* %P, i32 0, i32 0
	%C = icmp eq i32* %A, %B
	ret i1 %C
; CHECK: @test13
; CHECK:    %C = icmp eq i64 %X, -1
}


@G = external global [3 x i8]      
define i8* @test14(i32 %Idx) {
        %idx = zext i32 %Idx to i64
        %tmp = getelementptr i8* getelementptr ([3 x i8]* @G, i32 0, i32 0), i64 %idx
        ret i8* %tmp
; CHECK: @test14
; CHECK: getelementptr [3 x i8]* @G, i64 0, i64 %idx
}


; Test folding of constantexpr geps into normal geps.
@Array = external global [40 x i32]
define i32 *@test15(i64 %X) {
        %A = getelementptr i32* getelementptr ([40 x i32]* @Array, i64 0, i64 0), i64 %X
        ret i32* %A
; CHECK: @test15
; CHECK: getelementptr [40 x i32]* @Array, i64 0, i64 %X
}


define i32* @test16(i32* %X, i32 %Idx) {
        %R = getelementptr i32* %X, i32 %Idx       
        ret i32* %R
; CHECK: @test16
; CHECK: sext i32 %Idx to i64
}


define i1 @test17(i16* %P, i32 %I, i32 %J) {
        %X = getelementptr inbounds i16* %P, i32 %I
        %Y = getelementptr inbounds i16* %P, i32 %J
        %C = icmp ult i16* %X, %Y
        ret i1 %C
; CHECK: @test17
; CHECK: %C = icmp slt i32 %I, %J 
}

define i1 @test18(i16* %P, i32 %I) {
        %X = getelementptr inbounds i16* %P, i32 %I
        %C = icmp ult i16* %X, %P
        ret i1 %C
; CHECK: @test18
; CHECK: %C = icmp slt i32 %I, 0
}

define i32 @test19(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr inbounds i32* %P, i32 %A
        %tmp.9 = getelementptr inbounds i32* %P, i32 %B
        %tmp.10 = icmp eq i32* %tmp.4, %tmp.9
        %tmp.11 = zext i1 %tmp.10 to i32
        ret i32 %tmp.11
; CHECK: @test19
; CHECK: icmp eq i32 %A, %B
}

define i32 @test20(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr inbounds i32* %P, i32 %A
        %tmp.6 = icmp eq i32* %tmp.4, %P
        %tmp.7 = zext i1 %tmp.6 to i32
        ret i32 %tmp.7
; CHECK: @test20
; CHECK: icmp eq i32 %A, 0
}


define i32 @test21() {
        %pbob1 = alloca %intstruct
        %pbob2 = getelementptr %intstruct* %pbob1
        %pbobel = getelementptr %intstruct* %pbob2, i64 0, i32 0
        %rval = load i32* %pbobel
        ret i32 %rval
; CHECK: @test21
; CHECK: getelementptr %intstruct* %pbob1, i64 0, i32 0
}


@A = global i32 1               ; <i32*> [#uses=1]
@B = global i32 2               ; <i32*> [#uses=1]

define i1 @test22() {
        %C = icmp ult i32* getelementptr (i32* @A, i64 1), 
                           getelementptr (i32* @B, i64 2) 
        ret i1 %C
; CHECK: @test22
; CHECK: icmp ult (i32* getelementptr inbounds (i32* @A, i64 1), i32* getelementptr (i32* @B, i64 2))
}


%X = type { [10 x i32], float }

define i1 @test23() {
        %A = getelementptr %X* null, i64 0, i32 0, i64 0                ; <i32*> [#uses=1]
        %B = icmp ne i32* %A, null              ; <i1> [#uses=1]
        ret i1 %B
; CHECK: @test23
; CHECK: ret i1 false
}

define void @test25() {
entry:
        %tmp = getelementptr { i64, i64, i64, i64 }* null, i32 0, i32 3         ; <i64*> [#uses=1]
        %tmp.upgrd.1 = load i64* %tmp           ; <i64> [#uses=1]
        %tmp8.ui = load i64* null               ; <i64> [#uses=1]
        %tmp8 = bitcast i64 %tmp8.ui to i64             ; <i64> [#uses=1]
        %tmp9 = and i64 %tmp8, %tmp.upgrd.1             ; <i64> [#uses=1]
        %sext = trunc i64 %tmp9 to i32          ; <i32> [#uses=1]
        %tmp27.i = sext i32 %sext to i64                ; <i64> [#uses=1]
        tail call void @foo25( i32 0, i64 %tmp27.i )
        unreachable
; CHECK: @test25
}

declare void @foo25(i32, i64)


; PR1637
define i1 @test26(i8* %arr) {
        %X = getelementptr i8* %arr, i32 1
        %Y = getelementptr i8* %arr, i32 1
        %test = icmp uge i8* %X, %Y
        ret i1 %test
; CHECK: @test26
; CHECK: ret i1 true
}

	%struct.__large_struct = type { [100 x i64] }
	%struct.compat_siginfo = type { i32, i32, i32, { [29 x i32] } }
	%struct.siginfo_t = type { i32, i32, i32, { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] } }
	%struct.sigval_t = type { i8* }

define i32 @test27(%struct.compat_siginfo* %to, %struct.siginfo_t* %from) {
entry:
	%from_addr = alloca %struct.siginfo_t*	
	%tmp344 = load %struct.siginfo_t** %from_addr, align 8	
	%tmp345 = getelementptr %struct.siginfo_t* %tmp344, i32 0, i32 3
	%tmp346 = getelementptr { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] }* %tmp345, i32 0, i32 0
	%tmp346347 = bitcast { i32, i32, [0 x i8], %struct.sigval_t, i32 }* %tmp346 to { i32, i32, %struct.sigval_t }*	
	%tmp348 = getelementptr { i32, i32, %struct.sigval_t }* %tmp346347, i32 0, i32 2
	%tmp349 = getelementptr %struct.sigval_t* %tmp348, i32 0, i32 0
	%tmp349350 = bitcast i8** %tmp349 to i32*
	%tmp351 = load i32* %tmp349350, align 8	
	%tmp360 = call i32 asm sideeffect "...",
        "=r,ir,*m,i,0,~{dirflag},~{fpsr},~{flags}"( i32 %tmp351,
         %struct.__large_struct* null, i32 -14, i32 0 )
	unreachable
; CHECK: @test27
}

; PR1978
	%struct.x = type <{ i8 }>
@.str = internal constant [6 x i8] c"Main!\00"	
@.str1 = internal constant [12 x i8] c"destroy %p\0A\00"	

define i32 @test28() nounwind  {
entry:
	%orientations = alloca [1 x [1 x %struct.x]]
	%tmp3 = call i32 @puts( i8* getelementptr ([6 x i8]* @.str, i32 0, i32 0) ) nounwind 
	%tmp45 = getelementptr inbounds [1 x [1 x %struct.x]]* %orientations, i32 1, i32 0, i32 0
	%orientations62 = getelementptr [1 x [1 x %struct.x]]* %orientations, i32 0, i32 0, i32 0
	br label %bb10

bb10:
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb10 ]
	%tmp.0.reg2mem.0.rec = mul i32 %indvar, -1	
	%tmp12.rec = add i32 %tmp.0.reg2mem.0.rec, -1	
	%tmp12 = getelementptr inbounds %struct.x* %tmp45, i32 %tmp12.rec
	%tmp16 = call i32 (i8*, ...)* @printf( i8* getelementptr ([12 x i8]* @.str1, i32 0, i32 0), %struct.x* %tmp12 ) nounwind
	%tmp84 = icmp eq %struct.x* %tmp12, %orientations62
	%indvar.next = add i32 %indvar, 1
	br i1 %tmp84, label %bb17, label %bb10

bb17:	
	ret i32 0
; CHECK: @test28
; CHECK: icmp eq i32 %indvar, 0
}

declare i32 @puts(i8*)

declare i32 @printf(i8*, ...)




; rdar://6762290
	%T = type <{ i64, i64, i64 }>
define i32 @test29(i8* %start, i32 %X) nounwind {
entry:
	%tmp3 = load i64* null		
	%add.ptr = getelementptr i8* %start, i64 %tmp3
	%tmp158 = load i32* null
	%add.ptr159 = getelementptr %T* null, i32 %tmp158
	%add.ptr209 = getelementptr i8* %start, i64 0
	%add.ptr212 = getelementptr i8* %add.ptr209, i32 %X
	%cmp214 = icmp ugt i8* %add.ptr212, %add.ptr
	br i1 %cmp214, label %if.then216, label %if.end363

if.then216:
	ret i32 1

if.end363:
	ret i32 0
; CHECK: @test29
}


; PR3694
define i32 @test30(i32 %m, i32 %n) nounwind {
entry:
	%0 = alloca i32, i32 %n, align 4
	%1 = bitcast i32* %0 to [0 x i32]*
	call void @test30f(i32* %0) nounwind
	%2 = getelementptr [0 x i32]* %1, i32 0, i32 %m
	%3 = load i32* %2, align 4
	ret i32 %3
; CHECK: @test30
; CHECK: getelementptr i32
}

declare void @test30f(i32*)



define i1 @test31(i32* %A) {
        %B = getelementptr i32* %A, i32 1
        %C = getelementptr i32* %A, i64 1
        %V = icmp eq i32* %B, %C 
        ret i1 %V
; CHECK: @test31
; CHECK: ret i1 true
}


; PR1345
define i8* @test32(i8* %v) {
	%A = alloca [4 x i8*], align 16
	%B = getelementptr [4 x i8*]* %A, i32 0, i32 0
	store i8* null, i8** %B
	%C = bitcast [4 x i8*]* %A to { [16 x i8] }*
	%D = getelementptr { [16 x i8] }* %C, i32 0, i32 0, i32 8
	%E = bitcast i8* %D to i8**
	store i8* %v, i8** %E
	%F = getelementptr [4 x i8*]* %A, i32 0, i32 2	
	%G = load i8** %F
	ret i8* %G
; CHECK: @test32
; CHECK: %D = getelementptr [4 x i8*]* %A, i64 0, i64 1
; CHECK: %F = getelementptr [4 x i8*]* %A, i64 0, i64 2
}

; PR3290
%struct.Key = type { { i32, i32 } }
%struct.anon = type <{ i8, [3 x i8], i32 }>

define i32 *@test33(%struct.Key *%A) {
	%B = bitcast %struct.Key* %A to %struct.anon*
        %C = getelementptr %struct.anon* %B, i32 0, i32 2 
	ret i32 *%C
; CHECK: @test33
; CHECK: getelementptr %struct.Key* %A, i64 0, i32 0, i32 1
}



	%T2 = type { i8*, i8 }
define i8* @test34(i8* %Val, i64 %V) nounwind {
entry:
	%A = alloca %T2, align 8	
	%mrv_gep = bitcast %T2* %A to i64*
	%B = getelementptr %T2* %A, i64 0, i32 0
        
      	store i64 %V, i64* %mrv_gep
	%C = load i8** %B, align 8
	ret i8* %C
; CHECK: @test34
; CHECK: %V.c = inttoptr i64 %V to i8*
; CHECK: ret i8* %V.c
}

%t0 = type { i8*, [19 x i8] }
%t1 = type { i8*, [0 x i8] }

@array = external global [11 x i8]

@s = external global %t0
@"\01LC8" = external constant [17 x i8]

; Instcombine should be able to fold this getelementptr.

define i32 @test35() nounwind {
  call i32 (i8*, ...)* @printf(i8* getelementptr ([17 x i8]* @"\01LC8", i32 0, i32 0),
             i8* getelementptr (%t1* bitcast (%t0* @s to %t1*), i32 0, i32 1, i32 0)) nounwind
  ret i32 0
; CHECK: @test35
; CHECK: call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([17 x i8]* @"\01LC8", i64 0, i64 0), i8* getelementptr inbounds (%t0* @s, i64 0, i32 1, i64 0)) nounwind
}

; Instcombine should constant-fold the GEP so that indices that have
; static array extents are within bounds of those array extents.
; In the below, -1 is not in the range [0,11). After the transformation,
; the same address is computed, but 3 is in the range of [0,11).

define i8* @test36() nounwind {
  ret i8* getelementptr ([11 x i8]* @array, i32 0, i64 -1)
; CHECK: @test36
; CHECK: ret i8* getelementptr ([11 x i8]* @array, i64 1676976733973595601, i64 4)
}

; Instcombine shouldn't assume that gep(A,0,1) != gep(A,1,0).
@A37 = external constant [1 x i8]
define i1 @test37() nounwind {
; CHECK: @test37
; CHECK: ret i1 true
  %t = icmp eq i8* getelementptr ([1 x i8]* @A37, i64 0, i64 1),
                   getelementptr ([1 x i8]* @A37, i64 1, i64 0)
  ret i1 %t
}

; Test index promotion
define i32* @test38(i32* %I, i32 %n) {
        %A = getelementptr i32* %I, i32 %n
        ret i32* %A
; CHECK: @test38
; CHECK: = sext i32 %n to i64
; CHECK: %A = getelementptr i32* %I, i64 %
}

; Test that we don't duplicate work when the second gep is a "bitcast".
%pr10322_t = type { i8* }
declare void @pr10322_f2(%pr10322_t*)
declare void @pr10322_f3(i8**)
define void @pr10322_f1(%pr10322_t* %foo) {
entry:
  %arrayidx8 = getelementptr inbounds %pr10322_t* %foo, i64 2
  call void @pr10322_f2(%pr10322_t* %arrayidx8) nounwind
  %tmp2 = getelementptr inbounds %pr10322_t* %arrayidx8, i64 0, i32 0
  call void @pr10322_f3(i8** %tmp2) nounwind
  ret void

; CHECK: @pr10322_f1
; CHECK: %tmp2 = getelementptr inbounds %pr10322_t* %arrayidx8, i64 0, i32 0
}

; Test that we combine the last two geps in this sequence, before we
; would wait for gep1 and gep2 to be combined and never combine 2 and 3.
%three_gep_t = type {i32}
%three_gep_t2 = type {%three_gep_t}

define void @three_gep_f(%three_gep_t2* %x) {
  %gep1 = getelementptr %three_gep_t2* %x, i64 2
  call void @three_gep_h(%three_gep_t2* %gep1)
  %gep2 = getelementptr %three_gep_t2* %gep1, i64 0, i32 0
  %gep3 = getelementptr %three_gep_t* %gep2, i64 0, i32 0
  call void @three_gep_g(i32* %gep3)

; CHECK: @three_gep_f
; CHECK: %gep3 = getelementptr %three_gep_t2* %gep1, i64 0, i32 0, i32 0
  ret void
}

declare void @three_gep_g(i32*)
declare void @three_gep_h(%three_gep_t2*)
