; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64-p1:16:16-p2:32:32:32-p3:64:64:64"

%intstruct = type { i32 }
%pair = type { i32, i32 }
%struct.B = type { double }
%struct.A = type { %struct.B, i32, i32 }
%struct.C = type { [7 x i8] }


@Global = constant [10 x i8] c"helloworld"
@Global_as1 = addrspace(1) constant [10 x i8] c"helloworld"

; Test noop elimination
define i32* @test1(i32* %I) {
        %A = getelementptr i32, i32* %I, i64 0
        ret i32* %A
; CHECK-LABEL: @test1(
; CHECK: ret i32* %I
}

define i32 addrspace(1)* @test1_as1(i32 addrspace(1)* %I) {
  %A = getelementptr i32, i32 addrspace(1)* %I, i64 0
  ret i32 addrspace(1)* %A
; CHECK-LABEL: @test1_as1(
; CHECK: ret i32 addrspace(1)* %I
}

; Test noop elimination
define i32* @test2(i32* %I) {
        %A = getelementptr i32, i32* %I
        ret i32* %A
; CHECK-LABEL: @test2(
; CHECK: ret i32* %I
}

; Test that two array indexing geps fold
define i32* @test3(i32* %I) {
        %A = getelementptr i32, i32* %I, i64 17
        %B = getelementptr i32, i32* %A, i64 4
        ret i32* %B
; CHECK-LABEL: @test3(
; CHECK: getelementptr i32, i32* %I, i64 21
}

; Test that two getelementptr insts fold
define i32* @test4({ i32 }* %I) {
        %A = getelementptr { i32 }, { i32 }* %I, i64 1
        %B = getelementptr { i32 }, { i32 }* %A, i64 0, i32 0
        ret i32* %B
; CHECK-LABEL: @test4(
; CHECK: getelementptr { i32 }, { i32 }* %I, i64 1, i32 0
}

define void @test5(i8 %B) {
        ; This should be turned into a constexpr instead of being an instruction
        %A = getelementptr [10 x i8], [10 x i8]* @Global, i64 0, i64 4
        store i8 %B, i8* %A
        ret void
; CHECK-LABEL: @test5(
; CHECK: store i8 %B, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @Global, i64 0, i64 4)
}

define void @test5_as1(i8 %B) {
        ; This should be turned into a constexpr instead of being an instruction
        %A = getelementptr [10 x i8], [10 x i8] addrspace(1)* @Global_as1, i16 0, i16 4
        store i8 %B, i8 addrspace(1)* %A
        ret void
; CHECK-LABEL: @test5_as1(
; CHECK: store i8 %B, i8 addrspace(1)* getelementptr inbounds ([10 x i8], [10 x i8] addrspace(1)* @Global_as1, i16 0, i16 4)
}

%as1_ptr_struct = type { i32 addrspace(1)* }
%as2_ptr_struct = type { i32 addrspace(2)* }

@global_as2 = addrspace(2) global i32 zeroinitializer
@global_as1_as2_ptr = addrspace(1) global %as2_ptr_struct { i32 addrspace(2)* @global_as2 }

; This should be turned into a constexpr instead of being an instruction
define void @test_evaluate_gep_nested_as_ptrs(i32 addrspace(2)* %B) {
; CHECK-LABEL: @test_evaluate_gep_nested_as_ptrs(
; CHECK-NEXT: store i32 addrspace(2)* %B, i32 addrspace(2)* addrspace(1)* getelementptr inbounds (%as2_ptr_struct, %as2_ptr_struct addrspace(1)* @global_as1_as2_ptr, i16 0, i32 0), align 8
; CHECK-NEXT: ret void
  %A = getelementptr %as2_ptr_struct, %as2_ptr_struct addrspace(1)* @global_as1_as2_ptr, i16 0, i32 0
  store i32 addrspace(2)* %B, i32 addrspace(2)* addrspace(1)* %A
  ret void
}

@arst = addrspace(1) global [4 x i8 addrspace(2)*] zeroinitializer

define void @test_evaluate_gep_as_ptrs_array(i8 addrspace(2)* %B) {
; CHECK-LABEL: @test_evaluate_gep_as_ptrs_array(
; CHECK-NEXT: store i8 addrspace(2)* %B, i8 addrspace(2)* addrspace(1)* getelementptr inbounds ([4 x i8 addrspace(2)*], [4 x i8 addrspace(2)*] addrspace(1)* @arst, i16 0, i16 2), align 4

; CHECK-NEXT: ret void
  %A = getelementptr [4 x i8 addrspace(2)*], [4 x i8 addrspace(2)*] addrspace(1)* @arst, i16 0, i16 2
  store i8 addrspace(2)* %B, i8 addrspace(2)* addrspace(1)* %A
  ret void
}

define i32* @test7(i32* %I, i64 %C, i64 %D) {
        %A = getelementptr i32, i32* %I, i64 %C
        %B = getelementptr i32, i32* %A, i64 %D
        ret i32* %B
; CHECK-LABEL: @test7(
; CHECK: %A = getelementptr i32, i32* %I, i64 %C
; CHECK: %B = getelementptr i32, i32* %A, i64 %D
}

define i8* @test8([10 x i32]* %X) {
        ;; Fold into the cast.
        %A = getelementptr [10 x i32], [10 x i32]* %X, i64 0, i64 0
        %B = bitcast i32* %A to i8*
        ret i8* %B
; CHECK-LABEL: @test8(
; CHECK: bitcast [10 x i32]* %X to i8*
}

define i32 @test9() {
        %A = getelementptr { i32, double }, { i32, double }* null, i32 0, i32 1
        %B = ptrtoint double* %A to i32
        ret i32 %B
; CHECK-LABEL: @test9(
; CHECK: ret i32 8
}

define i1 @test10({ i32, i32 }* %x, { i32, i32 }* %y) {
        %tmp.1 = getelementptr { i32, i32 }, { i32, i32 }* %x, i32 0, i32 1
        %tmp.3 = getelementptr { i32, i32 }, { i32, i32 }* %y, i32 0, i32 1
        ;; seteq x, y
        %tmp.4 = icmp eq i32* %tmp.1, %tmp.3
        ret i1 %tmp.4
; CHECK-LABEL: @test10(
; CHECK: icmp eq { i32, i32 }* %x, %y
}

define i1 @test11({ i32, i32 }* %X) {
        %P = getelementptr { i32, i32 }, { i32, i32 }* %X, i32 0, i32 0
        %Q = icmp eq i32* %P, null
        ret i1 %Q
; CHECK-LABEL: @test11(
; CHECK: icmp eq { i32, i32 }* %X, null
}


; PR4748
define i32 @test12(%struct.A* %a) {
entry:
  %g3 = getelementptr %struct.A, %struct.A* %a, i32 0, i32 1
  store i32 10, i32* %g3, align 4

  %g4 = getelementptr %struct.A, %struct.A* %a, i32 0, i32 0

  %new_a = bitcast %struct.B* %g4 to %struct.A*

  %g5 = getelementptr %struct.A, %struct.A* %new_a, i32 0, i32 1
  %a_a = load i32, i32* %g5, align 4
  ret i32 %a_a
; CHECK-LABEL:      @test12(
; CHECK:      getelementptr %struct.A, %struct.A* %a, i64 0, i32 1
; CHECK-NEXT: store i32 10, i32* %g3
; CHECK-NEXT: ret i32 10
}


; PR2235
%S = type { i32, [ 100 x i32] }
define i1 @test13(i64 %X, %S* %P) {
        %A = getelementptr inbounds %S, %S* %P, i32 0, i32 1, i64 %X
        %B = getelementptr inbounds %S, %S* %P, i32 0, i32 0
	%C = icmp eq i32* %A, %B
	ret i1 %C
; CHECK-LABEL: @test13(
; CHECK:    %C = icmp eq i64 %X, -1
}

; This is a test of icmp + shl nuw in disguise - 4611... is 0x3fff...
define <2 x i1> @test13_vector(<2 x i64> %X, <2 x %S*> %P) nounwind {
; CHECK-LABEL: @test13_vector(
; CHECK-NEXT:    [[C:%.*]] = icmp eq <2 x i64> %X, <i64 4611686018427387903, i64 4611686018427387903>
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %A = getelementptr inbounds %S, <2 x %S*> %P, <2 x i64> zeroinitializer, <2 x i32> <i32 1, i32 1>, <2 x i64> %X
  %B = getelementptr inbounds %S, <2 x %S*> %P, <2 x i64> <i64 0, i64 0>, <2 x i32> <i32 0, i32 0>
  %C = icmp eq <2 x i32*> %A, %B
  ret <2 x i1> %C
}

define i1 @test13_as1(i16 %X, %S addrspace(1)* %P) {
; CHECK-LABEL: @test13_as1(
; CHECK-NEXT:  %C = icmp eq i16 %X, -1
; CHECK-NEXT: ret i1 %C
  %A = getelementptr inbounds %S, %S addrspace(1)* %P, i16 0, i32 1, i16 %X
  %B = getelementptr inbounds %S, %S addrspace(1)* %P, i16 0, i32 0
  %C = icmp eq i32 addrspace(1)* %A, %B
  ret i1 %C
}

; This is a test of icmp + shl nuw in disguise - 16383 is 0x3fff.
define <2 x i1> @test13_vector_as1(<2 x i16> %X, <2 x %S addrspace(1)*> %P) {
; CHECK-LABEL: @test13_vector_as1(
; CHECK-NEXT:    [[C:%.*]] = icmp eq <2 x i16> %X, <i16 16383, i16 16383>
; CHECK-NEXT:    ret <2 x i1> [[C]]
;
  %A = getelementptr inbounds %S, <2 x %S addrspace(1)*> %P, <2 x i16> <i16 0, i16 0>, <2 x i32> <i32 1, i32 1>, <2 x i16> %X
  %B = getelementptr inbounds %S, <2 x %S addrspace(1)*> %P, <2 x i16> <i16 0, i16 0>, <2 x i32> <i32 0, i32 0>
  %C = icmp eq <2 x i32 addrspace(1)*> %A, %B
  ret <2 x i1> %C
}

define i1 @test13_i32(i32 %X, %S* %P) {
; CHECK-LABEL: @test13_i32(
; CHECK: %C = icmp eq i32 %X, -1
  %A = getelementptr inbounds %S, %S* %P, i32 0, i32 1, i32 %X
  %B = getelementptr inbounds %S, %S* %P, i32 0, i32 0
  %C = icmp eq i32* %A, %B
  ret i1 %C
}

define i1 @test13_i16(i16 %X, %S* %P) {
; CHECK-LABEL: @test13_i16(
; CHECK: %C = icmp eq i16 %X, -1
  %A = getelementptr inbounds %S, %S* %P, i16 0, i32 1, i16 %X
  %B = getelementptr inbounds %S, %S* %P, i16 0, i32 0
  %C = icmp eq i32* %A, %B
  ret i1 %C
}

define i1 @test13_i128(i128 %X, %S* %P) {
; CHECK-LABEL: @test13_i128(
; CHECK: %C = icmp eq i64 %1, -1
  %A = getelementptr inbounds %S, %S* %P, i128 0, i32 1, i128 %X
  %B = getelementptr inbounds %S, %S* %P, i128 0, i32 0
  %C = icmp eq i32* %A, %B
  ret i1 %C
}


@G = external global [3 x i8]
define i8* @test14(i32 %Idx) {
        %idx = zext i32 %Idx to i64
        %tmp = getelementptr i8, i8* getelementptr ([3 x i8], [3 x i8]* @G, i32 0, i32 0), i64 %idx
        ret i8* %tmp
; CHECK-LABEL: @test14(
; CHECK: getelementptr [3 x i8], [3 x i8]* @G, i64 0, i64 %idx
}


; Test folding of constantexpr geps into normal geps.
@Array = external global [40 x i32]
define i32 *@test15(i64 %X) {
        %A = getelementptr i32, i32* getelementptr ([40 x i32], [40 x i32]* @Array, i64 0, i64 0), i64 %X
        ret i32* %A
; CHECK-LABEL: @test15(
; CHECK: getelementptr [40 x i32], [40 x i32]* @Array, i64 0, i64 %X
}


define i32* @test16(i32* %X, i32 %Idx) {
        %R = getelementptr i32, i32* %X, i32 %Idx
        ret i32* %R
; CHECK-LABEL: @test16(
; CHECK: sext i32 %Idx to i64
}


define i1 @test17(i16* %P, i32 %I, i32 %J) {
        %X = getelementptr inbounds i16, i16* %P, i32 %I
        %Y = getelementptr inbounds i16, i16* %P, i32 %J
        %C = icmp ult i16* %X, %Y
        ret i1 %C
; CHECK-LABEL: @test17(
; CHECK: %C = icmp slt i32 %I, %J
}

define i1 @test18(i16* %P, i32 %I) {
        %X = getelementptr inbounds i16, i16* %P, i32 %I
        %C = icmp ult i16* %X, %P
        ret i1 %C
; CHECK-LABEL: @test18(
; CHECK: %C = icmp slt i32 %I, 0
}

; Larger than the pointer size for a non-zero address space
define i1 @test18_as1(i16 addrspace(1)* %P, i32 %I) {
; CHECK-LABEL: @test18_as1(
; CHECK-NEXT: %1 = trunc i32 %I to i16
; CHECK-NEXT: %C = icmp slt i16 %1, 0
; CHECK-NEXT: ret i1 %C
  %X = getelementptr inbounds i16, i16 addrspace(1)* %P, i32 %I
  %C = icmp ult i16 addrspace(1)* %X, %P
  ret i1 %C
}

; Smaller than the pointer size for a non-zero address space
define i1 @test18_as1_i32(i16 addrspace(1)* %P, i32 %I) {
; CHECK-LABEL: @test18_as1_i32(
; CHECK-NEXT: %1 = trunc i32 %I to i16
; CHECK-NEXT: %C = icmp slt i16 %1, 0
; CHECK-NEXT: ret i1 %C
  %X = getelementptr inbounds i16, i16 addrspace(1)* %P, i32 %I
  %C = icmp ult i16 addrspace(1)* %X, %P
  ret i1 %C
}

; Smaller than pointer size
define i1 @test18_i16(i16* %P, i16 %I) {
; CHECK-LABEL: @test18_i16(
; CHECK: %C = icmp slt i16 %I, 0
  %X = getelementptr inbounds i16, i16* %P, i16 %I
  %C = icmp ult i16* %X, %P
  ret i1 %C
}

; Same as pointer size
define i1 @test18_i64(i16* %P, i64 %I) {
; CHECK-LABEL: @test18_i64(
; CHECK: %C = icmp slt i64 %I, 0
  %X = getelementptr inbounds i16, i16* %P, i64 %I
  %C = icmp ult i16* %X, %P
  ret i1 %C
}

; Larger than the pointer size
define i1 @test18_i128(i16* %P, i128 %I) {
; CHECK-LABEL: @test18_i128(
; CHECK: %C = icmp slt i64 %1, 0
  %X = getelementptr inbounds i16, i16* %P, i128 %I
  %C = icmp ult i16* %X, %P
  ret i1 %C
}

define i32 @test19(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr inbounds i32, i32* %P, i32 %A
        %tmp.9 = getelementptr inbounds i32, i32* %P, i32 %B
        %tmp.10 = icmp eq i32* %tmp.4, %tmp.9
        %tmp.11 = zext i1 %tmp.10 to i32
        ret i32 %tmp.11
; CHECK-LABEL: @test19(
; CHECK: icmp eq i32 %A, %B
}

define i32 @test20(i32* %P, i32 %A, i32 %B) {
        %tmp.4 = getelementptr inbounds i32, i32* %P, i32 %A
        %tmp.6 = icmp eq i32* %tmp.4, %P
        %tmp.7 = zext i1 %tmp.6 to i32
        ret i32 %tmp.7
; CHECK-LABEL: @test20(
; CHECK: icmp eq i32 %A, 0
}

define i32 @test20_as1(i32 addrspace(1)* %P, i32 %A, i32 %B) {
  %tmp.4 = getelementptr inbounds i32, i32 addrspace(1)* %P, i32 %A
  %tmp.6 = icmp eq i32 addrspace(1)* %tmp.4, %P
  %tmp.7 = zext i1 %tmp.6 to i32
  ret i32 %tmp.7
; CHECK-LABEL: @test20_as1(
; CHECK: icmp eq i16 %1, 0
}


define i32 @test21() {
        %pbob1 = alloca %intstruct
        %pbob2 = getelementptr %intstruct, %intstruct* %pbob1
        %pbobel = getelementptr %intstruct, %intstruct* %pbob2, i64 0, i32 0
        %rval = load i32, i32* %pbobel
        ret i32 %rval
; CHECK-LABEL: @test21(
; CHECK: getelementptr inbounds %intstruct, %intstruct* %pbob1, i64 0, i32 0
}


@A = global i32 1               ; <i32*> [#uses=1]
@B = global i32 2               ; <i32*> [#uses=1]

define i1 @test22() {
        %C = icmp ult i32* getelementptr (i32, i32* @A, i64 1),
                           getelementptr (i32, i32* @B, i64 2)
        ret i1 %C
; CHECK-LABEL: @test22(
; CHECK: icmp ult (i32* getelementptr inbounds (i32, i32* @A, i64 1), i32* getelementptr (i32, i32* @B, i64 2))
}


%X = type { [10 x i32], float }

define i1 @test23() {
        %A = getelementptr %X, %X* null, i64 0, i32 0, i64 0                ; <i32*> [#uses=1]
        %B = icmp ne i32* %A, null              ; <i1> [#uses=1]
        ret i1 %B
; CHECK-LABEL: @test23(
; CHECK: ret i1 false
}

define void @test25() {
entry:
        %tmp = getelementptr { i64, i64, i64, i64 }, { i64, i64, i64, i64 }* null, i32 0, i32 3         ; <i64*> [#uses=1]
        %tmp.upgrd.1 = load i64, i64* %tmp           ; <i64> [#uses=1]
        %tmp8.ui = load i64, i64* null               ; <i64> [#uses=1]
        %tmp8 = bitcast i64 %tmp8.ui to i64             ; <i64> [#uses=1]
        %tmp9 = and i64 %tmp8, %tmp.upgrd.1             ; <i64> [#uses=1]
        %sext = trunc i64 %tmp9 to i32          ; <i32> [#uses=1]
        %tmp27.i = sext i32 %sext to i64                ; <i64> [#uses=1]
        tail call void @foo25( i32 0, i64 %tmp27.i )
        unreachable
; CHECK-LABEL: @test25(
}

declare void @foo25(i32, i64)


; PR1637
define i1 @test26(i8* %arr) {
        %X = getelementptr i8, i8* %arr, i32 1
        %Y = getelementptr i8, i8* %arr, i32 1
        %test = icmp uge i8* %X, %Y
        ret i1 %test
; CHECK-LABEL: @test26(
; CHECK: ret i1 true
}

	%struct.__large_struct = type { [100 x i64] }
	%struct.compat_siginfo = type { i32, i32, i32, { [29 x i32] } }
	%struct.siginfo_t = type { i32, i32, i32, { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] } }
	%struct.sigval_t = type { i8* }

define i32 @test27(%struct.compat_siginfo* %to, %struct.siginfo_t* %from) {
entry:
	%from_addr = alloca %struct.siginfo_t*
	%tmp344 = load %struct.siginfo_t*, %struct.siginfo_t** %from_addr, align 8
	%tmp345 = getelementptr %struct.siginfo_t, %struct.siginfo_t* %tmp344, i32 0, i32 3
	%tmp346 = getelementptr { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] }, { { i32, i32, [0 x i8], %struct.sigval_t, i32 }, [88 x i8] }* %tmp345, i32 0, i32 0
	%tmp346347 = bitcast { i32, i32, [0 x i8], %struct.sigval_t, i32 }* %tmp346 to { i32, i32, %struct.sigval_t }*
	%tmp348 = getelementptr { i32, i32, %struct.sigval_t }, { i32, i32, %struct.sigval_t }* %tmp346347, i32 0, i32 2
	%tmp349 = getelementptr %struct.sigval_t, %struct.sigval_t* %tmp348, i32 0, i32 0
	%tmp349350 = bitcast i8** %tmp349 to i32*
	%tmp351 = load i32, i32* %tmp349350, align 8
	%tmp360 = call i32 asm sideeffect "...",
        "=r,ir,*m,i,0,~{dirflag},~{fpsr},~{flags}"( i32 %tmp351,
         %struct.__large_struct* null, i32 -14, i32 0 )
	unreachable
; CHECK-LABEL: @test27(
}

; PR1978
	%struct.x = type <{ i8 }>
@.str = internal constant [6 x i8] c"Main!\00"
@.str1 = internal constant [12 x i8] c"destroy %p\0A\00"

define i32 @test28() nounwind  {
entry:
	%orientations = alloca [1 x [1 x %struct.x]]
	%tmp3 = call i32 @puts( i8* getelementptr ([6 x i8], [6 x i8]* @.str, i32 0, i32 0) ) nounwind
	%tmp45 = getelementptr inbounds [1 x [1 x %struct.x]], [1 x [1 x %struct.x]]* %orientations, i32 1, i32 0, i32 0
	%orientations62 = getelementptr [1 x [1 x %struct.x]], [1 x [1 x %struct.x]]* %orientations, i32 0, i32 0, i32 0
	br label %bb10

bb10:
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %bb10 ]
	%tmp.0.reg2mem.0.rec = mul i32 %indvar, -1
	%tmp12.rec = add i32 %tmp.0.reg2mem.0.rec, -1
	%tmp12 = getelementptr inbounds %struct.x, %struct.x* %tmp45, i32 %tmp12.rec
	%tmp16 = call i32 (i8*, ...) @printf( i8* getelementptr ([12 x i8], [12 x i8]* @.str1, i32 0, i32 0), %struct.x* %tmp12 ) nounwind
	%tmp84 = icmp eq %struct.x* %tmp12, %orientations62
	%indvar.next = add i32 %indvar, 1
	br i1 %tmp84, label %bb17, label %bb10

bb17:
	ret i32 0
; CHECK-LABEL: @test28(
; CHECK: icmp eq i32 %indvar, 0
}

declare i32 @puts(i8*)

declare i32 @printf(i8*, ...)




; rdar://6762290
	%T = type <{ i64, i64, i64 }>
define i32 @test29(i8* %start, i32 %X) nounwind {
entry:
	%tmp3 = load i64, i64* null
	%add.ptr = getelementptr i8, i8* %start, i64 %tmp3
	%tmp158 = load i32, i32* null
	%add.ptr159 = getelementptr %T, %T* null, i32 %tmp158
	%add.ptr209 = getelementptr i8, i8* %start, i64 0
	%add.ptr212 = getelementptr i8, i8* %add.ptr209, i32 %X
	%cmp214 = icmp ugt i8* %add.ptr212, %add.ptr
	br i1 %cmp214, label %if.then216, label %if.end363

if.then216:
	ret i32 1

if.end363:
	ret i32 0
; CHECK-LABEL: @test29(
}


; PR3694
define i32 @test30(i32 %m, i32 %n) nounwind {
entry:
	%0 = alloca i32, i32 %n, align 4
	%1 = bitcast i32* %0 to [0 x i32]*
	call void @test30f(i32* %0) nounwind
	%2 = getelementptr [0 x i32], [0 x i32]* %1, i32 0, i32 %m
	%3 = load i32, i32* %2, align 4
	ret i32 %3
; CHECK-LABEL: @test30(
; CHECK: getelementptr i32
}

declare void @test30f(i32*)



define i1 @test31(i32* %A) {
        %B = getelementptr i32, i32* %A, i32 1
        %C = getelementptr i32, i32* %A, i64 1
        %V = icmp eq i32* %B, %C
        ret i1 %V
; CHECK-LABEL: @test31(
; CHECK: ret i1 true
}


; PR1345
define i8* @test32(i8* %v) {
	%A = alloca [4 x i8*], align 16
	%B = getelementptr [4 x i8*], [4 x i8*]* %A, i32 0, i32 0
	store i8* null, i8** %B
	%C = bitcast [4 x i8*]* %A to { [16 x i8] }*
	%D = getelementptr { [16 x i8] }, { [16 x i8] }* %C, i32 0, i32 0, i32 8
	%E = bitcast i8* %D to i8**
	store i8* %v, i8** %E
	%F = getelementptr [4 x i8*], [4 x i8*]* %A, i32 0, i32 2
	%G = load i8*, i8** %F
	ret i8* %G
; CHECK-LABEL: @test32(
; CHECK: %D = getelementptr inbounds [4 x i8*], [4 x i8*]* %A, i64 0, i64 1
; CHECK: %F = getelementptr inbounds [4 x i8*], [4 x i8*]* %A, i64 0, i64 2
}

; PR3290
%struct.Key = type { { i32, i32 } }
%struct.anon = type <{ i8, [3 x i8], i32 }>

define i32* @test33(%struct.Key* %A) {
; CHECK-LABEL: @test33(
; CHECK: getelementptr %struct.Key, %struct.Key* %A, i64 0, i32 0, i32 1
  %B = bitcast %struct.Key* %A to %struct.anon*
  %C = getelementptr %struct.anon, %struct.anon* %B, i32 0, i32 2
  ret i32* %C
}

define i32 addrspace(1)* @test33_as1(%struct.Key addrspace(1)* %A) {
; CHECK-LABEL: @test33_as1(
; CHECK: getelementptr %struct.Key, %struct.Key addrspace(1)* %A, i16 0, i32 0, i32 1
  %B = bitcast %struct.Key addrspace(1)* %A to %struct.anon addrspace(1)*
  %C = getelementptr %struct.anon, %struct.anon addrspace(1)* %B, i32 0, i32 2
  ret i32 addrspace(1)* %C
}

define i32 addrspace(1)* @test33_array_as1([10 x i32] addrspace(1)* %A) {
; CHECK-LABEL: @test33_array_as1(
; CHECK: getelementptr [10 x i32], [10 x i32] addrspace(1)* %A, i16 0, i16 2
  %B = bitcast [10 x i32] addrspace(1)* %A to [5 x i32] addrspace(1)*
  %C = getelementptr [5 x i32], [5 x i32] addrspace(1)* %B, i32 0, i32 2
  ret i32 addrspace(1)* %C
}

; Make sure the GEP indices use the right pointer sized integer
define i32 addrspace(1)* @test33_array_struct_as1([10 x %struct.Key] addrspace(1)* %A) {
; CHECK-LABEL: @test33_array_struct_as1(
; CHECK: getelementptr [10 x %struct.Key], [10 x %struct.Key] addrspace(1)* %A, i16 0, i16 1, i32 0, i32 0
  %B = bitcast [10 x %struct.Key] addrspace(1)* %A to [20 x i32] addrspace(1)*
  %C = getelementptr [20 x i32], [20 x i32] addrspace(1)* %B, i32 0, i32 2
  ret i32 addrspace(1)* %C
}

define i32 addrspace(1)* @test33_addrspacecast(%struct.Key* %A) {
; CHECK-LABEL: @test33_addrspacecast(
; CHECK: %C = getelementptr %struct.Key, %struct.Key* %A, i64 0, i32 0, i32 1
; CHECK-NEXT: addrspacecast i32* %C to i32 addrspace(1)*
; CHECK-NEXT: ret
  %B = addrspacecast %struct.Key* %A to %struct.anon addrspace(1)*
  %C = getelementptr %struct.anon, %struct.anon addrspace(1)* %B, i32 0, i32 2
  ret i32 addrspace(1)* %C
}

	%T2 = type { i8*, i8 }
define i8* @test34(i8* %Val, i64 %V) nounwind {
entry:
	%A = alloca %T2, align 8
	%mrv_gep = bitcast %T2* %A to i64*
	%B = getelementptr %T2, %T2* %A, i64 0, i32 0

      	store i64 %V, i64* %mrv_gep
	%C = load i8*, i8** %B, align 8
	ret i8* %C
; CHECK-LABEL: @test34(
; CHECK: %[[C:.*]] = inttoptr i64 %V to i8*
; CHECK: ret i8* %[[C]]
}

%t0 = type { i8*, [19 x i8] }
%t1 = type { i8*, [0 x i8] }

@array = external global [11 x i8]

@s = external global %t0
@"\01LC8" = external constant [17 x i8]

; Instcombine should be able to fold this getelementptr.

define i32 @test35() nounwind {
  call i32 (i8*, ...) @printf(i8* getelementptr ([17 x i8], [17 x i8]* @"\01LC8", i32 0, i32 0),
             i8* getelementptr (%t1, %t1* bitcast (%t0* @s to %t1*), i32 0, i32 1, i32 0)) nounwind
  ret i32 0
; CHECK-LABEL: @test35(
; CHECK: call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @"\01LC8", i64 0, i64 0), i8* getelementptr inbounds (%t0, %t0* @s, i64 0, i32 1, i64 0)) [[NUW:#[0-9]+]]
}

; Don't treat signed offsets as unsigned.
define i8* @test36() nounwind {
  ret i8* getelementptr ([11 x i8], [11 x i8]* @array, i32 0, i64 -1)
; CHECK-LABEL: @test36(
; CHECK: ret i8* getelementptr ([11 x i8], [11 x i8]* @array, i64 0, i64 -1)
}

; Instcombine shouldn't assume that gep(A,0,1) != gep(A,1,0).
@A37 = external constant [1 x i8]
define i1 @test37() nounwind {
; CHECK-LABEL: @test37(
; CHECK: ret i1 true
  %t = icmp eq i8* getelementptr ([1 x i8], [1 x i8]* @A37, i64 0, i64 1),
                   getelementptr ([1 x i8], [1 x i8]* @A37, i64 1, i64 0)
  ret i1 %t
}

; Test index promotion
define i32* @test38(i32* %I, i32 %n) {
        %A = getelementptr i32, i32* %I, i32 %n
        ret i32* %A
; CHECK-LABEL: @test38(
; CHECK: = sext i32 %n to i64
; CHECK: %A = getelementptr i32, i32* %I, i64 %
}

; Test that we don't duplicate work when the second gep is a "bitcast".
%pr10322_t = type { i8* }
declare void @pr10322_f2(%pr10322_t*)
declare void @pr10322_f3(i8**)
define void @pr10322_f1(%pr10322_t* %foo) {
entry:
  %arrayidx8 = getelementptr inbounds %pr10322_t, %pr10322_t* %foo, i64 2
  call void @pr10322_f2(%pr10322_t* %arrayidx8) nounwind
  %tmp2 = getelementptr inbounds %pr10322_t, %pr10322_t* %arrayidx8, i64 0, i32 0
  call void @pr10322_f3(i8** %tmp2) nounwind
  ret void

; CHECK-LABEL: @pr10322_f1(
; CHECK: %tmp2 = getelementptr inbounds %pr10322_t, %pr10322_t* %arrayidx8, i64 0, i32 0
}

; Test that we combine the last two geps in this sequence, before we
; would wait for gep1 and gep2 to be combined and never combine 2 and 3.
%three_gep_t = type {i32}
%three_gep_t2 = type {%three_gep_t}

define void @three_gep_f(%three_gep_t2* %x) {
  %gep1 = getelementptr %three_gep_t2, %three_gep_t2* %x, i64 2
  call void @three_gep_h(%three_gep_t2* %gep1)
  %gep2 = getelementptr %three_gep_t2, %three_gep_t2* %gep1, i64 0, i32 0
  %gep3 = getelementptr %three_gep_t, %three_gep_t* %gep2, i64 0, i32 0
  call void @three_gep_g(i32* %gep3)

; CHECK-LABEL: @three_gep_f(
; CHECK: %gep3 = getelementptr %three_gep_t2, %three_gep_t2* %gep1, i64 0, i32 0, i32 0
  ret void
}

declare void @three_gep_g(i32*)
declare void @three_gep_h(%three_gep_t2*)

%struct.ham = type { i32, %struct.zot*, %struct.zot*, %struct.zot* }
%struct.zot = type { i64, i8 }

define void @test39(%struct.ham* %arg, i8 %arg1) nounwind {
  %tmp = getelementptr inbounds %struct.ham, %struct.ham* %arg, i64 0, i32 2
  %tmp2 = load %struct.zot*, %struct.zot** %tmp, align 8
  %tmp3 = bitcast %struct.zot* %tmp2 to i8*
  %tmp4 = getelementptr inbounds i8, i8* %tmp3, i64 -8
  store i8 %arg1, i8* %tmp4, align 8
  ret void

; CHECK-LABEL: @test39(
; CHECK: getelementptr inbounds %struct.ham, %struct.ham* %arg, i64 0, i32 2
; CHECK: getelementptr inbounds i8, i8* %{{.+}}, i64 -8
}

define i1 @pr16483([1 x i8]* %a, [1 x i8]* %b) {
  %c = getelementptr [1 x i8], [1 x i8]* %a, i32 0, i32 0
  %d = getelementptr [1 x i8], [1 x i8]* %b, i32 0, i32 0
  %cmp = icmp ult i8* %c, %d
  ret i1 %cmp

; CHECK-LABEL: @pr16483(
; CHECK-NEXT: icmp ult  [1 x i8]* %a, %b
}

define i8 @test_gep_bitcast_as1(i32 addrspace(1)* %arr, i16 %N) {
; CHECK-LABEL: @test_gep_bitcast_as1(
; CHECK: getelementptr i32, i32 addrspace(1)* %arr, i16 %N
; CHECK: bitcast
  %cast = bitcast i32 addrspace(1)* %arr to i8 addrspace(1)*
  %V = mul i16 %N, 4
  %t = getelementptr i8, i8 addrspace(1)* %cast, i16 %V
  %x = load i8, i8 addrspace(1)* %t
  ret i8 %x
}

; The element size of the array matches the element size of the pointer
define i64 @test_gep_bitcast_array_same_size_element([100 x double]* %arr, i64 %N) {
; CHECK-LABEL: @test_gep_bitcast_array_same_size_element(
; CHECK: getelementptr [100 x double], [100 x double]* %arr, i64 0, i64 %V
; CHECK: bitcast
  %cast = bitcast [100 x double]* %arr to i64*
  %V = mul i64 %N, 8
  %t = getelementptr i64, i64* %cast, i64 %V
  %x = load i64, i64* %t
  ret i64 %x
}

; gep should be done in the original address space.
define i64 @test_gep_bitcast_array_same_size_element_addrspacecast([100 x double]* %arr, i64 %N) {
; CHECK-LABEL: @test_gep_bitcast_array_same_size_element_addrspacecast(
; CHECK: getelementptr [100 x double], [100 x double]* %arr, i64 0, i64 %V
; CHECK-NEXT: bitcast double*
; CHECK-NEXT: %t = addrspacecast i64*
; CHECK: load i64, i64 addrspace(3)* %t
  %cast = addrspacecast [100 x double]* %arr to i64 addrspace(3)*
  %V = mul i64 %N, 8
  %t = getelementptr i64, i64 addrspace(3)* %cast, i64 %V
  %x = load i64, i64 addrspace(3)* %t
  ret i64 %x
}

; The element size of the array is different the element size of the pointer
define i8 @test_gep_bitcast_array_different_size_element([100 x double]* %arr, i64 %N) {
; CHECK-LABEL: @test_gep_bitcast_array_different_size_element(
; CHECK: getelementptr [100 x double], [100 x double]* %arr, i64 0, i64 %N
; CHECK: bitcast
  %cast = bitcast [100 x double]* %arr to i8*
  %V = mul i64 %N, 8
  %t = getelementptr i8, i8* %cast, i64 %V
  %x = load i8, i8* %t
  ret i8 %x
}

define i64 @test_gep_bitcast_array_same_size_element_as1([100 x double] addrspace(1)* %arr, i16 %N) {
; CHECK-LABEL: @test_gep_bitcast_array_same_size_element_as1(
; CHECK: getelementptr [100 x double], [100 x double] addrspace(1)* %arr, i16 0, i16 %V
; CHECK: bitcast
  %cast = bitcast [100 x double] addrspace(1)* %arr to i64 addrspace(1)*
  %V = mul i16 %N, 8
  %t = getelementptr i64, i64 addrspace(1)* %cast, i16 %V
  %x = load i64, i64 addrspace(1)* %t
  ret i64 %x
}

define i8 @test_gep_bitcast_array_different_size_element_as1([100 x double] addrspace(1)* %arr, i16 %N) {
; CHECK-LABEL: @test_gep_bitcast_array_different_size_element_as1(
; CHECK: getelementptr [100 x double], [100 x double] addrspace(1)* %arr, i16 0, i16 %N
; CHECK: bitcast
  %cast = bitcast [100 x double] addrspace(1)* %arr to i8 addrspace(1)*
  %V = mul i16 %N, 8
  %t = getelementptr i8, i8 addrspace(1)* %cast, i16 %V
  %x = load i8, i8 addrspace(1)* %t
  ret i8 %x
}

define i64 @test40() {
  %array = alloca [3 x i32], align 4
  %gep = getelementptr inbounds [3 x i32], [3 x i32]* %array, i64 0, i64 2
  %gepi8 = bitcast i32* %gep to i8*
  %p = ptrtoint [3 x i32]* %array to i64
  %np = sub i64 0, %p
  %gep2 = getelementptr i8, i8* %gepi8, i64 %np
  %ret = ptrtoint i8* %gep2 to i64
  ret i64 %ret

; CHECK-LABEL: @test40
; CHECK-NEXT: ret i64 8
}

define i16 @test41([3 x i32] addrspace(1)* %array) {
  %gep = getelementptr inbounds [3 x i32], [3 x i32] addrspace(1)* %array, i16 0, i16 2
  %gepi8 = bitcast i32 addrspace(1)* %gep to i8 addrspace(1)*
  %p = ptrtoint [3 x i32] addrspace(1)* %array to i16
  %np = sub i16 0, %p
  %gep2 = getelementptr i8, i8 addrspace(1)* %gepi8, i16 %np
  %ret = ptrtoint i8 addrspace(1)* %gep2 to i16
  ret i16 %ret

; CHECK-LABEL: @test41(
; CHECK-NEXT: ret i16 8
}

define i8* @test42(i8* %c1, i8* %c2) {
  %ptrtoint = ptrtoint i8* %c1 to i64
  %sub = sub i64 0, %ptrtoint
  %gep = getelementptr inbounds i8, i8* %c2, i64 %sub
  ret i8* %gep

; CHECK-LABEL: @test42(
; CHECK-NEXT:  [[PTRTOINT1:%.*]] = ptrtoint i8* %c1 to i64
; CHECK-NEXT:  [[PTRTOINT2:%.*]] = ptrtoint i8* %c2 to i64
; CHECK-NEXT:  [[SUB:%.*]] = sub i64 [[PTRTOINT2]], [[PTRTOINT1]]
; CHECK-NEXT:  [[INTTOPTR:%.*]] = inttoptr i64 [[SUB]] to i8*
; CHECK-NEXT:  ret i8* [[INTTOPTR]]
}

define i16* @test43(i16* %c1, i16* %c2) {
  %ptrtoint = ptrtoint i16* %c1 to i64
  %sub = sub i64 0, %ptrtoint
  %shr = ashr i64 %sub, 1
  %gep = getelementptr inbounds i16, i16* %c2, i64 %shr
  ret i16* %gep

; CHECK-LABEL: @test43(
; CHECK-NEXT:  [[PTRTOINT1:%.*]] = ptrtoint i16* %c1 to i64
; CHECK-NEXT:  [[PTRTOINT2:%.*]] = ptrtoint i16* %c2 to i64
; CHECK-NEXT:  [[SUB:%.*]] = sub i64 [[PTRTOINT2]], [[PTRTOINT1]]
; CHECK-NEXT:  [[INTTOPTR:%.*]] = inttoptr i64 [[SUB]] to i16*
; CHECK-NEXT:  ret i16* [[INTTOPTR]]
}

define %struct.C* @test44(%struct.C* %c1, %struct.C* %c2) {
  %ptrtoint = ptrtoint %struct.C* %c1 to i64
  %sub = sub i64 0, %ptrtoint
  %shr = sdiv i64 %sub, 7
  %gep = getelementptr inbounds %struct.C, %struct.C* %c2, i64 %shr
  ret %struct.C* %gep

; CHECK-LABEL: @test44(
; CHECK-NEXT:  [[PTRTOINT1:%.*]] = ptrtoint %struct.C* %c1 to i64
; CHECK-NEXT:  [[PTRTOINT2:%.*]] = ptrtoint %struct.C* %c2 to i64
; CHECK-NEXT:  [[SUB:%.*]] = sub i64 [[PTRTOINT2]], [[PTRTOINT1]]
; CHECK-NEXT:  [[INTTOPTR:%.*]] = inttoptr i64 [[SUB]] to %struct.C*
; CHECK-NEXT:  ret %struct.C* [[INTTOPTR]]
}

define %struct.C* @test45(%struct.C* %c1, %struct.C** %c2) {
  %ptrtoint1 = ptrtoint %struct.C* %c1 to i64
  %ptrtoint2 = ptrtoint %struct.C** %c2 to i64
  %sub = sub i64 %ptrtoint2, %ptrtoint1 ; C2 - C1
  %shr = sdiv i64 %sub, 7
  %gep = getelementptr inbounds %struct.C, %struct.C* %c1, i64 %shr ; C1 + (C2 - C1)
  ret %struct.C* %gep

; CHECK-LABEL: @test45(
; CHECK-NEXT:  [[BITCAST:%.*]] = bitcast %struct.C** %c2 to %struct.C*
; CHECK-NEXT:  ret %struct.C* [[BITCAST]]
}

define %struct.C* @test46(%struct.C* %c1, %struct.C* %c2, i64 %N) {
  %ptrtoint = ptrtoint %struct.C* %c1 to i64
  %sub = sub i64 0, %ptrtoint
  %sdiv = sdiv i64 %sub, %N
  %gep = getelementptr inbounds %struct.C, %struct.C* %c2, i64 %sdiv
  ret %struct.C* %gep

; CHECK-LABEL: @test46(
; CHECK-NEXT:  [[PTRTOINT:%.*]] = ptrtoint %struct.C* %c1 to i64
; CHECK-NEXT:  [[SUB:%.*]] = sub i64 0, [[PTRTOINT]]
; CHECK-NEXT:  [[SDIV:%.*]] = sdiv i64 [[SUB]], %N
; CHECK-NEXT:  [[GEP:%.*]] = getelementptr inbounds %struct.C, %struct.C* %c2, i64 %sdiv
; CHECK-NEXT:  ret %struct.C* [[GEP]]
}

define i32* @test47(i32* %I, i64 %C, i64 %D) {
  %sub = sub i64 %D, %C
  %A = getelementptr i32, i32* %I, i64 %C
  %B = getelementptr i32, i32* %A, i64 %sub
  ret i32* %B
; CHECK-LABEL: @test47(
; CHECK-NEXT: %B = getelementptr i32, i32* %I, i64 %D
}

define i32* @test48(i32* %I, i64 %C, i64 %D) {
  %sub = sub i64 %D, %C
  %A = getelementptr i32, i32* %I, i64 %sub
  %B = getelementptr i32, i32* %A, i64 %C
  ret i32* %B
; CHECK-LABEL: @test48(
; CHECK-NEXT: %B = getelementptr i32, i32* %I, i64 %D
}

define i32* @test49(i32* %I, i64 %C) {
  %notC = xor i64 -1, %C
  %A = getelementptr i32, i32* %I, i64 %C
  %B = getelementptr i32, i32* %A, i64 %notC
  ret i32* %B
; CHECK-LABEL: @test49(
; CHECK-NEXT: %B = getelementptr i32, i32* %I, i64 -1
}

define i32 addrspace(1)* @ascast_0_gep(i32* %p) nounwind {
; CHECK-LABEL: @ascast_0_gep(
; CHECK-NOT: getelementptr
; CHECK: ret
  %gep = getelementptr i32, i32* %p, i32 0
  %x = addrspacecast i32* %gep to i32 addrspace(1)*
  ret i32 addrspace(1)* %x
}

; Do not merge the GEP and the addrspacecast, because it would undo the
; addrspacecast canonicalization.
define i32 addrspace(1)* @ascast_0_0_gep([128 x i32]* %p) nounwind {
; CHECK-LABEL: @ascast_0_0_gep(
; CHECK-NEXT: getelementptr [128 x i32]
; CHECK-NEXT: addrspacecast i32*
; CHECK-NEXT: ret i32 addrspace(1)*
  %gep = getelementptr [128 x i32], [128 x i32]* %p, i32 0, i32 0
  %x = addrspacecast i32* %gep to i32 addrspace(1)*
  ret i32 addrspace(1)* %x
}

define <2 x i32*> @PR32414(i32** %ptr) {
; CHECK-LABEL: @PR32414(
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast i32** %ptr to i32*
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i32, i32* [[TMP0]], <2 x i64> <i64 0, i64 1>
; CHECK-NEXT:    ret <2 x i32*> [[TMP1]]
;
  %tmp0 = bitcast i32** %ptr to i32*
  %tmp1 = getelementptr inbounds i32, i32* %tmp0, <2 x i64> <i64 0, i64 1>
  ret <2 x i32*> %tmp1
}

; CHECK: attributes [[NUW]] = { nounwind }
