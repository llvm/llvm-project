; RUN: opt < %s -inline -argpromotion -disable-output
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.6"
	%struct.quad_struct = type { i32, i32, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct*, %struct.quad_struct* }
@NumNodes = external global i32		; <i32*> [#uses=0]
@"\01LC" = external constant [43 x i8]		; <[43 x i8]*> [#uses=0]
@"\01LC1" = external constant [19 x i8]		; <[19 x i8]*> [#uses=0]
@"\01LC2" = external constant [17 x i8]		; <[17 x i8]*> [#uses=0]

declare i32 @dealwithargs(i32, i8** nocapture) nounwind

declare i32 @atoi(i8*)

define internal fastcc i32 @adj(i32 %d, i32 %ct) nounwind readnone {
entry:
	switch i32 %d, label %return [
		i32 0, label %bb
		i32 1, label %bb10
		i32 2, label %bb5
		i32 3, label %bb15
	]

bb:		; preds = %entry
	switch i32 %ct, label %bb3 [
		i32 1, label %return
		i32 0, label %return
	]

bb3:		; preds = %bb
	ret i32 0

bb5:		; preds = %entry
	switch i32 %ct, label %bb8 [
		i32 3, label %return
		i32 2, label %return
	]

bb8:		; preds = %bb5
	ret i32 0

bb10:		; preds = %entry
	switch i32 %ct, label %bb13 [
		i32 1, label %return
		i32 3, label %return
	]

bb13:		; preds = %bb10
	ret i32 0

bb15:		; preds = %entry
	switch i32 %ct, label %bb18 [
		i32 2, label %return
		i32 0, label %return
	]

bb18:		; preds = %bb15
	ret i32 0

return:		; preds = %bb15, %bb15, %bb10, %bb10, %bb5, %bb5, %bb, %bb, %entry
	ret i32 1
}

declare fastcc i32 @reflect(i32, i32) nounwind readnone

declare i32 @CountTree(%struct.quad_struct* nocapture) nounwind readonly

define internal fastcc %struct.quad_struct* @child(%struct.quad_struct* nocapture %tree, i32 %ct) nounwind readonly {
entry:
	switch i32 %ct, label %bb5 [
		i32 0, label %bb1
		i32 1, label %bb
		i32 2, label %bb3
		i32 3, label %bb2
	]

bb:		; preds = %entry
	%0 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 3		; <%struct.quad_struct**> [#uses=1]
	%1 = load %struct.quad_struct*, %struct.quad_struct** %0, align 4		; <%struct.quad_struct*> [#uses=1]
	ret %struct.quad_struct* %1

bb1:		; preds = %entry
	%2 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 2		; <%struct.quad_struct**> [#uses=1]
	%3 = load %struct.quad_struct*, %struct.quad_struct** %2, align 4		; <%struct.quad_struct*> [#uses=1]
	ret %struct.quad_struct* %3

bb2:		; preds = %entry
	%4 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 5		; <%struct.quad_struct**> [#uses=1]
	%5 = load %struct.quad_struct*, %struct.quad_struct** %4, align 4		; <%struct.quad_struct*> [#uses=1]
	ret %struct.quad_struct* %5

bb3:		; preds = %entry
	%6 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 4		; <%struct.quad_struct**> [#uses=1]
	%7 = load %struct.quad_struct*, %struct.quad_struct** %6, align 4		; <%struct.quad_struct*> [#uses=1]
	ret %struct.quad_struct* %7

bb5:		; preds = %entry
	ret %struct.quad_struct* null
}

define internal fastcc %struct.quad_struct* @gtequal_adj_neighbor(%struct.quad_struct* nocapture %tree, i32 %d) nounwind readonly {
entry:
	%0 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 6		; <%struct.quad_struct**> [#uses=1]
	%1 = load %struct.quad_struct*, %struct.quad_struct** %0, align 4		; <%struct.quad_struct*> [#uses=4]
	%2 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 1		; <i32*> [#uses=1]
	%3 = load i32, i32* %2, align 4		; <i32> [#uses=2]
	%4 = icmp eq %struct.quad_struct* %1, null		; <i1> [#uses=1]
	br i1 %4, label %bb3, label %bb

bb:		; preds = %entry
	%5 = call fastcc i32 @adj(i32 %d, i32 %3) nounwind		; <i32> [#uses=1]
	%6 = icmp eq i32 %5, 0		; <i1> [#uses=1]
	br i1 %6, label %bb3, label %bb1

bb1:		; preds = %bb
	%7 = call fastcc %struct.quad_struct* @gtequal_adj_neighbor(%struct.quad_struct* %1, i32 %d) nounwind		; <%struct.quad_struct*> [#uses=1]
	br label %bb3

bb3:		; preds = %bb1, %bb, %entry
	%q.0 = phi %struct.quad_struct* [ %7, %bb1 ], [ %1, %bb ], [ %1, %entry ]		; <%struct.quad_struct*> [#uses=4]
	%8 = icmp eq %struct.quad_struct* %q.0, null		; <i1> [#uses=1]
	br i1 %8, label %bb7, label %bb4

bb4:		; preds = %bb3
	%9 = getelementptr %struct.quad_struct, %struct.quad_struct* %q.0, i32 0, i32 0		; <i32*> [#uses=1]
	%10 = load i32, i32* %9, align 4		; <i32> [#uses=1]
	%11 = icmp eq i32 %10, 2		; <i1> [#uses=1]
	br i1 %11, label %bb5, label %bb7

bb5:		; preds = %bb4
	%12 = call fastcc i32 @reflect(i32 %d, i32 %3) nounwind		; <i32> [#uses=1]
	%13 = call fastcc %struct.quad_struct* @child(%struct.quad_struct* %q.0, i32 %12) nounwind		; <%struct.quad_struct*> [#uses=1]
	ret %struct.quad_struct* %13

bb7:		; preds = %bb4, %bb3
	ret %struct.quad_struct* %q.0
}

declare fastcc i32 @sum_adjacent(%struct.quad_struct* nocapture, i32, i32, i32) nounwind readonly

define i32 @perimeter(%struct.quad_struct* nocapture %tree, i32 %size) nounwind readonly {
entry:
	%0 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 0		; <i32*> [#uses=1]
	%1 = load i32, i32* %0, align 4		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 2		; <i1> [#uses=1]
	br i1 %2, label %bb, label %bb2

bb:		; preds = %entry
	%3 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 4		; <%struct.quad_struct**> [#uses=1]
	%4 = load %struct.quad_struct*, %struct.quad_struct** %3, align 4		; <%struct.quad_struct*> [#uses=1]
	%5 = sdiv i32 %size, 2		; <i32> [#uses=1]
	%6 = call i32 @perimeter(%struct.quad_struct* %4, i32 %5) nounwind		; <i32> [#uses=1]
	%7 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 5		; <%struct.quad_struct**> [#uses=1]
	%8 = load %struct.quad_struct*, %struct.quad_struct** %7, align 4		; <%struct.quad_struct*> [#uses=1]
	%9 = sdiv i32 %size, 2		; <i32> [#uses=1]
	%10 = call i32 @perimeter(%struct.quad_struct* %8, i32 %9) nounwind		; <i32> [#uses=1]
	%11 = add i32 %10, %6		; <i32> [#uses=1]
	%12 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 3		; <%struct.quad_struct**> [#uses=1]
	%13 = load %struct.quad_struct*, %struct.quad_struct** %12, align 4		; <%struct.quad_struct*> [#uses=1]
	%14 = sdiv i32 %size, 2		; <i32> [#uses=1]
	%15 = call i32 @perimeter(%struct.quad_struct* %13, i32 %14) nounwind		; <i32> [#uses=1]
	%16 = add i32 %15, %11		; <i32> [#uses=1]
	%17 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 2		; <%struct.quad_struct**> [#uses=1]
	%18 = load %struct.quad_struct*, %struct.quad_struct** %17, align 4		; <%struct.quad_struct*> [#uses=1]
	%19 = sdiv i32 %size, 2		; <i32> [#uses=1]
	%20 = call i32 @perimeter(%struct.quad_struct* %18, i32 %19) nounwind		; <i32> [#uses=1]
	%21 = add i32 %20, %16		; <i32> [#uses=1]
	ret i32 %21

bb2:		; preds = %entry
	%22 = getelementptr %struct.quad_struct, %struct.quad_struct* %tree, i32 0, i32 0		; <i32*> [#uses=1]
	%23 = load i32, i32* %22, align 4		; <i32> [#uses=1]
	%24 = icmp eq i32 %23, 0		; <i1> [#uses=1]
	br i1 %24, label %bb3, label %bb23

bb3:		; preds = %bb2
	%25 = call fastcc %struct.quad_struct* @gtequal_adj_neighbor(%struct.quad_struct* %tree, i32 0) nounwind		; <%struct.quad_struct*> [#uses=4]
	%26 = icmp eq %struct.quad_struct* %25, null		; <i1> [#uses=1]
	br i1 %26, label %bb8, label %bb4

bb4:		; preds = %bb3
	%27 = getelementptr %struct.quad_struct, %struct.quad_struct* %25, i32 0, i32 0		; <i32*> [#uses=1]
	%28 = load i32, i32* %27, align 4		; <i32> [#uses=1]
	%29 = icmp eq i32 %28, 1		; <i1> [#uses=1]
	br i1 %29, label %bb8, label %bb6

bb6:		; preds = %bb4
	%30 = getelementptr %struct.quad_struct, %struct.quad_struct* %25, i32 0, i32 0		; <i32*> [#uses=1]
	%31 = load i32, i32* %30, align 4		; <i32> [#uses=1]
	%32 = icmp eq i32 %31, 2		; <i1> [#uses=1]
	br i1 %32, label %bb7, label %bb8

bb7:		; preds = %bb6
	%33 = call fastcc i32 @sum_adjacent(%struct.quad_struct* %25, i32 3, i32 2, i32 %size) nounwind		; <i32> [#uses=1]
	br label %bb8

bb8:		; preds = %bb7, %bb6, %bb4, %bb3
	%retval1.1 = phi i32 [ 0, %bb6 ], [ %33, %bb7 ], [ %size, %bb4 ], [ %size, %bb3 ]		; <i32> [#uses=3]
	%34 = call fastcc %struct.quad_struct* @gtequal_adj_neighbor(%struct.quad_struct* %tree, i32 1) nounwind		; <%struct.quad_struct*> [#uses=4]
	%35 = icmp eq %struct.quad_struct* %34, null		; <i1> [#uses=1]
	br i1 %35, label %bb10, label %bb9

bb9:		; preds = %bb8
	%36 = getelementptr %struct.quad_struct, %struct.quad_struct* %34, i32 0, i32 0		; <i32*> [#uses=1]
	%37 = load i32, i32* %36, align 4		; <i32> [#uses=1]
	%38 = icmp eq i32 %37, 1		; <i1> [#uses=1]
	br i1 %38, label %bb10, label %bb11

bb10:		; preds = %bb9, %bb8
	%39 = add i32 %retval1.1, %size		; <i32> [#uses=1]
	br label %bb13

bb11:		; preds = %bb9
	%40 = getelementptr %struct.quad_struct, %struct.quad_struct* %34, i32 0, i32 0		; <i32*> [#uses=1]
	%41 = load i32, i32* %40, align 4		; <i32> [#uses=1]
	%42 = icmp eq i32 %41, 2		; <i1> [#uses=1]
	br i1 %42, label %bb12, label %bb13

bb12:		; preds = %bb11
	%43 = call fastcc i32 @sum_adjacent(%struct.quad_struct* %34, i32 2, i32 0, i32 %size) nounwind		; <i32> [#uses=1]
	%44 = add i32 %43, %retval1.1		; <i32> [#uses=1]
	br label %bb13

bb13:		; preds = %bb12, %bb11, %bb10
	%retval1.2 = phi i32 [ %retval1.1, %bb11 ], [ %44, %bb12 ], [ %39, %bb10 ]		; <i32> [#uses=3]
	%45 = call fastcc %struct.quad_struct* @gtequal_adj_neighbor(%struct.quad_struct* %tree, i32 2) nounwind		; <%struct.quad_struct*> [#uses=4]
	%46 = icmp eq %struct.quad_struct* %45, null		; <i1> [#uses=1]
	br i1 %46, label %bb15, label %bb14

bb14:		; preds = %bb13
	%47 = getelementptr %struct.quad_struct, %struct.quad_struct* %45, i32 0, i32 0		; <i32*> [#uses=1]
	%48 = load i32, i32* %47, align 4		; <i32> [#uses=1]
	%49 = icmp eq i32 %48, 1		; <i1> [#uses=1]
	br i1 %49, label %bb15, label %bb16

bb15:		; preds = %bb14, %bb13
	%50 = add i32 %retval1.2, %size		; <i32> [#uses=1]
	br label %bb18

bb16:		; preds = %bb14
	%51 = getelementptr %struct.quad_struct, %struct.quad_struct* %45, i32 0, i32 0		; <i32*> [#uses=1]
	%52 = load i32, i32* %51, align 4		; <i32> [#uses=1]
	%53 = icmp eq i32 %52, 2		; <i1> [#uses=1]
	br i1 %53, label %bb17, label %bb18

bb17:		; preds = %bb16
	%54 = call fastcc i32 @sum_adjacent(%struct.quad_struct* %45, i32 0, i32 1, i32 %size) nounwind		; <i32> [#uses=1]
	%55 = add i32 %54, %retval1.2		; <i32> [#uses=1]
	br label %bb18

bb18:		; preds = %bb17, %bb16, %bb15
	%retval1.3 = phi i32 [ %retval1.2, %bb16 ], [ %55, %bb17 ], [ %50, %bb15 ]		; <i32> [#uses=3]
	%56 = call fastcc %struct.quad_struct* @gtequal_adj_neighbor(%struct.quad_struct* %tree, i32 3) nounwind		; <%struct.quad_struct*> [#uses=4]
	%57 = icmp eq %struct.quad_struct* %56, null		; <i1> [#uses=1]
	br i1 %57, label %bb20, label %bb19

bb19:		; preds = %bb18
	%58 = getelementptr %struct.quad_struct, %struct.quad_struct* %56, i32 0, i32 0		; <i32*> [#uses=1]
	%59 = load i32, i32* %58, align 4		; <i32> [#uses=1]
	%60 = icmp eq i32 %59, 1		; <i1> [#uses=1]
	br i1 %60, label %bb20, label %bb21

bb20:		; preds = %bb19, %bb18
	%61 = add i32 %retval1.3, %size		; <i32> [#uses=1]
	ret i32 %61

bb21:		; preds = %bb19
	%62 = getelementptr %struct.quad_struct, %struct.quad_struct* %56, i32 0, i32 0		; <i32*> [#uses=1]
	%63 = load i32, i32* %62, align 4		; <i32> [#uses=1]
	%64 = icmp eq i32 %63, 2		; <i1> [#uses=1]
	br i1 %64, label %bb22, label %bb23

bb22:		; preds = %bb21
	%65 = call fastcc i32 @sum_adjacent(%struct.quad_struct* %56, i32 1, i32 3, i32 %size) nounwind		; <i32> [#uses=1]
	%66 = add i32 %65, %retval1.3		; <i32> [#uses=1]
	ret i32 %66

bb23:		; preds = %bb21, %bb2
	%retval1.0 = phi i32 [ 0, %bb2 ], [ %retval1.3, %bb21 ]		; <i32> [#uses=1]
	ret i32 %retval1.0
}

declare i32 @main(i32, i8** nocapture) noreturn nounwind

declare i32 @printf(i8*, ...) nounwind

declare void @exit(i32) noreturn nounwind

declare fastcc i32 @CheckOutside(i32, i32) nounwind readnone

declare fastcc i32 @CheckIntersect(i32, i32, i32) nounwind readnone

declare %struct.quad_struct* @MakeTree(i32, i32, i32, i32, i32, %struct.quad_struct*, i32, i32) nounwind
