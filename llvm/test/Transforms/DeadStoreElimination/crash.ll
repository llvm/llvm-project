; RUN: opt < %s -basicaa -dse -S

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

@g80 = external global i8                         ; <i8*> [#uses=3]

declare signext i8 @foo(i8 signext, i8 signext) nounwind readnone ssp

declare i32 @func68(i32) nounwind readonly ssp

; PR4815
define void @test1(i32 %int32p54) noreturn nounwind ssp {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %storemerge = phi i8 [ %2, %bb ], [ 1, %entry ] ; <i8> [#uses=1]
  store i8 %storemerge, i8* @g80
  %0 = tail call i32 @func68(i32 1) nounwind ssp  ; <i32> [#uses=1]
  %1 = trunc i32 %0 to i8                         ; <i8> [#uses=1]
  store i8 %1, i8* @g80, align 1
  store i8 undef, i8* @g80, align 1
  %2 = tail call signext i8 @foo(i8 signext undef, i8 signext 1) nounwind ; <i8> [#uses=1]
  br label %bb
}

define fastcc i32 @test2() nounwind ssp {
bb14:                                             ; preds = %bb4
  %0 = bitcast i8* undef to i8**                  ; <i8**> [#uses=1]
  %1 = getelementptr inbounds i8** %0, i64 undef  ; <i8**> [#uses=1]
  %2 = bitcast i8** %1 to i16*                    ; <i16*> [#uses=2]
  %3 = getelementptr inbounds i16* %2, i64 undef  ; <i16*> [#uses=1]
  %4 = bitcast i16* %3 to i8*                     ; <i8*> [#uses=1]
  %5 = getelementptr inbounds i8* %4, i64 undef   ; <i8*> [#uses=1]
  %6 = getelementptr inbounds i16* %2, i64 undef  ; <i16*> [#uses=1]
  store i16 undef, i16* %6, align 2
  %7 = getelementptr inbounds i8* %5, i64 undef   ; <i8*> [#uses=1]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %7, i8* undef, i64 undef, i32 1, i1 false)
  unreachable
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind


; rdar://7635088
define i32 @test3() {
entry:
  ret i32 0
  
dead:
  %P2 = getelementptr i32 *%P2, i32 52
  %Q2 = getelementptr i32 *%Q2, i32 52
  store i32 4, i32* %P2
  store i32 4, i32* %Q2
  br label %dead
}


; PR3141
%struct.ada__tags__dispatch_table = type { [1 x i32] }
%struct.f393a00_1__object = type { %struct.ada__tags__dispatch_table*, i8 }
%struct.f393a00_2__windmill = type { %struct.f393a00_1__object, i16 }

define void @test4(%struct.f393a00_2__windmill* %a, %struct.f393a00_2__windmill* %b) {
entry:
	%t = alloca %struct.f393a00_2__windmill		; <%struct.f393a00_2__windmill*> [#uses=1]
	%0 = getelementptr %struct.f393a00_2__windmill* %t, i32 0, i32 0, i32 0		; <%struct.ada__tags__dispatch_table**> [#uses=1]
	%1 = load %struct.ada__tags__dispatch_table** null, align 4		; <%struct.ada__tags__dispatch_table*> [#uses=1]
	%2 = load %struct.ada__tags__dispatch_table** %0, align 8		; <%struct.ada__tags__dispatch_table*> [#uses=1]
	store %struct.ada__tags__dispatch_table* %2, %struct.ada__tags__dispatch_table** null, align 4
	store %struct.ada__tags__dispatch_table* %1, %struct.ada__tags__dispatch_table** null, align 4
	ret void
}
