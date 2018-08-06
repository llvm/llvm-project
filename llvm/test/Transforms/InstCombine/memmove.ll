; This test makes sure that memmove instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

@S = internal constant [33 x i8] c"panic: restorelist inconsistency\00"		; <[33 x i8]*> [#uses=1]
@h = constant [2 x i8] c"h\00"		; <[2 x i8]*> [#uses=1]
@hel = constant [4 x i8] c"hel\00"		; <[4 x i8]*> [#uses=1]
@hello_u = constant [8 x i8] c"hello_u\00"		; <[8 x i8]*> [#uses=1]

define void @test1(i8* %A, i8* %B, i32 %N) {
  ;; CHECK-LABEL: test1
  ;; CHECK-NEXT: ret void
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %A, i8* %B, i32 0, i1 false)
  ret void
}

define void @test2(i8* %A, i32 %N) {
  ;; dest can't alias source since we can't write to source!
  ;; CHECK-LABEL: test2
  ;; CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %A, i8* align 16 getelementptr inbounds ([33 x i8], [33 x i8]* @S, i{{32|64}} 0, i{{32|64}} 0), i32 %N, i1 false)
  ;; CHECK-NEXT: ret void
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %A, i8* getelementptr inbounds ([33 x i8], [33 x i8]* @S, i32 0, i32 0), i32 %N, i1 false)
  ret void
}

define i32 @test3([1024 x i8]* %target) { ; arg: [1024 x i8]*> [#uses=1]
  ;; CHECK-LABEL: test3
  ;; CHECK-NEXT: [[P1:%[^\s]+]] = bitcast [1024 x i8]* %target to i16*
  ;; CHECK-NEXT: store i16 104, i16* [[P1]], align 2
  ;; CHECK-NEXT: [[P2:%[^\s]+]] = bitcast [1024 x i8]* %target to i32*
  ;; CHECK-NEXT: store i32 7103848, i32* [[P2]], align 4
  ;; CHECK-NEXT: [[P3:%[^\s]+]] = bitcast [1024 x i8]* %target to i64*
  ;; CHECK-NEXT: store i64 33037504440198504, i64* [[P3]], align 8
  ;; CHECK-NEXT: ret i32 0
  %h_p = getelementptr [2 x i8], [2 x i8]* @h, i32 0, i32 0		; <i8*> [#uses=1]
  %hel_p = getelementptr [4 x i8], [4 x i8]* @hel, i32 0, i32 0		; <i8*> [#uses=1]
  %hello_u_p = getelementptr [8 x i8], [8 x i8]* @hello_u, i32 0, i32 0		; <i8*> [#uses=1]
  %target_p = getelementptr [1024 x i8], [1024 x i8]* %target, i32 0, i32 0		; <i8*> [#uses=3]
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 2 %target_p, i8* align 2 %h_p, i32 2, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 4 %target_p, i8* align 4 %hel_p, i32 4, i1 false)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 8 %target_p, i8* align 8 %hello_u_p, i32 8, i1 false)
  ret i32 0
}

; PR2370
define void @test4(i8* %a) {
  ;; CHECK-LABEL: test4
  ;; CHECK-NEXT: ret void
  tail call void @llvm.memmove.p0i8.p0i8.i32(i8* %a, i8* %a, i32 100, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) argmemonly nounwind
