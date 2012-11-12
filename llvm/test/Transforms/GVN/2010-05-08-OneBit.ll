; RUN: opt < %s -gvn
; PR7052

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main(i32 %argc, i8** nocapture %argv) {
entry:
  %0 = getelementptr inbounds i8* undef, i64 5    ; <i8*> [#uses=1]
  %1 = bitcast i8* %0 to i32*                     ; <i32*> [#uses=1]
  store i32 undef, i32* %1, align 1
  br i1 undef, label %k121.i.i, label %l117.i.i

l117.i.i:                                         ; preds = %entry
  invoke fastcc void @foo()
          to label %.noexc5 unwind label %landing_pad

.noexc5:                                          ; preds = %l117.i.i
  unreachable

k121.i.i:                                         ; preds = %entry
  br i1 undef, label %l129.i.i, label %k133.i.i

l129.i.i:                                         ; preds = %k121.i.i
  invoke fastcc void @foo()
          to label %.noexc7 unwind label %landing_pad

.noexc7:                                          ; preds = %l129.i.i
  unreachable

k133.i.i:                                         ; preds = %k121.i.i
  %2 = getelementptr i8* undef, i64 5             ; <i8*> [#uses=1]
  %3 = bitcast i8* %2 to i1*                      ; <i1*> [#uses=1]
  %4 = load i1* %3                                ; <i1> [#uses=1]
  br i1 %4, label %k151.i.i, label %l147.i.i

l147.i.i:                                         ; preds = %k133.i.i
  invoke fastcc void @foo()
          to label %.noexc10 unwind label %landing_pad

.noexc10:                                         ; preds = %l147.i.i
  unreachable

k151.i.i:                                         ; preds = %k133.i.i
  ret i32 0

landing_pad:                                      ; preds = %l147.i.i, %l129.i.i, %l117.i.i
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            cleanup
  switch i32 undef, label %fin [
    i32 1, label %catch1
    i32 2, label %catch
  ]

fin:                                              ; preds = %landing_pad
  unreachable

catch:                                            ; preds = %landing_pad
  ret i32 1

catch1:                                           ; preds = %landing_pad
  ret i32 2
}

declare fastcc void @foo()

declare i32 @__gxx_personality_v0(...)
