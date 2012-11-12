; RUN: opt < %s -scalarrepl -S | FileCheck %s
; PR10987

; Make sure scalarrepl doesn't move a load across an invoke which could
; modify the loaded value.
; (The PHI could theoretically be transformed by splitting the critical
; edge, but scalarrepl doesn't modify the CFG, at least at the moment.)

declare void @extern_fn(i32*)
declare i32 @extern_fn2(i32)
declare i32 @__gcc_personality_v0(i32, i64, i8*, i8*)

define void @odd_fn(i1) noinline {
  %retptr1 = alloca i32
  %retptr2 = alloca i32
  br i1 %0, label %then, label %else

then:                                             ; preds = %2
  invoke void @extern_fn(i32* %retptr1)
          to label %join unwind label %unwind

else:                                             ; preds = %2
  store i32 3, i32* %retptr2
  br label %join

join:                                             ; preds = %then, %else
  %storemerge.in = phi i32* [ %retptr2, %else ], [ %retptr1, %then ]
  %storemerge = load i32* %storemerge.in
  %x3 = call i32 @extern_fn2(i32 %storemerge)
  ret void

unwind:                                           ; preds = %then
  %info = landingpad { i8*, i32 } personality i32 (i32, i64, i8*, i8*)* @__gcc_personality_v0
          cleanup
  call void @extern_fn(i32* null)
  unreachable
}

; CHECK: define void @odd_fn
; CHECK: %storemerge.in = phi i32* [ %retptr2, %else ], [ %retptr1, %then ]
