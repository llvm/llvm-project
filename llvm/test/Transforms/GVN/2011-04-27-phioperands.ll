; RUN: opt -passes=gvn -disable-output < %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-f128:128:128-n8:16:32:64"

@nuls = external global [10 x i8]

define fastcc void @p_ere() nounwind {
entry:
  br label %"<bb 5>"

"<L18>.i":
  br i1 undef, label %"<bb 3>.i30.i", label %doemit.exit51.i

"<bb 3>.i30.i":
  unreachable

doemit.exit51.i:
  br label %"<bb 53>.i"

"<L19>.i":
  br i1 undef, label %"<bb 3>.i55.i", label %doemit.exit76.i

"<bb 3>.i55.i":
  unreachable

doemit.exit76.i:
  br label %"<bb 53>.i"

"<L98>.i":
  store ptr @nuls, ptr undef, align 8
  br label %"<bb 53>.i"

"<L99>.i":
  br label %"<bb 53>.i"

"<L24>.i":
  br i1 undef, label %"<bb 53>.i", label %"<bb 35>.i"

"<bb 35>.i":
  br label %"<bb 53>.i"

"<L28>.i":
  br label %"<bb 53>.i"

"<L29>.i":
  br label %"<bb 53>.i"

"<L39>.i":
  br label %"<bb 53>.i"

"<bb 53>.i":
  %wascaret_2.i = phi i32 [ 0, %"<L39>.i" ], [ 0, %"<L29>.i" ], [ 0, %"<L28>.i" ], [ 0, %"<bb 35>.i" ], [ 0, %"<L99>.i" ], [ 0, %"<L98>.i" ], [ 0, %doemit.exit76.i ], [ 1, %doemit.exit51.i ], [ 0, %"<L24>.i" ]
  %D.5496_84.i = load ptr, ptr undef, align 8
  br i1 undef, label %"<bb 54>.i", label %"<bb 5>"

"<bb 54>.i":
  br i1 undef, label %"<bb 5>", label %"<bb 58>.i"

"<bb 58>.i":
  br i1 undef, label %"<bb 64>.i", label %"<bb 59>.i"

"<bb 59>.i":
  br label %"<bb 64>.i"

"<bb 64>.i":
  switch i32 undef, label %"<bb 5>" [
    i32 42, label %"<L54>.i"
    i32 43, label %"<L55>.i"
    i32 63, label %"<L56>.i"
    i32 123, label %"<bb 5>.i258.i"
  ]

"<L54>.i":
  br i1 undef, label %"<bb 3>.i105.i", label %doemit.exit127.i

"<bb 3>.i105.i":
  unreachable

doemit.exit127.i:
  unreachable

"<L55>.i":
  br i1 undef, label %"<bb 3>.i157.i", label %"<bb 5>"

"<bb 3>.i157.i":
  unreachable

"<L56>.i":
  br label %"<bb 5>"

"<bb 5>.i258.i":
  unreachable

"<bb 5>":
  switch i32 undef, label %"<L39>.i" [
    i32 36, label %"<L19>.i"
    i32 94, label %"<L18>.i"
    i32 124, label %"<L98>.i"
    i32 42, label %"<L99>.i"
    i32 43, label %"<L99>.i"
    i32 46, label %"<L24>.i"
    i32 63, label %"<L99>.i"
    i32 91, label %"<L28>.i"
    i32 92, label %"<L29>.i"
  ]
}
