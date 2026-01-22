; This file contains the post-"deadargelim" IR.

define void @outer(i32 noundef %arg) {
  ; The parameter was originally `noundef %arg`, changed to `poison` by "deadargelim".
  call void @inner(i32 poison)
  call void @inner2(i32 poison)
  ret void
}

; %arg was originally `noundef`, removed by "deadargelim".
define void @inner(i32 %arg) #0 {
  ret void
}

define void @inner2(i32 %arg) #0 {
  ret void
}

attributes #0 = { noinline }
