; RUN: llc < %s
; PR4975

; NVPTX does not support zero sized type arg
; UNSUPPORTED: target=nvptx{{.*}}

%0 = type <{ [0 x i32] }>
%union.T0 = type { }

@.str = private constant [1 x i8] c" "

define void @t(%0) nounwind {
entry:
  %arg0 = alloca %union.T0
  store %0 %0, ptr %arg0, align 1
  ret void
}

declare i32 @printf(ptr, ...)
