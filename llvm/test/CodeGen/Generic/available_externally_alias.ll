; RUN: llc < %s

; NVPTX does not support 'alias' yet
; XFAIL: target=nvptx{{.*}}

@v = available_externally global i32 42, align 4
@va = available_externally alias i32, ptr @v

define available_externally i32 @f() {
entry:
  ret i32 0
}

@fa = available_externally alias i32(), ptr @f
