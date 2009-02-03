; RUN: llvm-as < %s | opt -sretpromotion | llvm-dis > %t
; RUN: cat %t | grep sret | count 1

; This function is promotable
define internal void @promotable({i32, i32}* sret %s) {
  %A = getelementptr {i32, i32}* %s, i32 0, i32 0
  store i32 0, i32* %A
  %B = getelementptr {i32, i32}* %s, i32 0, i32 0
  store i32 1, i32* %B
  ret void
}

; This function is not promotable (due to it's use below)
define internal void @notpromotable({i32, i32}* sret %s) {
  %A = getelementptr {i32, i32}* %s, i32 0, i32 0
  store i32 0, i32* %A
  %B = getelementptr {i32, i32}* %s, i32 0, i32 0
  store i32 1, i32* %B
  ret void
}

define void @caller({i32, i32}* %t) {
  %s = alloca {i32, i32}
  call void @promotable({i32, i32}* %s)
  %A = getelementptr {i32, i32}* %s, i32 0, i32 0
  %a = load i32* %A
  %B = getelementptr {i32, i32}* %s, i32 0, i32 0
  %b = load i32* %B
  ; This passes in something that's not an alloca, which makes the argument not
  ; promotable
  call void @notpromotable({i32, i32}* %t)
  ret void
}
