; RUN: echo 'hi' > %t.1 | echo 'hello' > %t.2
; RUN: not grep 'hi' %t.1
; RUN: grep 'hello' %t.2




