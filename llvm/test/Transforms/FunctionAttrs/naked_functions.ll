; RUN: opt -S -function-attrs %s | FileCheck %s
; RUN: opt -S -passes='function-attrs' %s | FileCheck %s

; Don't change the attributes of parameters of naked functions, in particular
; don't mark them as readnone

@g = common global i32 0, align 4

define i32 @bar() {
entry:
  %call = call i32 @foo(ptr @g)
; CHECK: %call = call i32 @foo(ptr @g)
  ret i32 %call
}

define internal i32 @foo(ptr) #0 {
entry:
  %retval = alloca i32, align 4
  call void asm sideeffect "ldr r0, [r0] \0Abx lr        \0A", ""()
  unreachable
}

; CHECK: define internal i32 @foo(ptr %0)

attributes #0 = { naked }
