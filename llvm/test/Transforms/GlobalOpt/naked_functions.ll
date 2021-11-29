; RUN: opt < %s -passes=globalopt -S | FileCheck %s

; Check that naked functions don't get marked with fast calling conventions

@g = common global i32 0, align 4

define i32 @bar() {
entry:
  %call = call i32 @foo(i32* @g)
; CHECK: %call = call i32 @foo(i32* @g)
  ret i32 %call
}

define internal i32 @foo(i32*) #0 {
entry:
  %retval = alloca i32, align 4
  call void asm sideeffect "ldr r0, [r0] \0Abx lr        \0A", ""()
  unreachable
}

; CHECK: define internal i32 @foo(i32* %0)

attributes #0 = { naked }
