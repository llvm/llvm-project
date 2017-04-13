; RUN: opt -S -O2 < %s | FileCheck %s

; Check that the inlined loads are hoisted.
; CHECK-LABEL: define i32 @fun(
; CHECK-LABEL: entry:
; CHECK: load i32, i32* @A
; CHECK: if.then:

@A = external global i32
@B = external global i32
@C = external global i32

define i32 @loadA() {
   %a = load i32, i32* @A
   ret i32 %a
}

define i32 @fun(i1 %c) {
entry:
  br i1 %c, label %if.then, label %if.else

if.then:
  store i32 1, i32* @B
  %call1 = call i32 @loadA()
  store i32 2, i32* @C
  br label %if.endif

if.else:
  store i32 2, i32* @C
  %call2 = call i32 @loadA()
  store i32 1, i32* @B
  br label %if.endif

if.endif:
  %ret = phi i32 [ %call1, %if.then ], [ %call2, %if.else ]
  ret i32 %ret
}

