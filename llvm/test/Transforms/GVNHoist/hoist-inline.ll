; RUN: opt -S -O2 -enable-gvn-hoist < %s | FileCheck %s

; Check that the inlined loads are hoisted.
; CHECK-LABEL: define i32 @fun(
; CHECK-LABEL: entry:
; CHECK: load i32, ptr @A
; CHECK: if.then:

@A = external global i32
@B = external global i32
@C = external global i32
@D = external global i32
@E = external global i32

define i32 @loadA() {
   %a = load i32, ptr @A
   ret i32 %a
}

define i32 @fun(i1 %c) {
entry:
  br i1 %c, label %if.then, label %if.else

if.then:
  store i32 1, ptr @B
  %call1 = call i32 @loadA()
  store i32 2, ptr @C
  br label %if.endif

if.else:
  store i32 2, ptr @D
  %call2 = call i32 @loadA()
  store i32 1, ptr @E
  br label %if.endif

if.endif:
  %ret = phi i32 [ %call1, %if.then ], [ %call2, %if.else ]
  ret i32 %ret
}

