; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s --passes=inline -S | FileCheck %s

; Test that functions with attribute optnone are not inlined.
; Also test that only functions with attribute alwaysinline are
; valid candidates for inlining if the caller has the optnone attribute.

; Function Attrs: alwaysinline nounwind readnone uwtable
define i32 @alwaysInlineFunction(i32 %a) #0 {
entry:
  %mul = mul i32 %a, %a
  ret i32 %mul
}

; Function Attrs: nounwind readnone uwtable
define i32 @simpleFunction(i32 %a) #1 {
entry:
  %add = add i32 %a, %a
  ret i32 %add
}

; Function Attrs: nounwind noinline optnone readnone uwtable
define i32 @OptnoneFunction(i32 %a) #2 {
entry:
  %0 = tail call i32 @alwaysInlineFunction(i32 %a)
  %1 = tail call i32 @simpleFunction(i32 %a)
  %add = add i32 %0, %1
  ret i32 %add
}

; CHECK-LABEL: @OptnoneFunction
; CHECK-NOT: call i32 @alwaysInlineFunction(i32 %a)
; CHECK: call i32 @simpleFunction(i32 %a)
; CHECK: ret

; Function Attrs: nounwind readnone uwtable
define i32 @bar(i32 %a) #1 {
entry:
  %0 = tail call i32 @OptnoneFunction(i32 5)
  %1 = tail call i32 @simpleFunction(i32 6)
  %add = add i32 %0, %1
  ret i32 %add
}

; CHECK-LABEL: @bar
; CHECK: call i32 @OptnoneFunction(i32 5)
; CHECK-NOT: call i32 @simpleFunction(i32 6)
; CHECK: ret


attributes #0 = { alwaysinline nounwind readnone uwtable }
attributes #1 = { nounwind readnone uwtable }
attributes #2 = { nounwind noinline optnone readnone uwtable }
