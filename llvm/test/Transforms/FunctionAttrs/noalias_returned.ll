; RUN: opt -S -attributor -attributor-disable=false < %s | FileCheck %s

; TEST 1 - negative.

; void *G;
; void *foo(){
;   void *V = malloc(4);
;   G = V;
;   return V;
; }

@G = external global i8*

; CHECK: define i8* @foo()
define i8* @foo() {
  %1 = tail call noalias i8* @malloc(i64 4)
  store i8* %1, i8** @G, align 8
  ret i8* %1
}

declare noalias i8* @malloc(i64)

; TEST 2
; call noalias function in return instruction.

; CHECK: define noalias i8* @return_noalias()
define i8* @return_noalias(){
  %1 = tail call noalias i8* @malloc(i64 4)
  ret i8* %1
}

declare i8* @alias()

; TEST 3
; CHECK: define i8* @call_alias()
; CHECK-NOT: noalias
define i8* @call_alias(){
  %1 = tail call i8* @alias()
  ret i8* %1
}

; TEST 4
; void *baz();
; void *foo(int a);
;
; void *bar()  {
;   foo(0);
;    return baz();
; }
;
; void *foo(int a)  {
;   if (a)
;   bar();
;   return malloc(4);
; }

; CHECK: define i8* @bar()
define i8* @bar() nounwind uwtable {
  %1 = tail call i8* (...) @baz()
  ret i8* %1
}

; CHECK: define noalias i8* @foo1(i32)
define i8* @foo1(i32) nounwind uwtable {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %5, label %3

3:                                                ; preds = %1
  %4 = tail call i8* (...) @baz()
  br label %5

5:                                                ; preds = %1, %3
  %6 = tail call noalias i8* @malloc(i64 4)
  ret i8* %6
}

declare i8* @baz(...) nounwind uwtable

; TEST 5

; Returning global pointer. Should not be noalias.
; CHECK: define nonnull align 8 dereferenceable(8) i8** @getter()
define i8** @getter() {
  ret i8** @G
}

; Returning global pointer. Should not be noalias.
; CHECK: define nonnull align 8 dereferenceable(8) i8** @calle1()
define i8** @calle1(){
  %1 = call i8** @getter()
  ret i8** %1
}

; TEST 6
declare noalias i8* @strdup(i8* nocapture) nounwind

; CHECK: define noalias i8* @test6()
define i8* @test6() nounwind uwtable ssp {
  %x = alloca [2 x i8], align 1
  %arrayidx = getelementptr inbounds [2 x i8], [2 x i8]* %x, i64 0, i64 0
  store i8 97, i8* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds [2 x i8], [2 x i8]* %x, i64 0, i64 1
  store i8 0, i8* %arrayidx1, align 1
  %call = call noalias i8* @strdup(i8* %arrayidx) nounwind
  ret i8* %call
}

; TEST 7

; CHECK: define noalias i8* @test7()
define i8* @test7() nounwind {
entry:
  %A = call noalias i8* @malloc(i64 4) nounwind
  %tobool = icmp eq i8* %A, null
  br i1 %tobool, label %return, label %if.end

if.end:
  store i8 7, i8* %A
  br label %return

return:
  %retval.0 = phi i8* [ %A, %if.end ], [ null, %entry ]
  ret i8* %retval.0
}

; TEST 8

; CHECK: define noalias i8* @test8(i32*)
define i8* @test8(i32*) nounwind uwtable {
  %2 = tail call noalias i8* @malloc(i64 4)
  %3 = icmp ne i32* %0, null
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  store i8 10, i8* %2
  br label %5

5:                                                ; preds = %1, %4
  ret i8* %2
}
