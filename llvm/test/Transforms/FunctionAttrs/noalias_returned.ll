; RUN: opt -S -passes=attributor -aa-pipeline='basic-aa' -attributor-disable=false -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=3 < %s | FileCheck %s

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

define void @nocapture(i8* %a){
  ret void
}

; CHECK: define noalias i8* @return_noalias_looks_like_capture()
define i8* @return_noalias_looks_like_capture(){
  %1 = tail call noalias i8* @malloc(i64 4)
  call void @nocapture(i8* %1)
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

; CHECK: define noalias i8* @foo1(i32 %0)
define i8* @foo1(i32 %0) nounwind uwtable {
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

; CHECK: define noalias i8* @test8(i32* %0)
define i8* @test8(i32* %0) nounwind uwtable {
  %2 = tail call noalias i8* @malloc(i64 4)
  %3 = icmp ne i32* %0, null
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  store i8 10, i8* %2
  br label %5

5:                                                ; preds = %1, %4
  ret i8* %2
}

; TEST 9
; Simple Argument Test
define internal void @test9(i8* %a, i8* %b) {
; CHECK: define internal void @test9(i8* noalias nocapture nofree readnone %a, i8* nocapture nofree readnone %b)
  ret void
}
define void @test9_helper(i8* %a, i8* %b) {
  tail call void @test9(i8* noalias %a, i8* %b)
  tail call void @test9(i8* noalias %b, i8* noalias %a)
  ret void
}


; TEST 10
; Simple CallSite Test

declare void @test10_helper_1(i8* %a)
define void @test10_helper_2(i8* noalias %a) {
; CHECK:   tail call void @test10_helper_1(i8* %a)
  tail call void @test10_helper_1(i8* %a)
  ret void
}
define void @test10(i8* noalias %a) {
; CHECK: define void @test10(i8* noalias %a)
; FIXME: missing noalias
; CHECK-NEXT:   tail call void @test10_helper_1(i8* %a)
  tail call void @test10_helper_1(i8* %a)

; CHECK-NEXT:   tail call void @test10_helper_2(i8* noalias %a)
  tail call void @test10_helper_2(i8* %a)
  ret void
}

; TEST 11
; CallSite Test

declare void @test11_helper(i8* %a, i8 *%b)
define void @test11(i8* noalias %a) {
; CHECK: define void @test11(i8* noalias %a)
; CHECK-NEXT:   tail call void @test11_helper(i8* %a, i8* %a)
  tail call void @test11_helper(i8* %a, i8* %a)
  ret void
}


; TEST 12
; CallSite Argument
declare void @use_nocapture(i8* nocapture)
declare void @use(i8*)
define void @test12_1() {
; CHECK-LABEL: @test12_1(
; CHECK-NEXT:    [[A:%.*]] = alloca i8, align 4
; CHECK-NEXT:    [[B:%.*]] = tail call noalias i8* @malloc(i64 4)
; CHECK-NEXT:    tail call void @use_nocapture(i8* noalias nonnull align 4 dereferenceable(1) [[A]])
; CHECK-NEXT:    tail call void @use_nocapture(i8* noalias nonnull align 4 dereferenceable(1) [[A]])
; CHECK-NEXT:    tail call void @use_nocapture(i8* noalias nocapture [[B]])
; CHECK-NEXT:    tail call void @use_nocapture(i8* noalias nocapture [[B]])
; CHECK-NEXT:    ret void
;
  %A = alloca i8, align 4
  %B = tail call noalias i8* @malloc(i64 4)
  tail call void @use_nocapture(i8* %A)
  tail call void @use_nocapture(i8* %A)
  tail call void @use_nocapture(i8* %B)
  tail call void @use_nocapture(i8* %B)
  ret void
}

define void @test12_2(){
; CHECK-LABEL: @test12_2(
; CHECK-NEXT:    [[A:%.*]] = tail call noalias i8* @malloc(i64 4)
; FIXME: This should be @use_nocapture(i8* noalias [[A]])
; CHECK-NEXT:    tail call void @use_nocapture(i8* nocapture [[A]])
; FIXME: This should be @use_nocapture(i8* noalias nocapture [[A]])
; CHECK-NEXT:    tail call void @use_nocapture(i8* nocapture [[A]])
; CHECK-NEXT:    tail call void @use(i8* [[A]])
; CHECK-NEXT:    tail call void @use_nocapture(i8* nocapture [[A]])
; CHECK-NEXT:    ret void
;
  %A = tail call noalias i8* @malloc(i64 4)
  tail call void @use_nocapture(i8* %A)
  tail call void @use_nocapture(i8* %A)
  tail call void @use(i8* %A)
  tail call void @use_nocapture(i8* %A)
  ret void
}

declare void @two_args(i8* nocapture , i8* nocapture)
define void @test12_3(){
; CHECK-LABEL: @test12_3(
  %A = tail call noalias i8* @malloc(i64 4)
; CHECK: tail call void @two_args(i8* nocapture %A, i8* nocapture %A)
  tail call void @two_args(i8* %A, i8* %A)
  ret void
}

define void @test12_4(){
; CHECK-LABEL: @test12_4(
  %A = tail call noalias i8* @malloc(i64 4)
  %B = tail call noalias i8* @malloc(i64 4)
  %A_0 = getelementptr i8, i8* %A, i64 0
  %A_1 = getelementptr i8, i8* %A, i64 1
  %B_0 = getelementptr i8, i8* %B, i64 0

; CHECK: tail call void @two_args(i8* noalias nocapture %A, i8* noalias nocapture %B)
  tail call void @two_args(i8* %A, i8* %B)

; CHECK: tail call void @two_args(i8* nocapture %A, i8* nocapture %A_0)
  tail call void @two_args(i8* %A, i8* %A_0)

; CHECK: tail call void @two_args(i8* nocapture %A, i8* nocapture %A_1)
  tail call void @two_args(i8* %A, i8* %A_1)

; FIXME: This should be @two_args(i8* noalias nocapture %A_0, i8* noalias nocapture %B_0)
; CHECK: tail call void @two_args(i8* nocapture %A_0, i8* nocapture %B_0)
  tail call void @two_args(i8* %A_0, i8* %B_0)
  ret void
}
