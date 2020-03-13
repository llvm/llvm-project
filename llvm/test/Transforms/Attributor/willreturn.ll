; RUN: opt -passes=attributor --attributor-disable=false -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=6 -S < %s | FileCheck %s --check-prefixes=ATTRIBUTOR,ATTRIBUTOR_MODULE
; RUN: opt -passes=attributor-cgscc --attributor-disable=false -attributor-annotate-decl-cs -S < %s | FileCheck %s --check-prefixes=ATTRIBUTOR,ATTRIBUTOR_CGSCC


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Test cases specifically designed for the "willreturn" function attribute.
; We use FIXME's to indicate problems and missing attributes.


; TEST 1 (positive case)
; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: nofree noinline norecurse nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR-NEXT: define void @only_return()
define void @only_return() #0 {
    ret void
}


; TEST 2 (positive & negative case)
; 2.1 (positive case)
; recursive function which will halt
; int fib(int n){
;    return n<=1? n : fib(n-1) + fib(n-2);
; }

; FIXME: missing willreturn
; ATTRIBUTOR: Function Attrs: nofree noinline nosync nounwind readnone uwtable
; ATTRIBUTOR-NEXT: define i32 @fib(i32 %0) local_unnamed_addr
define i32 @fib(i32 %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 2
  br i1 %2, label %9, label %3

; <label>:3:                                      ; preds = %1
  %4 = add nsw i32 %0, -1
  %5 = tail call i32 @fib(i32 %4)
  %6 = add nsw i32 %0, -2
  %7 = tail call i32 @fib(i32 %6)
  %8 = add nsw i32 %7, %5
  ret i32 %8

; <label>:9:                                      ; preds = %1
  ret i32 %0
}

; 2.2 (negative case)
; recursive function which doesn't stop for some input.
; int fact_maybe_not_halt(int n) {
;   if (n==0) {
;     return 1;
;   }
;   return fact_maybe_not_halt( n > 0 ? n-1 : n) * n;
; }
; fact_maybe_not(-1) doesn't stop.

; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline nosync nounwind readnone uwtable
; ATTRIBUTOR_CGSCC: Function Attrs: nofree noinline norecurse nosync nounwind readnone uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define i32 @fact_maybe_not_halt(i32 %0) local_unnamed_addr
define i32 @fact_maybe_not_halt(i32 %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 0
  br i1 %2, label %11, label %3

; <label>:3:                                      ; preds = %1, %3
  %4 = phi i32 [ %8, %3 ], [ %0, %1 ]
  %5 = phi i32 [ %9, %3 ], [ 1, %1 ]
  %6 = icmp sgt i32 %4, 0
  %7 = sext i1 %6 to i32
  %8 = add nsw i32 %4, %7
  %9 = mul nsw i32 %4, %5
  %10 = icmp eq i32 %8, 0
  br i1 %10, label %11, label %3

; <label>:11:                                     ; preds = %3, %1
  %12 = phi i32 [ 1, %1 ], [ %9, %3 ]
  ret i32 %12
}


; TEST 3 (positive case)
; loop
; int fact_loop(int n ){
;   int ans = 1;
;   for(int i = 1;i<=n;i++){
;     ans *= i;
;   }
;   return ans;
; }

; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: nofree noinline norecurse nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR-NEXT: define i32 @fact_loop(i32 %0) local_unnamed_addr
define i32 @fact_loop(i32 %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %5, %1
  %4 = phi i32 [ 1, %1 ], [ %8, %5 ]
  ret i32 %4

; <label>:5:                                      ; preds = %1, %5
  %6 = phi i32 [ %9, %5 ], [ 1, %1 ]
  %7 = phi i32 [ %8, %5 ], [ 1, %1 ]
  %8 = mul nsw i32 %6, %7
  %9 = add nuw nsw i32 %6, 1
  %10 = icmp eq i32 %6, %0
  br i1 %10, label %3, label %5
}

; TEST 4 (negative case)
; mutual recursion
; void mutual_recursion1(){
;    mutual_recursion2();
; }
; void mutual_recursion2(){
;     mutual_recursion1();
; }

declare void @sink() nounwind willreturn nosync nofree

; ATTRIBUTOR: Function Attrs: nofree noinline nosync nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @mutual_recursion1(i1 %c)
define void @mutual_recursion1(i1 %c) #0 {
  br i1 %c, label %rec, label %end
rec:
  call void @sink()
  call void @mutual_recursion2(i1 %c)
  br label %end
end:
  ret void
}


; ATTRIBUTOR: Function Attrs: nofree noinline nosync nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @mutual_recursion2(i1 %c)
define void @mutual_recursion2(i1 %c) #0 {
  call void @mutual_recursion1(i1 %c)
  ret void
}


; TEST 5 (negative case)
; call exit/abort (has noreturn attribute)
; ATTRIBUTOR: Function Attrs: noreturn
; ATTRIBUTOR-NEXT: declare void @exit(i32) local_unnamed_add
declare void @exit(i32 %0) local_unnamed_addr noreturn

; ATTRIBUTOR: Function Attrs: noinline noreturn nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @only_exit() local_unnamed_addr
define void @only_exit() local_unnamed_addr #0 {
  tail call void @exit(i32 0)
  unreachable
}

; conditional exit
; void conditional_exit(int cond, int *p){
;     if(cond){
;       exit(0);
;     }
;     if(*p){
;       exit(1);
;     }
;     return;
; }
; ATTRIBUTOR: Function Attrs: noinline nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @conditional_exit(i32 %0, i32* nocapture readonly %1) local_unnamed_addr
define void @conditional_exit(i32 %0, i32* nocapture readonly %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %5, label %4

; <label>:4:                                      ; preds = %2
  tail call void @exit(i32 0)
  unreachable

; <label>:5:                                      ; preds = %2
  %6 = load i32, i32* %1, align 4
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %9, label %8

; <label>:8:                                      ; preds = %5
  tail call void @exit(i32 1)
  unreachable

; <label>:9:                                      ; preds = %5
  ret void
}

; TEST 6 (positive case)
; Call intrinsic function
; ATTRIBUTOR: Function Attrs: nounwind readnone speculatable willreturn
; ATTRIBUTOR-NEXT: declare float @llvm.floor.f32(float)
declare float @llvm.floor.f32(float)

; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR_CGSCC:  Function Attrs: nofree noinline norecurse nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR-NEXT:   define void @call_floor(float %a)
define void @call_floor(float %a) #0 {
    tail call float @llvm.floor.f32(float %a)
    ret void
}

; ATTRIBUTOR: Function Attrs: noinline nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR-NEXT: define float @call_floor2(float %a)
define float @call_floor2(float %a) #0 {
    %c = tail call float @llvm.floor.f32(float %a)
    ret float %c
}


; TEST 7 (negative case)
; Call function declaration without willreturn

; ATTRIBUTOR: Function Attrs: noinline nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: declare void @maybe_noreturn()
declare void @maybe_noreturn() #0

; ATTRIBUTOR: Function Attrs: noinline nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @call_maybe_noreturn()
define void @call_maybe_noreturn() #0 {
    tail call void @maybe_noreturn()
    ret void
}


; TEST 8 (positive case)
; Check propagation.

; ATTRIBUTOR: Function Attrs: norecurse willreturn
; ATTRIBUTOR-NEXT: declare void @will_return()
declare void @will_return() willreturn norecurse

; ATTRIBUTOR_MODULE: Function Attrs: noinline nounwind uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: noinline norecurse nounwind uwtable willreturn
; ATTRIBUTOR-NEXT: define void @f1()
define void @f1() #0 {
    tail call void @will_return()
    ret void
}

; ATTRIBUTOR_MODULE: Function Attrs: noinline nounwind uwtable
; FIXME: Because we do not derive norecurse in the module run anymore, willreturn is missing as well.
; ATTRIBUTOR_MODULE-NOT: willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: noinline norecurse nounwind uwtable willreturn
; ATTRIBUTOR-NEXT: define void @f2()
define void @f2() #0 {
    tail call void @f1()
    ret void
}


; TEST 9 (negative case)
; call willreturn function in endless loop.

; ATTRIBUTOR_MODULE: Function Attrs: noinline noreturn nounwind uwtable
; ATTRIBUTOR_CGSCC: Function Attrs: noinline norecurse noreturn nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @call_will_return_but_has_loop()
define void @call_will_return_but_has_loop() #0 {
  br label %label1
label1:
  tail call void @will_return()
  br label %label2
label2:
  br label %label1
}


; TEST 10 (positive case)
; invoke a function with willreturn

; ATTRIBUTOR: Function Attrs: noinline uwtable willreturn
; ATTRIBUTOR-NEXT: declare i1 @maybe_raise_exception()
declare i1 @maybe_raise_exception() #1 willreturn

; ATTRIBUTOR: Function Attrs: nounwind willreturn
; ATTRIBUTOR-NEXT: define void @invoke_test()
define void @invoke_test() personality i32 (...)* @__gxx_personality_v0 {
  invoke i1 @maybe_raise_exception()
      to label %N unwind label %F
  N:
    ret void
  F:
    %val = landingpad { i8*, i32 }
                  catch i8* null
    ret void
}

declare i32 @__gxx_personality_v0(...)


; TEST 11 (positive case)
; constant trip count
; int loop_constant_trip_count(int*p){
;    int ans = 0;
;    for(int i = 0;i<10;i++){
;        ans += p[i];
;    }
;    return ans;
; }

; ATTRIBUTOR_MODULE: Function Attrs: argmemonly nofree noinline nosync nounwind readonly uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: argmemonly nofree noinline norecurse nosync nounwind readonly uwtable willreturn
; ATTRIBUTOR-NEXT: define i32 @loop_constant_trip_count(i32* nocapture nofree readonly %0)
define i32 @loop_constant_trip_count(i32* nocapture readonly %0) #0 {
  br label %3

; <label>:2:                                      ; preds = %3
  ret i32 %8

; <label>:3:                                      ; preds = %3, %1
  %4 = phi i64 [ 0, %1 ], [ %9, %3 ]
  %5 = phi i32 [ 0, %1 ], [ %8, %3 ]
  %6 = getelementptr inbounds i32, i32* %0, i64 %4
  %7 = load i32, i32* %6, align 4
  %8 = add nsw i32 %7, %5
  %9 = add nuw nsw i64 %4, 1
  %10 = icmp eq i64 %9, 10
  br i1 %10, label %2, label %3
}


; TEST 12 (negative case)
; unbounded trip count

; int loop_trip_count_unbound(unsigned s,unsigned e, int *p, int offset){
;     int ans = 0;
;     for(unsigned i = s;i != e;i+=offset){
;         ans += p[i];
;     }
;     return ans;
; }
; FNATTR-NEXT: define i32 @loop_trip_count_unbound(i32 %0, i32 %1, i32* nocapture readonly %2, i32 %3) local_unnamed_addr
; ATTRIBUTOR_MODULE: Function Attrs: argmemonly nofree noinline nosync nounwind readonly uwtable
; ATTRIBUTOR_CGSCC: Function Attrs: argmemonly nofree noinline norecurse nosync nounwind readonly uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define i32 @loop_trip_count_unbound(i32 %0, i32 %1, i32* nocapture nofree readonly %2, i32 %3) local_unnamed_addr
define i32 @loop_trip_count_unbound(i32 %0, i32 %1, i32* nocapture readonly %2, i32 %3) local_unnamed_addr #0 {
  %5 = icmp eq i32 %0, %1
  br i1 %5, label %6, label %8

; <label>:6:                                      ; preds = %8, %4
  %7 = phi i32 [ 0, %4 ], [ %14, %8 ]
  ret i32 %7

; <label>:8:                                      ; preds = %4, %8
  %9 = phi i32 [ %15, %8 ], [ %0, %4 ]
  %10 = phi i32 [ %14, %8 ], [ 0, %4 ]
  %11 = zext i32 %9 to i64
  %12 = getelementptr inbounds i32, i32* %2, i64 %11
  %13 = load i32, i32* %12, align 4
  %14 = add nsw i32 %13, %10
  %15 = add i32 %9, %3
  %16 = icmp eq i32 %15, %1
  br i1 %16, label %6, label %8
}


; TEST 13 (positive case)
; Function Attrs: norecurse nounwind readonly uwtable
;  int loop_trip_dec(int n, int *p){
;    int ans = 0;
;    for(;n >= 0;n--){
;        ans += p[n];
;    }
;    return ans;
;  }


; ATTRIBUTOR_MODULE: Function Attrs: argmemonly nofree noinline nosync nounwind readonly uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: argmemonly nofree noinline norecurse nosync nounwind readonly uwtable willreturn
; ATTRIBUTOR-NEXT: define i32 @loop_trip_dec(i32 %0, i32* nocapture nofree readonly %1) local_unnamed_addr

define i32 @loop_trip_dec(i32 %0, i32* nocapture readonly %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %0, -1
  br i1 %3, label %4, label %14

; <label>:4:                                      ; preds = %2
  %5 = sext i32 %0 to i64
  br label %6

; <label>:6:                                      ; preds = %4, %6
  %7 = phi i64 [ %5, %4 ], [ %12, %6 ]
  %8 = phi i32 [ 0, %4 ], [ %11, %6 ]
  %9 = getelementptr inbounds i32, i32* %1, i64 %7
  %10 = load i32, i32* %9, align 4
  %11 = add nsw i32 %10, %8
  %12 = add nsw i64 %7, -1
  %13 = icmp sgt i64 %7, 0
  br i1 %13, label %6, label %14

; <label>:14:                                     ; preds = %6, %2
  %15 = phi i32 [ 0, %2 ], [ %11, %6 ]
  ret i32 %15
}

; TEST 14 (positive case)
; multiple return

; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: nofree noinline norecurse nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR-NEXT: define i32 @multiple_return(i32 %a)
define i32 @multiple_return(i32 %a) #0 {
  %b =  icmp eq i32 %a, 0
  br i1 %b, label %t, label %f

t:
  ret i32 1
f:
  ret i32 0
}

; TEST 15 (positive & negative case)
; unreachable exit

; 15.1 (positive case)
; ATTRIBUTOR_MODULE: Function Attrs: noinline nounwind uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: noinline norecurse nounwind uwtable willreturn
; ATTRIBUTOR-NEXT: define void @unreachable_exit_positive1()
define void @unreachable_exit_positive1() #0 {
  tail call void @will_return()
  ret void

unreachable_label:
  tail call void @exit(i32 0)
  unreachable
}

; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: nofree noinline norecurse nosync nounwind readnone uwtable willreturn
; ATTRIBUTOR-NEXT: define i32 @unreachable_exit_positive2(i32 %0)
define i32 @unreachable_exit_positive2(i32) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %5, %1
  %4 = phi i32 [ 1, %1 ], [ %8, %5 ]
  ret i32 %4

; <label>:5:                                      ; preds = %1, %5
  %6 = phi i32 [ %9, %5 ], [ 1, %1 ]
  %7 = phi i32 [ %8, %5 ], [ 1, %1 ]
  %8 = mul nsw i32 %6, %7
  %9 = add nuw nsw i32 %6, 1
  %10 = icmp eq i32 %6, %0
  br i1 %10, label %3, label %5

unreachable_label:
  tail call void @exit(i32 0)
  unreachable
}


;15.2

; ATTRIBUTOR: Function Attrs: noinline noreturn nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @unreachable_exit_negative1()
define void @unreachable_exit_negative1() #0 {
  tail call void @exit(i32 0)
  ret void

unreachable_label:
  tail call void @exit(i32 0)
  unreachable
}

; ATTRIBUTOR_MODULE: Function Attrs: nofree noinline noreturn nosync nounwind readnone uwtable
; ATTRIBUTOR_CGSCC: Function Attrs: nofree noinline norecurse noreturn nosync nounwind readnone uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @unreachable_exit_negative2()
define void @unreachable_exit_negative2() #0 {

  br label %L1
L1:
  br label %L2
L2:
  br label %L1

unreachable_label:
  tail call void @exit(i32 0)
  unreachable
}

; ATTRIBUTOR: Function Attrs: noreturn nounwind
; ATTRIBUTOR-NEXT: declare void @llvm.eh.sjlj.longjmp(i8*)
declare void @llvm.eh.sjlj.longjmp(i8*)

; ATTRIBUTOR: Function Attrs: noinline noreturn nounwind uwtable
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @call_longjmp(i8* nocapture readnone %0) local_unnamed_addr
define void @call_longjmp(i8* nocapture readnone %0) local_unnamed_addr #0 {
  tail call void @llvm.eh.sjlj.longjmp(i8* %0)
  ret void
}


; TEST 16 (negative case) 
; int infinite_loop_inside_bounded_loop(int n) {
;   int ans = 0;
;   for (int i = 0; i < n; i++) {
;     while (1)
;       ans++;
;   }
;   return ans;
; }

; ATTRIBUTOR_MODULE: Function Attrs:  nofree nosync nounwind readnone
; ATTRIBUTOR_CGSCC: Function Attrs: nofree norecurse nosync nounwind readnone
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define i32 @infinite_loop_inside_bounded_loop(i32 %n)
define i32 @infinite_loop_inside_bounded_loop(i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %while.cond

while.cond:                                       ; preds = %while.body, %for.body
  br label %while.body

while.body:                                       ; preds = %while.cond
  br label %while.cond

for.inc:                                          ; No predecessors!
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret i32 0
}


; TEST 17 (positive case)
; Nested loops with constant max trip count
; int bounded_nested_loops(int n) {
;   int ans = 0;
;   for (int i = 0; i < n; i++) {
;     while (n--) {
;       ans++;
;     }
;   }
;   return ans;
; }

; ATTRIBUTOR_MODULE: Function Attrs:  nofree nosync nounwind readnone willreturn
; ATTRIBUTOR_CGSCC: Function Attrs: nofree norecurse nosync nounwind readnone willreturn
; ATTRIBUTOR-NEXT: define i32 @bounded_nested_loops(i32 %n)
define i32 @bounded_nested_loops(i32 %n) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %ans.0 = phi i32 [ 0, %entry ], [ %tmp, %for.inc ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ -1, %for.inc ]
  %cmp = icmp slt i32 %i.0, %n.addr.0
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %ans.0.lcssa = phi i32 [ %ans.0, %for.cond ]
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %while.cond

while.cond:                                       ; preds = %while.body, %for.body
  br i1 true, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  br label %while.cond

while.end:                                        ; preds = %while.cond
  %tmp = add i32 %n.addr.0, %ans.0
  br label %for.inc

for.inc:                                          ; preds = %while.end
  %inc1 = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  ret i32 %ans.0.lcssa
}


; TEST 18 (negative case)
; int bounded_loop_inside_unbounded_loop(int n) {
;   int ans = 0;
;   while (n++) {
;     for (int i = 0; i < n; i++) {
;       ans++;
;     }
;   }
;   return ans;
; }

; ATTRIBUTOR_MODULE: Function Attrs:  nofree nosync nounwind readnone
; ATTRIBUTOR_CGSCC: Function Attrs: nofree norecurse nosync nounwind readnone
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define i32 @bounded_loop_inside_unbounded_loop(i32 %n)
define i32 @bounded_loop_inside_unbounded_loop(i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %for.end, %entry
  %ans.0 = phi i32 [ 0, %entry ], [ %tmp2, %for.end ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ %inc, %for.end ]
  %tmp = icmp sgt i32 %n.addr.0, -1
  %smax = select i1 %tmp, i32 %n.addr.0, i32 -1
  %inc = add nsw i32 %n.addr.0, 1
  %tobool = icmp eq i32 %n.addr.0, 0
  br i1 %tobool, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %tmp1 = add i32 %ans.0, 1
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %while.body
  br i1 true, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond
  %tmp2 = add i32 %tmp1, %smax
  br label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  br label %for.cond

for.end:                                          ; preds = %for.cond.cleanup
  br label %while.cond

while.end:                                        ; preds = %while.cond
  %ans.0.lcssa = phi i32 [ %ans.0, %while.cond ]
  ret i32 %ans.0.lcssa
}


; TEST 19 (negative case)
; int nested_unbounded_loops(int n) {
;   int ans = 0;
;   while (n--) {
;     while (n--) {
;       ans++;
;     }
;     while (n--) {
;       ans++;
;     }
;   }
;   return ans;
; }

; ATTRIBUTOR_MODULE: Function Attrs:  nofree nosync nounwind readnone
; ATTRIBUTOR_CGSCC: Function Attrs: nofree norecurse nosync nounwind readnone
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define i32 @nested_unbounded_loops(i32 %n)
define i32 @nested_unbounded_loops(i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.end10, %entry
  %ans.0 = phi i32 [ 0, %entry ], [ %tmp1, %while.end10 ]
  %n.addr.0 = phi i32 [ %n, %entry ], [ -1, %while.end10 ]
  %tobool = icmp eq i32 %n.addr.0, 0
  br i1 %tobool, label %while.end11, label %while.body

while.body:                                       ; preds = %while.cond
  br label %while.cond1

while.cond1:                                      ; preds = %while.body4, %while.body
  br i1 true, label %while.end, label %while.body4

while.body4:                                      ; preds = %while.cond1
  br label %while.cond1

while.end:                                        ; preds = %while.cond1
  %tmp = add i32 %n.addr.0, -2
  br label %while.cond5

while.cond5:                                      ; preds = %while.body8, %while.end
  br i1 true, label %while.end10, label %while.body8

while.body8:                                      ; preds = %while.cond5
  br label %while.cond5

while.end10:                                      ; preds = %while.cond5
  %tmp1 = add i32 %tmp, %ans.0
  br label %while.cond

while.end11:                                      ; preds = %while.cond
  %ans.0.lcssa = phi i32 [ %ans.0, %while.cond ]
  ret i32 %ans.0.lcssa
}


; TEST 20 (negative case)
;    void non_loop_cycle(int n) {
;      if (fact_loop(n)>5)
;        goto entry1;
;      else
;        goto entry2;
;
;    entry1:
;      if (fact_loop(n)>5)
;        goto exit;
;      else
;        goto entry2;
;    entry2:
;      if (fact_loop(n)>5)
;        goto exit;
;      else
;        goto entry1;
;    exit:
;      return;
;    }

; ATTRIBUTOR_MODULE: Function Attrs:  nofree nosync nounwind readnone
; ATTRIBUTOR_CGSCC: Function Attrs: nofree norecurse nosync nounwind readnone
; ATTRIBUTOR-NOT: willreturn
; ATTRIBUTOR-NEXT: define void @non_loop_cycle(i32 %n)
define void @non_loop_cycle(i32 %n) {
entry:
  %call = call i32 @fact_loop(i32 %n)
  %cmp = icmp sgt i32 %call, 5
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %entry1

if.else:                                          ; preds = %entry
  br label %entry2

entry1:                                           ; preds = %if.else8, %if.then
  %call1 = call i32 @fact_loop(i32 %n)
  %cmp2 = icmp sgt i32 %call1, 5
  br i1 %cmp2, label %if.then3, label %if.else4

if.then3:                                         ; preds = %entry1
  br label %exit

if.else4:                                         ; preds = %entry1
  br label %entry2

entry2:                                           ; preds = %if.else4, %if.else
  %call5 = call i32 @fact_loop(i32 %n)
  %cmp6 = icmp sgt i32 %call5, 5
  br i1 %cmp6, label %if.then7, label %if.else8

if.then7:                                         ; preds = %entry2
  br label %exit

if.else8:                                         ; preds = %entry2
  br label %entry1

exit:                                             ; preds = %if.then7, %if.then3
  ret void
}

attributes #0 = { nounwind uwtable noinline }
attributes #1 = { uwtable noinline }
