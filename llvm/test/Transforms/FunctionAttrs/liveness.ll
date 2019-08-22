; RUN: opt -attributor --attributor-disable=false -attributor-max-iterations=6 -S < %s | FileCheck %s

declare void @no_return_call() nofree noreturn nounwind readnone

declare void @normal_call() readnone

declare i32 @foo()

declare i32 @foo_noreturn_nounwind() noreturn nounwind

declare i32 @foo_noreturn() noreturn

declare i32 @bar() nosync readnone

; This internal function has no live call sites, so all its BBs are considered dead,
; and nothing should be deduced for it.

; CHECK: define internal i32 @dead_internal_func(i32 %0)
define internal i32 @dead_internal_func(i32 %0) {
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

; CHECK: Function Attrs: nofree norecurse nounwind uwtable willreturn
define i32 @volatile_load(i32*) norecurse nounwind uwtable {
  %2 = load volatile i32, i32* %0, align 4
  ret i32 %2
}

; CHECK: Function Attrs: nofree norecurse nosync nounwind uwtable willreturn
; CHECK-NEXT: define internal i32 @internal_load(i32* nonnull %0)
define internal i32 @internal_load(i32*) norecurse nounwind uwtable {
  %2 = load i32, i32* %0, align 4
  ret i32 %2
}
; TEST 1: Only first block is live.

; CHECK: Function Attrs: nofree noreturn nosync nounwind
; CHECK-NEXT: define i32 @first_block_no_return(i32 %a, i32* nonnull %ptr1, i32* %ptr2)
define i32 @first_block_no_return(i32 %a, i32* nonnull %ptr1, i32* %ptr2) #0 {
entry:
  call i32 @internal_load(i32* %ptr1)
  ; CHECK: call i32 @internal_load(i32* nonnull %ptr1)
  call void @no_return_call()
  ; CHECK: call void @no_return_call()
  ; CHECK-NEXT: unreachable
  call i32 @dead_internal_func(i32 10)
  ; CHECK call i32 undef(i32 10)
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call i32 @internal_load(i32* %ptr2)
  ; CHECK: call i32 @internal_load(i32* %ptr2)
  %load = call i32 @volatile_load(i32* %ptr1)
  call void @normal_call()
  %call = call i32 @foo()
  br label %cond.end

cond.false:                                       ; preds = %entry
  call void @normal_call()
  %call1 = call i32 @bar()
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %call, %cond.true ], [ %call1, %cond.false ]
  ret i32 %cond
}

; TEST 2: cond.true is dead, but cond.end is not, since cond.false is live

; This is just an example. For example we can put a sync call in a
; dead block and check if it is deduced.

; CHECK: Function Attrs: nosync
; CHECK-NEXT: define i32 @dead_block_present(i32 %a, i32* %ptr1)
define i32 @dead_block_present(i32 %a, i32* %ptr1) #0 {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call void @no_return_call()
  ; CHECK: call void @no_return_call()
  ; CHECK-NEXT: unreachable
  %call = call i32 @volatile_load(i32* %ptr1)
  br label %cond.end

cond.false:                                       ; preds = %entry
  call void @normal_call()
  %call1 = call i32 @bar()
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %call, %cond.true ], [ %call1, %cond.false ]
  ret i32 %cond
}

; TEST 3: both cond.true and cond.false are dead, therfore cond.end is dead as well.

define i32 @all_dead(i32 %a) #0 {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call void @no_return_call()
  ; CHECK: call void @no_return_call()
  ; CHECK-NEXT: unreachable
  call i32 @dead_internal_func(i32 10)
  ; CHECK call i32 undef(i32 10)
  %call = call i32 @foo()
  br label %cond.end

cond.false:                                       ; preds = %entry
  call void @no_return_call()
  ; CHECK: call void @no_return_call()
  ; CHECK-NEXT: unreachable
  call i32 @dead_internal_func(i32 10)
  ; CHECK call i32 undef(i32 10)
  %call1 = call i32 @bar()
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %call, %cond.true ], [ %call1, %cond.false ]
  ret i32 %cond
}

declare i32 @__gxx_personality_v0(...)

; TEST 4: All blocks are live.

; CHECK: define i32 @all_live(i32 %a)
define i32 @all_live(i32 %a) #0 {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call void @normal_call()
  %call = call i32 @foo_noreturn()
  br label %cond.end

cond.false:                                       ; preds = %entry
  call void @normal_call()
  %call1 = call i32 @bar()
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %cond.true
  %cond = phi i32 [ %call, %cond.true ], [ %call1, %cond.false ]
  ret i32 %cond
}

; TEST 5 noreturn invoke instruction with a unreachable normal successor block.

; CHECK: define i32 @invoke_noreturn(i32 %a)
define i32 @invoke_noreturn(i32 %a) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call void @normal_call()
  %call = invoke i32 @foo_noreturn() to label %continue
            unwind label %cleanup
  ; CHECK:      %call = invoke i32 @foo_noreturn()
  ; CHECK-NEXT:         to label %continue unwind label %cleanup

cond.false:                                       ; preds = %entry
  call void @normal_call()
  %call1 = call i32 @bar()
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %continue
  %cond = phi i32 [ %call, %continue ], [ %call1, %cond.false ]
  ret i32 %cond

continue:
  ; CHECK:      continue:
  ; CHECK-NEXT: unreachable
  br label %cond.end

cleanup:
  %res = landingpad { i8*, i32 }
  catch i8* null
  ret i32 0
}

; TEST 4.1 noreturn invoke instruction replaced by a call and an unreachable instruction
; put after it.

; CHECK: define i32 @invoke_noreturn_nounwind(i32 %a)
define i32 @invoke_noreturn_nounwind(i32 %a) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %cond.true, label %cond.false

cond.true:                                        ; preds = %entry
  call void @normal_call()
  %call = invoke i32 @foo_noreturn_nounwind() to label %continue
            unwind label %cleanup
  ; CHECK:      call void @normal_call()
  ; CHECK-NEXT: call i32 @foo_noreturn_nounwind()
  ; CHECK-NEXT: unreachable

  ; We keep the invoke around as other attributes might have references to it.
  ; CHECK:       cond.true.split:                                  ; No predecessors!
  ; CHECK-NEXT:      invoke i32 @foo_noreturn_nounwind()

cond.false:                                       ; preds = %entry
  call void @normal_call()
  %call1 = call i32 @bar()
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %continue
  %cond = phi i32 [ %call, %continue ], [ %call1, %cond.false ]
  ret i32 %cond

continue:
  br label %cond.end

cleanup:
  %res = landingpad { i8*, i32 }
  catch i8* null
  ret i32 0
}

; TEST 6: Undefined behvior, taken from LangRef.
; FIXME: Should be able to detect undefined behavior.

; CHECK: define void @ub(i32* %0)
define void @ub(i32* %0) {
  %poison = sub nuw i32 0, 1           ; Results in a poison value.
  %still_poison = and i32 %poison, 0   ; 0, but also poison.
  %poison_yet_again = getelementptr i32, i32* %0, i32 %still_poison
  store i32 0, i32* %poison_yet_again  ; Undefined behavior due to store to poison.
  ret void
}

define void @inf_loop() #0 {
entry:
  br label %while.body

while.body:                                       ; preds = %entry, %while.body
  br label %while.body
}

; TEST 7: Infinite loop.
; FIXME: Detect infloops, and mark affected blocks dead.

define i32 @test5(i32, i32) #0 {
  %3 = icmp sgt i32 %0, %1
  br i1 %3, label %cond.if, label %cond.elseif

cond.if:                                                ; preds = %2
  %4 = tail call i32 @bar()
  br label %cond.end

cond.elseif:                                                ; preds = %2
  call void @inf_loop()
  %5 = icmp slt i32 %0, %1
  br i1 %5, label %cond.end, label %cond.else

cond.else:                                                ; preds = %cond.elseif
  %6 = tail call i32 @foo()
  br label %cond.end

cond.end:                                               ; preds = %cond.if, %cond.else, %cond.elseif
  %7 = phi i32 [ %1, %cond.elseif ], [ 0, %cond.else ], [ 0, %cond.if ]
  ret i32 %7
}

define void @rec() #0 {
entry:
  call void @rec()
  ret void
}

; TEST 8: Recursion
; FIXME: everything after first block should be marked dead
; and unreachable should be put after call to @rec().

define i32 @test6(i32, i32) #0 {
  call void @rec()
  %3 = icmp sgt i32 %0, %1
  br i1 %3, label %cond.if, label %cond.elseif

cond.if:                                                ; preds = %2
  %4 = tail call i32 @bar()
  br label %cond.end

cond.elseif:                                                ; preds = %2
  call void @rec()
  %5 = icmp slt i32 %0, %1
  br i1 %5, label %cond.end, label %cond.else

cond.else:                                                ; preds = %cond.elseif
  %6 = tail call i32 @foo()
  br label %cond.end

cond.end:                                               ; preds = %cond.if, %cond.else, %cond.elseif
  %7 = phi i32 [ %1, %cond.elseif ], [ 0, %cond.else ], [ 0, %cond.if ]
  ret i32 %7
}
; TEST 9: Recursion
; FIXME: contains recursive call to itself in cond.elseif block

define i32 @test7(i32, i32) #0 {
  %3 = icmp sgt i32 %0, %1
  br i1 %3, label %cond.if, label %cond.elseif

cond.if:                                                ; preds = %2
  %4 = tail call i32 @bar()
  br label %cond.end

cond.elseif:                                                ; preds = %2
  %5 = tail call i32 @test7(i32 %0, i32 %1)
  %6 = icmp slt i32 %0, %1
  br i1 %6, label %cond.end, label %cond.else

cond.else:                                                ; preds = %cond.elseif
  %7 = tail call i32 @foo()
  br label %cond.end

cond.end:                                               ; preds = %cond.if, %cond.else, %cond.elseif
  %8 = phi i32 [ %1, %cond.elseif ], [ 0, %cond.else ], [ 0, %cond.if ]
  ret i32 %8
}

; SCC test
;
; char a1 __attribute__((aligned(8)));
; char a2 __attribute__((aligned(16)));
;
; char* f1(char* a ){
;     return a?a:f2(&a1);
; }
; char* f2(char* a){
;     return a?f1(a):f3(&a2);
; }
;
; char* f3(char* a){
;     return a?&a1: f1(&a2);
; }

@a1 = common global i8 0, align 8
@a2 = common global i8 0, align 16

define internal i8* @f1(i8* readnone %0) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal i8* @f1(i8* readnone %0)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
; ATTRIBUTOR: %4 = tail call i8* undef(i8* nonnull align 8 @a1)
  %4 = tail call i8* @f2(i8* nonnull @a1)
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi i8* [ %4, %3 ], [ %0, %1 ]
  ret i8* %6
}

define internal i8* @f2(i8* readnone %0) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal i8* @f2(i8* readnone %0)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %5, label %3

; <label>:3:                                      ; preds = %1

; ATTRIBUTOR: %4 = tail call i8* undef(i8* nonnull align 8 %0)
  %4 = tail call i8* @f1(i8* nonnull %0)
  br label %7

; <label>:5:                                      ; preds = %1
; ATTRIBUTOR: %6 = tail call i8* undef(i8* nonnull align 16 @a2)
  %6 = tail call i8* @f3(i8* nonnull @a2)
  br label %7

; <label>:7:                                      ; preds = %5, %3
  %8 = phi i8* [ %4, %3 ], [ %6, %5 ]
  ret i8* %8
}

define internal i8* @f3(i8* readnone %0) local_unnamed_addr #0 {
; ATTRIBUTOR: define internal i8* @f3(i8* readnone %0)
  %2 = icmp eq i8* %0, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %1
; ATTRIBUTOR: %4 = tail call i8* undef(i8* nonnull align 16 @a2)
  %4 = tail call i8* @f1(i8* nonnull @a2)
  br label %5

; <label>:5:                                      ; preds = %1, %3
  %6 = phi i8* [ %4, %3 ], [ @a1, %1 ]
  ret i8* %6
}

define void @test_unreachable() {
; CHECK:       define void @test_unreachable()
; CHECK-NEXT:    call void @test_unreachable()
; CHECK-NEXT:    unreachable
; CHECK-NEXT:  }
  call void @test_unreachable()
  unreachable
}

