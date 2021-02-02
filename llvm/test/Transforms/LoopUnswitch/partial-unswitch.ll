; RUN: opt -loop-unswitch -verify-dom-info -verify-memoryssa -S -enable-new-pm=0 %s | FileCheck %s

declare void @clobber()

define i32 @partial_unswitch_true_successor(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[SPLIT_TRUE_PH:[a-z._]+]], label %[[FALSE_CRIT:[a-z._]+]]

; CHECK:      [[FALSE_CRIT]]:
; CHECK-NEXT:   br label %[[FALSE_PH:[a-z.]+]]

; CHECK:      [[SPLIT_TRUE_PH]]:
; CHECK-NEXT:   br label %[[TRUE_HEADER:[a-z.]+]]

; CHECK: [[TRUE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[TRUE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[TRUE_C:%[a-z.0-9]+]] = icmp eq i32 [[TRUE_LV]], 100
; CHECK-NEXT:    br i1 true, label %[[TRUE_NOCLOBBER:.+]], label %[[TRUE_CLOBBER:[a-z0-9._]+]]

; CHECK: [[TRUE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_NOCLOBBER]]:
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[TRUE_HEADER]]


; CHECK:      [[FALSE_PH]]:
; CHECK-NEXT:   br label %[[FALSE_HEADER:[a-z.]+]]

; CHECK: [[FALSE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[FALSE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[FALSE_C:%[a-z.0-9]+]] = icmp eq i32 [[FALSE_LV]], 100
; CHECK-NEXT:     br i1 [[FALSE_C]], label  %[[FALSE_NOCLOBBER:.+]], label %[[FALSE_CLOBBER:[a-z0-9._]+]]

; CHECK: [[FALSE_NOCLOBBER]]:
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[FALSE_HEADER]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_false_successor(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_false_successor
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[SPLIT_TRUE_PH:[a-z._]+]], label %[[FALSE_CRIT:[a-z._]+]]

; CHECK:      [[FALSE_CRIT]]:
; CHECK-NEXT:   br label %[[FALSE_PH:[a-z.]+]]

; CHECK:      [[SPLIT_TRUE_PH]]:
; CHECK-NEXT:   br label %[[TRUE_HEADER:[a-z.]+]]

; CHECK: [[TRUE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[TRUE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[TRUE_C:%[a-z.0-9]+]] = icmp eq i32 [[TRUE_LV]], 100
; CHECK-NEXT:    br i1 [[TRUE_C]], label %[[TRUE_CLOBBER:.+]], label %[[TRUE_NOCLOBBER:[a-z0-9._]+]]

; CHECK: [[TRUE_NOCLOBBER]]:
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[TRUE_HEADER]]


; CHECK:      [[FALSE_PH]]:
; CHECK-NEXT:   br label %[[FALSE_HEADER:[a-z.]+]]

; CHECK: [[FALSE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[FALSE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[FALSE_C:%[a-z.0-9]+]] = icmp eq i32 [[FALSE_LV]], 100
; CHECK-NEXT:     br i1 false, label  %[[FALSE_CLOBBER:.+]], label %[[FALSE_NOCLOBBER:[a-z0-9._]+]]

; CHECK: [[FALSE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_NOCLOBBER]]:
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[FALSE_HEADER]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %clobber, label %noclobber

clobber:
  call void @clobber()
  br label %loop.latch

noclobber:
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswtich_gep_load_icmp(i32** %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswtich_gep_load_icmp
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[GEP:%[a-z.0-9]+]] = getelementptr i32*, i32** %ptr, i32 1
; CHECK-NEXT:   [[LV0:%[a-z.0-9]+]] = load i32*, i32** [[GEP]]
; CHECK-NEXT:   [[LV1:%[a-z.0-9]+]] = load i32, i32* [[LV0]]
; CHECK-NEXT:   [[C:%[a-z.0-9]+]] = icmp eq i32 [[LV1]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[SPLIT_TRUE_PH:[a-z._]+]], label %[[FALSE_CRIT:[a-z._]+]]

; CHECK:      [[FALSE_CRIT]]:
; CHECK-NEXT:   br label %[[FALSE_PH:[a-z.]+]]

; CHECK:      [[SPLIT_TRUE_PH]]:
; CHECK-NEXT:   br label %[[TRUE_HEADER:[a-z.]+]]

; CHECK: [[TRUE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:   [[TRUE_GEP:%[a-z.0-9]+]] = getelementptr i32*, i32** %ptr, i32 1
; CHECK-NEXT:   [[TRUE_LV0:%[a-z.0-9]+]] = load i32*, i32** [[TRUE_GEP]]
; CHECK-NEXT:   [[TRUE_LV1:%[a-z.0-9]+]] = load i32, i32* [[TRUE_LV0]]
; CHECK-NEXT:   [[TRUE_C:%[a-z.0-9]+]] = icmp eq i32 [[TRUE_LV1]], 100
; CHECK-NEXT:   br i1 true, label %[[TRUE_NOCLOBBER:.+]], label %[[TRUE_CLOBBER:[a-z0-9._]+]]

; CHECK: [[TRUE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_NOCLOBBER]]:
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[TRUE_HEADER]]

; CHECK:      [[FALSE_PH]]:
; CHECK-NEXT:   br label %[[FALSE_HEADER:[a-z.]+]]

; CHECK: [[FALSE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:   [[FALSE_GEP:%[a-z.0-9]+]] = getelementptr i32*, i32** %ptr, i32 1
; CHECK-NEXT:   [[FALSE_LV0:%[a-z.0-9]+]] = load i32*, i32** [[FALSE_GEP]]
; CHECK-NEXT:   [[FALSE_LV1:%[a-z.0-9]+]] = load i32, i32* [[FALSE_LV0]]
; CHECK-NEXT:   [[FALSE_C:%[a-z.0-9]+]] = icmp eq i32 [[FALSE_LV1]], 100
; CHECK-NEXT:   br i1 [[FALSE_C]], label  %[[FALSE_NOCLOBBER:.+]], label %[[FALSE_CLOBBER:[a-z0-9._]+]]

; CHECK: [[FALSE_NOCLOBBER]]:
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]


; CHECK: [[FALSE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[FALSE_HEADER]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr i32*, i32** %ptr, i32 1
  %lv.1 = load i32*, i32** %gep
  %lv = load i32, i32* %lv.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_reduction_phi(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_reduction_phi
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[SPLIT_TRUE_PH:[a-z._]+]], label %[[FALSE_CRIT:[a-z._]+]]

; CHECK:      [[FALSE_CRIT]]:
; CHECK-NEXT:   br label %[[FALSE_PH:[a-z.]+]]

; CHECK:      [[SPLIT_TRUE_PH]]:
; CHECK-NEXT:   br label %[[TRUE_HEADER:[a-z.]+]]

; CHECK: [[TRUE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[TRUE_RED:%[a-z.0-9]+]] = phi i32 [ 20, %[[SPLIT_TRUE_PH]] ], [ [[TRUE_RED_NEXT:%[a-z.0-9]+]], %[[TRUE_LATCH:[a-z.0-9]+]]
; CHECK-NEXT:    [[TRUE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[TRUE_C:%[a-z.0-9]+]] = icmp eq i32 [[TRUE_LV]], 100
; CHECK-NEXT:    br i1 [[TRUE_C]], label %[[TRUE_CLOBBER:.+]], label %[[TRUE_NOCLOBBER:[a-z0-9._]+]]

; CHECK: [[TRUE_NOCLOBBER]]:
; CHECK-NEXT:  [[TRUE_ADD10:%.+]] = add i32 [[TRUE_RED]], 10
; CHECK-NEXT:  br label %[[TRUE_LATCH]]

; CHECK: [[TRUE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  [[TRUE_ADD5:%.+]] = add i32 [[TRUE_RED]], 5
; CHECK-NEXT:  br label %[[TRUE_LATCH]]

; CHECK: [[TRUE_LATCH]]:
; CHECK-NEXT:   [[TRUE_RED_NEXT]] = phi i32 [ [[TRUE_ADD5]], %[[TRUE_CLOBBER]] ], [ [[TRUE_ADD10]], %[[TRUE_NOCLOBBER]] ]
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[TRUE_HEADER]]


; CHECK:      [[FALSE_PH]]:
; CHECK-NEXT:   br label %[[FALSE_HEADER:[a-z.]+]]

; CHECK: [[FALSE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[FALSE_RED:%[a-z.0-9]+]] = phi i32 [ 20, %[[FALSE_PH]] ], [ [[FALSE_RED_NEXT:%[a-z.0-9]+]], %[[FALSE_LATCH:[a-z.0-9]+]]
; CHECK-NEXT:    [[FALSE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[FALSE_C:%[a-z.0-9]+]] = icmp eq i32 [[FALSE_LV]], 100
; CHECK-NEXT:     br i1 false, label  %[[FALSE_CLOBBER:.+]], label %[[FALSE_NOCLOBBER:[a-z0-9._]+]]

; CHECK: [[FALSE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  [[FALSE_ADD5:%.+]] = add i32 [[FALSE_RED]], 5
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_NOCLOBBER]]:
; CHECK-NEXT:  [[FALSE_ADD10:%.+]] = add i32 [[FALSE_RED]], 10
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_LATCH]]:
; CHECK-NEXT:   [[FALSE_RED_NEXT]] = phi i32 [ [[FALSE_ADD5]], %[[FALSE_CLOBBER]] ], [ [[FALSE_ADD10]], %[[FALSE_NOCLOBBER]] ]
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[FALSE_HEADER]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %red = phi i32 [ 20, %entry ], [ %red.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %clobber, label %noclobber

clobber:
  call void @clobber()
  %add.5 = add i32 %red, 5
  br label %loop.latch

noclobber:
  %add.10 = add i32 %red, 10
  br label %loop.latch

loop.latch:
  %red.next = phi i32 [ %add.5, %clobber ], [ %add.10, %noclobber ]
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  %red.next.lcssa = phi i32 [ %red.next, %loop.latch ]
  ret i32 %red.next.lcssa
}

; Partial unswitching is possible, because the store in %noclobber does not
; alias the load of the condition.
define i32 @partial_unswitch_true_successor_noclobber(i32* noalias %ptr.1, i32* noalias %ptr.2, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor
; CHECK-NEXT:  entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr.1, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]], label %[[SPLIT_TRUE_PH:[a-z._]+]], label %[[FALSE_CRIT:[a-z._]+]]

; CHECK:      [[FALSE_CRIT]]:
; CHECK-NEXT:   br label %[[FALSE_PH:[a-z.]+]]

; CHECK:      [[SPLIT_TRUE_PH]]:
; CHECK-NEXT:   br label %[[TRUE_HEADER:[a-z.]+]]

; CHECK: [[TRUE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[TRUE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr.1, align 4
; CHECK-NEXT:    [[TRUE_C:%[a-z.0-9]+]] = icmp eq i32 [[TRUE_LV]], 100
; CHECK-NEXT:    br i1 true, label %[[TRUE_NOCLOBBER:.+]], label %[[TRUE_CLOBBER:[a-z0-9._]+]]

; CHECK: [[TRUE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_NOCLOBBER]]:
; CHECK-NEXT:  [[TRUE_GEP:%[a-z0-9._]+]]  = getelementptr i32, i32* %ptr.2
; CHECK-NEXT:  store i32 [[TRUE_LV]], i32* [[TRUE_GEP]], align 4
; CHECK-NEXT:  br label %[[TRUE_LATCH:[a-z0-9._]+]]

; CHECK: [[TRUE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[TRUE_HEADER]]


; CHECK:      [[FALSE_PH]]:
; CHECK-NEXT:   br label %[[FALSE_HEADER:[a-z.]+]]

; CHECK: [[FALSE_HEADER]]:
; CHECK-NEXT:   phi i32
; CHECK-NEXT:    [[FALSE_LV:%[a-z.0-9]+]] = load i32, i32* %ptr.1, align 4
; CHECK-NEXT:    [[FALSE_C:%[a-z.0-9]+]] = icmp eq i32 [[FALSE_LV]], 100
; CHECK-NEXT:     br i1 [[FALSE_C]], label  %[[FALSE_NOCLOBBER:.+]], label %[[FALSE_CLOBBER:[a-z0-9._]+]]

; CHECK: [[FALSE_NOCLOBBER]]:
; CHECK-NEXT:  [[FALSE_GEP:%[a-z0-9._]+]]  = getelementptr i32, i32* %ptr.2
; CHECK-NEXT:  store i32 [[FALSE_LV]], i32* [[FALSE_GEP]], align 4
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_CLOBBER]]:
; CHECK-NEXT:  call
; CHECK-NEXT:  br label %[[FALSE_LATCH:[a-z0-9._]+]]

; CHECK: [[FALSE_LATCH]]:
; CHECK-NEXT:   icmp
; CHECK-NEXT:   add
; CHECK-NEXT:   br {{.*}} label %[[FALSE_HEADER]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %gep.1 = getelementptr i32, i32* %ptr.2, i32 %iv
  store i32 %lv, i32* %gep.1
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define void @no_partial_unswitch_phi_cond(i1 %lc, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_phi_cond
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %sc = phi i1 [ %lc, %entry ], [ true, %loop.latch ]
  br i1 %sc, label %clobber, label %noclobber

clobber:
  call void @clobber()
  br label %loop.latch

noclobber:
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define void @no_partial_unswitch_clobber_latch(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_clobber_latch
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  call void @clobber()
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define void @no_partial_unswitch_clobber_header(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_clobber_header
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  call void @clobber()
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define void @no_partial_unswitch_clobber_both(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_clobber_both
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  call void @clobber()
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret void
}

define i32 @no_partial_unswitch_true_successor_storeclobber(i32* %ptr.1, i32* %ptr.2, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_true_successor_storeclobber
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr.1
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  %gep.1 = getelementptr i32, i32* %ptr.2, i32 %iv
  store i32 %lv, i32* %gep.1
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Make sure the duplicated instructions are moved to a preheader that always
; executes when the loop body also executes. Do not check the unswitched code,
; because it is already checked in the @partial_unswitch_true_successor test
; case.
define i32 @partial_unswitch_true_successor_preheader_insertion(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor_preheader_insertion(
; CHECK-NEXT:  entry:
; CHECK-NEXT:   [[EC:%[a-z]+]] = icmp ne i32* %ptr, null
; CHECK-NEXT:   br i1 [[EC]], label %[[PH:[a-z0-9.]+]], label %[[EXIT:[a-z0-9.]+]]

; CHECK: [[PH]]:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]]
;
entry:
  %ec = icmp ne i32* %ptr, null
  br i1 %ec, label %loop.ph, label %exit

loop.ph:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %loop.ph ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Make sure the duplicated instructions are hoisted just before the branch of
; the preheader. Do not check the unswitched code, because it is already checked
; in the @partial_unswitch_true_successor test case
define i32 @partial_unswitch_true_successor_insert_point(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor_insert_point(
; CHECK-NEXT:  entry:
; CHECK-NEXT:   call void @clobber()
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]]
;
entry:
  call void @clobber()
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Make sure invariant instructions in the loop are also hoisted to the preheader.
; Do not check the unswitched code, because it is already checked in the
; @partial_unswitch_true_successor test case
define i32 @partial_unswitch_true_successor_hoist_invariant(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_successor_hoist_invariant(
; CHECK-NEXT:  entry:
; CHECK-NEXT:   [[GEP:%[0-9]+]] = getelementptr i32, i32* %ptr, i64 1
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* [[GEP]], align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]]
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep = getelementptr i32, i32* %ptr, i64 1
  %lv = load i32, i32* %gep
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Do not unswitch if the condition depends on an atomic load. Duplicating such
; loads is not safe.
define i32 @no_partial_unswitch_atomic_load_unordered(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_atomic_load_unordered
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load atomic i32, i32* %ptr unordered, align 4
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

; Do not unswitch if the condition depends on an atomic load. Duplicating such
; loads is not safe.
define i32 @no_partial_unswitch_atomic_load_monotonic(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_atomic_load_monotonic
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load atomic i32, i32* %ptr monotonic, align 4
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}


declare i32 @get_value()

; Do not unswitch if the condition depends on a call, that may clobber memory.
; Duplicating such a call is not safe.
define i32 @no_partial_unswitch_cond_call(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_cond_call
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = call i32 @get_value()
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %clobber

noclobber:
  br label %loop.latch

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @no_partial_unswitch_true_successor_exit(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_true_successor_exit
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %exit, label %clobber

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @no_partial_unswitch_true_same_successor(i32* %ptr, i32 %N) {
; CHECK-LABEL: @no_partial_unswitch_true_same_successor
; CHECK-LABEL: entry:
; CHECK-NEXT:   br label %loop.header
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %noclobber, label %noclobber

noclobber:
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}

define i32 @partial_unswitch_true_to_latch(i32* %ptr, i32 %N) {
; CHECK-LABEL: @partial_unswitch_true_to_latch
; CHECK-LABEL: entry:
; CHECK-NEXT:   [[LV:%[0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:   [[C:%[0-9]+]] = icmp eq i32 [[LV]], 100
; CHECK-NEXT:   br i1 [[C]],
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %lv = load i32, i32* %ptr
  %sc = icmp eq i32 %lv, 100
  br i1 %sc, label %loop.latch, label %clobber

clobber:
  call void @clobber()
  br label %loop.latch

loop.latch:
  %c = icmp ult i32 %iv, %N
  %iv.next = add i32 %iv, 1
  br i1 %c, label %loop.header, label %exit

exit:
  ret i32 10
}
