; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck --check-prefixes=CHECK,RETP %s
; RUN: sed -e 's,+retpoline,-retpoline,g' %s | opt -S -passes=wholeprogramdevirt -whole-program-visibility | FileCheck --check-prefixes=CHECK,NORETP %s

; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck --check-prefixes=CHECK,RETP %s

; RUN: opt -passes='wholeprogramdevirt,default<O3>' -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t  -S -o - %s | FileCheck --check-prefixes=CHECK %s

; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY:      TypeIdMap:       
; SUMMARY-NEXT:  typeid1_rv:
; SUMMARY-NEXT:    TTRes:
; SUMMARY-NEXT:      Kind:            Unknown
; SUMMARY-NEXT:      SizeM1BitWidth:  0
; SUMMARY-NEXT:      AlignLog2:       0
; SUMMARY-NEXT:      SizeM1:          0
; SUMMARY-NEXT:      BitMask:         0
; SUMMARY-NEXT:      InlineBits:      0
; SUMMARY-NEXT:    WPDRes:
; SUMMARY-NEXT:      0:
; SUMMARY-NEXT:        Kind:            BranchFunnel
; SUMMARY-NEXT:        SingleImplName:  ''
; SUMMARY-NEXT:        ResByArg:
; SUMMARY-NEXT:  typeid2_rv:
; SUMMARY-NEXT:    TTRes:
; SUMMARY-NEXT:      Kind:            Unknown
; SUMMARY-NEXT:      SizeM1BitWidth:  0
; SUMMARY-NEXT:      AlignLog2:       0
; SUMMARY-NEXT:      SizeM1:          0
; SUMMARY-NEXT:      BitMask:         0
; SUMMARY-NEXT:      InlineBits:      0
; SUMMARY-NEXT:    WPDRes:
; SUMMARY-NEXT:      0:
; SUMMARY-NEXT:        Kind:            Indir
; SUMMARY-NEXT:        SingleImplName:  ''
; SUMMARY-NEXT:        ResByArg:
; SUMMARY-NEXT:   typeid3_rv:
; SUMMARY-NEXT:     TTRes:           
; SUMMARY-NEXT:       Kind:            Unknown
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:          
; SUMMARY-NEXT:       0:               
; SUMMARY-NEXT:         Kind:            BranchFunnel
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:        
; SUMMARY-NEXT:   typeid3:
; SUMMARY-NEXT:     TTRes:           
; SUMMARY-NEXT:       Kind:            Unknown
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:          
; SUMMARY-NEXT:       0:               
; SUMMARY-NEXT:         Kind:            BranchFunnel
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:        
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:           
; SUMMARY-NEXT:       Kind:            Unknown
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:          
; SUMMARY-NEXT:       0:               
; SUMMARY-NEXT:         Kind:            BranchFunnel
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:        
; SUMMARY-NEXT:   typeid2:
; SUMMARY-NEXT:     TTRes:           
; SUMMARY-NEXT:       Kind:            Unknown
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:          
; SUMMARY-NEXT:       0:               
; SUMMARY-NEXT:         Kind:            Indir
; SUMMARY-NEXT:         SingleImplName:  ''
; SUMMARY-NEXT:         ResByArg:        

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1_1 = constant [1 x ptr] [ptr @vf1_1], !type !0
@vt1_2 = constant [1 x ptr] [ptr @vf1_2], !type !0

declare i32 @vf1_1(ptr %this, i32 %arg)
declare i32 @vf1_2(ptr %this, i32 %arg)

@vt2_1 = constant [1 x ptr] [ptr @vf2_1], !type !1
@vt2_2 = constant [1 x ptr] [ptr @vf2_2], !type !1
@vt2_3 = constant [1 x ptr] [ptr @vf2_3], !type !1
@vt2_4 = constant [1 x ptr] [ptr @vf2_4], !type !1
@vt2_5 = constant [1 x ptr] [ptr @vf2_5], !type !1
@vt2_6 = constant [1 x ptr] [ptr @vf2_6], !type !1
@vt2_7 = constant [1 x ptr] [ptr @vf2_7], !type !1
@vt2_8 = constant [1 x ptr] [ptr @vf2_8], !type !1
@vt2_9 = constant [1 x ptr] [ptr @vf2_9], !type !1
@vt2_10 = constant [1 x ptr] [ptr @vf2_10], !type !1
@vt2_11 = constant [1 x ptr] [ptr @vf2_11], !type !1

declare i32 @vf2_1(ptr %this, i32 %arg)
declare i32 @vf2_2(ptr %this, i32 %arg)
declare i32 @vf2_3(ptr %this, i32 %arg)
declare i32 @vf2_4(ptr %this, i32 %arg)
declare i32 @vf2_5(ptr %this, i32 %arg)
declare i32 @vf2_6(ptr %this, i32 %arg)
declare i32 @vf2_7(ptr %this, i32 %arg)
declare i32 @vf2_8(ptr %this, i32 %arg)
declare i32 @vf2_9(ptr %this, i32 %arg)
declare i32 @vf2_10(ptr %this, i32 %arg)
declare i32 @vf2_11(ptr %this, i32 %arg)

@vt3_1 = constant [1 x ptr] [ptr @vf3_1], !type !2
@vt3_2 = constant [1 x ptr] [ptr @vf3_2], !type !2

declare i32 @vf3_1(ptr %this, i32 %arg)
declare i32 @vf3_2(ptr %this, i32 %arg)

@vt4_1 = constant [1 x ptr] [ptr @vf4_1], !type !3
@vt4_2 = constant [1 x ptr] [ptr @vf4_2], !type !3

declare i32 @vf4_1(ptr %this, i32 %arg)
declare i32 @vf4_2(ptr %this, i32 %arg)

declare ptr @llvm.load.relative.i32(ptr, i32)

;; These are relative vtables equivalent to the ones above.
@vt1_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1_1 to i64), i64 ptrtoint (ptr @vt1_1_rv to i64)) to i32)], !type !5
@vt1_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf1_2 to i64), i64 ptrtoint (ptr @vt1_2_rv to i64)) to i32)], !type !5

@vt2_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_1 to i64), i64 ptrtoint (ptr @vt2_1_rv to i64)) to i32)], !type !6
@vt2_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_2 to i64), i64 ptrtoint (ptr @vt2_2_rv to i64)) to i32)], !type !6
@vt2_3_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_3 to i64), i64 ptrtoint (ptr @vt2_3_rv to i64)) to i32)], !type !6
@vt2_4_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_4 to i64), i64 ptrtoint (ptr @vt2_4_rv to i64)) to i32)], !type !6
@vt2_5_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_5 to i64), i64 ptrtoint (ptr @vt2_5_rv to i64)) to i32)], !type !6
@vt2_6_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_6 to i64), i64 ptrtoint (ptr @vt2_6_rv to i64)) to i32)], !type !6
@vt2_7_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_7 to i64), i64 ptrtoint (ptr @vt2_7_rv to i64)) to i32)], !type !6
@vt2_8_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_8 to i64), i64 ptrtoint (ptr @vt2_8_rv to i64)) to i32)], !type !6
@vt2_9_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_9 to i64), i64 ptrtoint (ptr @vt2_9_rv to i64)) to i32)], !type !6
@vt2_10_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_10 to i64), i64 ptrtoint (ptr @vt2_10_rv to i64)) to i32)], !type !6
@vt2_11_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf2_11 to i64), i64 ptrtoint (ptr @vt2_11_rv to i64)) to i32)], !type !6

@vt3_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf3_1 to i64), i64 ptrtoint (ptr @vt3_1_rv to i64)) to i32)], !type !7
@vt3_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf3_2 to i64), i64 ptrtoint (ptr @vt3_2_rv to i64)) to i32)], !type !7

@vt4_1_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf4_1 to i64), i64 ptrtoint (ptr @vt4_1_rv to i64)) to i32)], !type !8
@vt4_2_rv = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @vf4_2 to i64), i64 ptrtoint (ptr @vt4_2_rv to i64)) to i32)], !type !8


; CHECK-LABEL: define i32 @fn1
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn1(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; RETP: call i32 @__typeid_typeid1_0_branch_funnel(ptr nest %vtable, ptr %obj, i32 1)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn1_rv
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn1_rv(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  ; RETP: call i32 @__typeid_typeid1_rv_0_branch_funnel(ptr nest %vtable, ptr %obj, i32 1)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn2
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn2(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call i32 %
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn2_rv
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn2_rv(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  ; CHECK: call i32 %
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn3
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn3(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !4)
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; RETP: call i32 @branch_funnel(ptr
  ; NORETP: call i32 %
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn3_rv
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn3_rv(ptr %obj) #0 {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !9)
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0)
  ; RETP: call i32 @branch_funnel.1(ptr
  ; NORETP: call i32 %
  %result = call i32 %fptr(ptr %obj, i32 1)
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn4
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn4(ptr %obj) #0 {
  %p = call i1 @llvm.type.test(ptr @vt1_1, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr @vt1_1
  ; RETP: call i32 @__typeid_typeid1_0_branch_funnel(ptr nest @vt1_1, ptr %obj, i32 1)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn4_cpy
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn4_cpy(ptr %obj) #0 {
  %p = call i1 @llvm.type.test(ptr @vt1_1, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr @vt1_1
  ; RETP: call i32 @__typeid_typeid1_0_branch_funnel(ptr nest @vt1_1, ptr %obj, i32 1)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn4_rv
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn4_rv(ptr %obj) #0 {
  %p = call i1 @llvm.type.test(ptr @vt1_1_rv, metadata !"typeid1_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr @vt1_1_rv, i32 0)
  ; RETP: call i32 @__typeid_typeid1_rv_0_branch_funnel(ptr nest @vt1_1_rv, ptr %obj, i32 1)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn4_rv_cpy
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn4_rv_cpy(ptr %obj) #0 {
  %p = call i1 @llvm.type.test(ptr @vt1_1_rv, metadata !"typeid1_rv")
  call void @llvm.assume(i1 %p)
  %fptr = call ptr @llvm.load.relative.i32(ptr @vt1_1_rv, i32 0)
  ; RETP: call i32 @__typeid_typeid1_rv_0_branch_funnel(ptr nest @vt1_1_rv, ptr %obj, i32 1)
  %result = call i32 %fptr(ptr %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...)
; CHECK-NEXT: musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr {{(nonnull )?}}@vt1_1, ptr {{(nonnull )?}}@vf1_1, ptr {{(nonnull )?}}@vt1_2, ptr {{(nonnull )?}}@vf1_2, ...)

; CHECK-LABEL: define hidden void @__typeid_typeid1_rv_0_branch_funnel(ptr nest %0, ...)
; CHECK-NEXT: musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr {{(nonnull )?}}@vt1_1_rv, ptr {{(nonnull )?}}@vf1_1, ptr {{(nonnull )?}}@vt1_2_rv, ptr {{(nonnull )?}}@vf1_2, ...)

; CHECK: define internal void @branch_funnel(ptr
; CHECK: define internal void @branch_funnel.1(ptr

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !4}
!4 = distinct !{}
!5 = !{i32 0, !"typeid1_rv"}
!6 = !{i32 0, !"typeid2_rv"}
!7 = !{i32 0, !"typeid3_rv"}
!8 = !{i32 0, !9}
!9 = distinct !{}

attributes #0 = { "target-features"="+retpoline" }
