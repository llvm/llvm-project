; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck --check-prefixes=CHECK,RETP %s
; RUN: sed -e 's,+retpoline,-retpoline,g' %s | opt -S -passes=wholeprogramdevirt -whole-program-visibility | FileCheck --check-prefixes=CHECK,NORETP %s

; RUN: opt -passes=wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck --check-prefixes=CHECK,RETP %s

; RUN: opt -passes='wholeprogramdevirt,default<O3>' -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t  -S -o - %s | FileCheck --check-prefixes=CHECK %s

; RUN: FileCheck --check-prefix=SUMMARY %s < %t

; SUMMARY:      TypeIdMap:       
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

; CHECK-LABEL: define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...)
; CHECK-NEXT: musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr {{(nonnull )?}}@vt1_1, ptr {{(nonnull )?}}@vf1_1, ptr {{(nonnull )?}}@vt1_2, ptr {{(nonnull )?}}@vf1_2, ...)

; CHECK: define internal void @branch_funnel(ptr

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !4}
!4 = distinct !{}

attributes #0 = { "target-features"="+retpoline" }
