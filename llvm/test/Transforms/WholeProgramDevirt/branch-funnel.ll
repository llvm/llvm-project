; RUN: opt -S -wholeprogramdevirt -whole-program-visibility %s | FileCheck --check-prefixes=CHECK,RETP %s
; RUN: sed -e 's,+retpoline,-retpoline,g' %s | opt -S -wholeprogramdevirt -whole-program-visibility | FileCheck --check-prefixes=CHECK,NORETP %s

; RUN: opt -wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t -S -o - %s | FileCheck --check-prefixes=CHECK,RETP %s

; RUN: opt -wholeprogramdevirt -whole-program-visibility -wholeprogramdevirt-summary-action=export -wholeprogramdevirt-read-summary=%S/Inputs/export.yaml -wholeprogramdevirt-write-summary=%t  -O3 -S -o - %s | FileCheck --check-prefixes=CHECK %s

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

@vt1_1 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf1_1 to i8*)], !type !0
@vt1_2 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf1_2 to i8*)], !type !0

declare i32 @vf1_1(i8* %this, i32 %arg)
declare i32 @vf1_2(i8* %this, i32 %arg)

@vt2_1 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_1 to i8*)], !type !1
@vt2_2 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_2 to i8*)], !type !1
@vt2_3 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_3 to i8*)], !type !1
@vt2_4 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_4 to i8*)], !type !1
@vt2_5 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_5 to i8*)], !type !1
@vt2_6 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_6 to i8*)], !type !1
@vt2_7 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_7 to i8*)], !type !1
@vt2_8 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_8 to i8*)], !type !1
@vt2_9 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_9 to i8*)], !type !1
@vt2_10 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_10 to i8*)], !type !1
@vt2_11 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf2_11 to i8*)], !type !1

declare i32 @vf2_1(i8* %this, i32 %arg)
declare i32 @vf2_2(i8* %this, i32 %arg)
declare i32 @vf2_3(i8* %this, i32 %arg)
declare i32 @vf2_4(i8* %this, i32 %arg)
declare i32 @vf2_5(i8* %this, i32 %arg)
declare i32 @vf2_6(i8* %this, i32 %arg)
declare i32 @vf2_7(i8* %this, i32 %arg)
declare i32 @vf2_8(i8* %this, i32 %arg)
declare i32 @vf2_9(i8* %this, i32 %arg)
declare i32 @vf2_10(i8* %this, i32 %arg)
declare i32 @vf2_11(i8* %this, i32 %arg)

@vt3_1 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf3_1 to i8*)], !type !2
@vt3_2 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf3_2 to i8*)], !type !2

declare i32 @vf3_1(i8* %this, i32 %arg)
declare i32 @vf3_2(i8* %this, i32 %arg)

@vt4_1 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf4_1 to i8*)], !type !3
@vt4_2 = constant [1 x i8*] [i8* bitcast (i32 (i8*, i32)* @vf4_2 to i8*)], !type !3

declare i32 @vf4_1(i8* %this, i32 %arg)
declare i32 @vf4_2(i8* %this, i32 %arg)



; CHECK-LABEL: define i32 @fn1
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn1(i8* %obj) #0 {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i32)*
  ; RETP: {{.*}} = bitcast {{.*}} to i8*
  ; RETP: [[VT1:%.*]] = bitcast {{.*}} to i8*
  ; RETP: call i32 bitcast (void (i8*, ...)* @__typeid_typeid1_0_branch_funnel to i32 (i8*, i8*, i32)*)(i8* nest [[VT1]], i8* %obj, i32 1)
  %result = call i32 %fptr_casted(i8* %obj, i32 1)
  ; NORETP: call i32 %
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn2
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn2(i8* %obj) #0 {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i32)*
  ; CHECK: call i32 %
  %result = call i32 %fptr_casted(i8* %obj, i32 1)
  ret i32 %result
}

; CHECK-LABEL: define i32 @fn3
; CHECK-NOT: call void (...) @llvm.icall.branch.funnel
define i32 @fn3(i8* %obj) #0 {
  %vtableptr = bitcast i8* %obj to [1 x i8*]**
  %vtable = load [1 x i8*]*, [1 x i8*]** %vtableptr
  %vtablei8 = bitcast [1 x i8*]* %vtable to i8*
  %p = call i1 @llvm.type.test(i8* %vtablei8, metadata !4)
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr [1 x i8*], [1 x i8*]* %vtable, i32 0, i32 0
  %fptr = load i8*, i8** %fptrptr
  %fptr_casted = bitcast i8* %fptr to i32 (i8*, i32)*
  ; RETP: call i32 bitcast (void (i8*, ...)* @branch_funnel to
  ; NORETP: call i32 %
  %result = call i32 %fptr_casted(i8* %obj, i32 1)
  ret i32 %result
}

; CHECK-LABEL: define hidden void @__typeid_typeid1_0_branch_funnel(i8* nest %0, ...)
; CHECK-NEXT: musttail call void (...) @llvm.icall.branch.funnel(i8* %0, i8* bitcast ([1 x i8*]* {{(nonnull )?}}@vt1_1 to i8*), i32 (i8*, i32)* {{(nonnull )?}}@vf1_1, i8* bitcast ([1 x i8*]* {{(nonnull )?}}@vt1_2 to i8*), i32 (i8*, i32)* {{(nonnull )?}}@vf1_2, ...)

; CHECK: define internal void @branch_funnel(i8*

declare i1 @llvm.type.test(i8*, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
!3 = !{i32 0, !4}
!4 = distinct !{}

attributes #0 = { "target-features"="+retpoline" }
