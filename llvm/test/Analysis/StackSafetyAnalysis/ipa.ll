; REQUIRES: aarch64-registered-target

; RUN: llvm-as %s -o %t0.bc
; RUN: llvm-as %S/Inputs/ipa.ll -o %t1.bc
; RUN: llvm-link -disable-lazy-loading %t0.bc %t1.bc -o %t.combined.bc

; RUN: opt -S -analyze -stack-safety-local %t.combined.bc | FileCheck %s --check-prefixes=CHECK,LOCAL
; RUN: opt -S -passes="print<stack-safety-local>" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL

; RUN: opt -S -analyze -stack-safety %t.combined.bc | FileCheck %s --check-prefixes=CHECK,GLOBAL,NOLTO
; RUN: opt -S -passes="print-stack-safety" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,NOLTO

; Do an end-to-test using the new LTO API
; TODO: Hideous llvm-lto2 invocation, add a --default-symbol-resolution to llvm-lto2?
; RUN: opt -module-summary %s -o %t.summ0.bc
; RUN: opt -module-summary %S/Inputs/ipa.ll -o %t.summ1.bc

; RUN: llvm-lto2 run %t.summ0.bc %t.summ1.bc -o %t.lto -stack-safety-print -stack-safety-run -save-temps -thinlto-threads 1 -O0 \
; RUN:  -r %t.summ0.bc,ExternalCall, \
; RUN:  -r %t.summ0.bc,f1,px \
; RUN:  -r %t.summ0.bc,f2,px \
; RUN:  -r %t.summ0.bc,f3,px \
; RUN:  -r %t.summ0.bc,f4,px \
; RUN:  -r %t.summ0.bc,f5,px \
; RUN:  -r %t.summ0.bc,f6,px \
; RUN:  -r %t.summ0.bc,f7,px \
; RUN:  -r %t.summ0.bc,f8left,px \
; RUN:  -r %t.summ0.bc,f8oobleft,px \
; RUN:  -r %t.summ0.bc,f8oobright,px \
; RUN:  -r %t.summ0.bc,f8right,px \
; RUN:  -r %t.summ0.bc,InterposableCall,px \
; RUN:  -r %t.summ0.bc,InterposableWrite1, \
; RUN:  -r %t.summ0.bc,PreemptableCall,px \
; RUN:  -r %t.summ0.bc,PreemptableWrite1, \
; RUN:  -r %t.summ0.bc,PrivateCall,px \
; RUN:  -r %t.summ0.bc,Rec2, \
; RUN:  -r %t.summ0.bc,RecursiveNoOffset, \
; RUN:  -r %t.summ0.bc,RecursiveWithOffset, \
; RUN:  -r %t.summ0.bc,ReturnDependent, \
; RUN:  -r %t.summ0.bc,TestCrossModuleConflict,px \
; RUN:  -r %t.summ0.bc,TestCrossModuleOnce,px \
; RUN:  -r %t.summ0.bc,TestCrossModuleTwice,px \
; RUN:  -r %t.summ0.bc,TestCrossModuleWeak,px \
; RUN:  -r %t.summ0.bc,TestRecursiveNoOffset,px \
; RUN:  -r %t.summ0.bc,TestRecursiveWithOffset,px \
; RUN:  -r %t.summ0.bc,TestUpdateArg,px \
; RUN:  -r %t.summ0.bc,TwoArguments,px \
; RUN:  -r %t.summ0.bc,TwoArgumentsOOBBoth,px \
; RUN:  -r %t.summ0.bc,TwoArgumentsOOBOne,px \
; RUN:  -r %t.summ0.bc,TwoArgumentsOOBOther,px \
; RUN:  -r %t.summ0.bc,Weak,x \
; RUN:  -r %t.summ0.bc,Write1, \
; RUN:  -r %t.summ0.bc,Write1DiffModule,x \
; RUN:  -r %t.summ0.bc,Write1Module0,px \
; RUN:  -r %t.summ0.bc,Write1Private,x \
; RUN:  -r %t.summ0.bc,Write1SameModule,x \
; RUN:  -r %t.summ0.bc,Write1Weak,x \
; RUN:  -r %t.summ0.bc,Write4_2, \
; RUN:  -r %t.summ0.bc,Write4, \
; RUN:  -r %t.summ0.bc,Write8, \
; RUN:  -r %t.summ0.bc,WriteAndReturn8, \
; RUN:  -r %t.summ1.bc,ExternalCall,px \
; RUN:  -r %t.summ1.bc,InterposableWrite1,px \
; RUN:  -r %t.summ1.bc,PreemptableWrite1,px \
; RUN:  -r %t.summ1.bc,Rec0,px \
; RUN:  -r %t.summ1.bc,Rec1,px \
; RUN:  -r %t.summ1.bc,Rec2,px \
; RUN:  -r %t.summ1.bc,RecursiveNoOffset,px \
; RUN:  -r %t.summ1.bc,RecursiveWithOffset,px \
; RUN:  -r %t.summ1.bc,ReturnAlloca,px \
; RUN:  -r %t.summ1.bc,ReturnDependent,px \
; RUN:  -r %t.summ1.bc,Weak,x \
; RUN:  -r %t.summ1.bc,Write1,px \
; RUN:  -r %t.summ1.bc,Write1DiffModule,px \
; RUN:  -r %t.summ1.bc,Write1Module0,x \
; RUN:  -r %t.summ1.bc,Write1Private,px \
; RUN:  -r %t.summ1.bc,Write1SameModule,px \
; RUN:  -r %t.summ1.bc,Write1Weak,px \
; RUN:  -r %t.summ1.bc,Write4_2,px \
; RUN:  -r %t.summ1.bc,Write4,px \
; RUN:  -r %t.summ1.bc,Write8,px \
; RUN:  -r %t.summ1.bc,WriteAndReturn8,px \
; RUN:    2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,LTO

; RUN: llvm-dis %t.lto.index.bc -o - | FileCheck --check-prefix=INDEX %s

; RUN: llvm-lto2 run %t.summ0.bc %t.summ1.bc -o %t-newpm.lto -use-new-pm -stack-safety-print -stack-safety-run -save-temps -thinlto-threads 1 -O0 \
; RUN:  -r %t.summ0.bc,ExternalCall, \
; RUN:  -r %t.summ0.bc,f1,px \
; RUN:  -r %t.summ0.bc,f2,px \
; RUN:  -r %t.summ0.bc,f3,px \
; RUN:  -r %t.summ0.bc,f4,px \
; RUN:  -r %t.summ0.bc,f5,px \
; RUN:  -r %t.summ0.bc,f6,px \
; RUN:  -r %t.summ0.bc,f7,px \
; RUN:  -r %t.summ0.bc,f8left,px \
; RUN:  -r %t.summ0.bc,f8oobleft,px \
; RUN:  -r %t.summ0.bc,f8oobright,px \
; RUN:  -r %t.summ0.bc,f8right,px \
; RUN:  -r %t.summ0.bc,InterposableCall,px \
; RUN:  -r %t.summ0.bc,InterposableWrite1, \
; RUN:  -r %t.summ0.bc,PreemptableCall,px \
; RUN:  -r %t.summ0.bc,PreemptableWrite1, \
; RUN:  -r %t.summ0.bc,PrivateCall,px \
; RUN:  -r %t.summ0.bc,Rec2, \
; RUN:  -r %t.summ0.bc,RecursiveNoOffset, \
; RUN:  -r %t.summ0.bc,RecursiveWithOffset, \
; RUN:  -r %t.summ0.bc,ReturnDependent, \
; RUN:  -r %t.summ0.bc,TestCrossModuleConflict,px \
; RUN:  -r %t.summ0.bc,TestCrossModuleOnce,px \
; RUN:  -r %t.summ0.bc,TestCrossModuleTwice,px \
; RUN:  -r %t.summ0.bc,TestCrossModuleWeak,px \
; RUN:  -r %t.summ0.bc,TestRecursiveNoOffset,px \
; RUN:  -r %t.summ0.bc,TestRecursiveWithOffset,px \
; RUN:  -r %t.summ0.bc,TestUpdateArg,px \
; RUN:  -r %t.summ0.bc,TwoArguments,px \
; RUN:  -r %t.summ0.bc,TwoArgumentsOOBBoth,px \
; RUN:  -r %t.summ0.bc,TwoArgumentsOOBOne,px \
; RUN:  -r %t.summ0.bc,TwoArgumentsOOBOther,px \
; RUN:  -r %t.summ0.bc,Weak,x \
; RUN:  -r %t.summ0.bc,Write1, \
; RUN:  -r %t.summ0.bc,Write1DiffModule,x \
; RUN:  -r %t.summ0.bc,Write1Module0,px \
; RUN:  -r %t.summ0.bc,Write1Private,x \
; RUN:  -r %t.summ0.bc,Write1SameModule,x \
; RUN:  -r %t.summ0.bc,Write1Weak,x \
; RUN:  -r %t.summ0.bc,Write4_2, \
; RUN:  -r %t.summ0.bc,Write4, \
; RUN:  -r %t.summ0.bc,Write8, \
; RUN:  -r %t.summ0.bc,WriteAndReturn8, \
; RUN:  -r %t.summ1.bc,ExternalCall,px \
; RUN:  -r %t.summ1.bc,InterposableWrite1,px \
; RUN:  -r %t.summ1.bc,PreemptableWrite1,px \
; RUN:  -r %t.summ1.bc,Rec0,px \
; RUN:  -r %t.summ1.bc,Rec1,px \
; RUN:  -r %t.summ1.bc,Rec2,px \
; RUN:  -r %t.summ1.bc,RecursiveNoOffset,px \
; RUN:  -r %t.summ1.bc,RecursiveWithOffset,px \
; RUN:  -r %t.summ1.bc,ReturnAlloca,px \
; RUN:  -r %t.summ1.bc,ReturnDependent,px \
; RUN:  -r %t.summ1.bc,Weak,x \
; RUN:  -r %t.summ1.bc,Write1,px \
; RUN:  -r %t.summ1.bc,Write1DiffModule,px \
; RUN:  -r %t.summ1.bc,Write1Module0,x \
; RUN:  -r %t.summ1.bc,Write1Private,px \
; RUN:  -r %t.summ1.bc,Write1SameModule,px \
; RUN:  -r %t.summ1.bc,Write1Weak,px \
; RUN:  -r %t.summ1.bc,Write4_2,px \
; RUN:  -r %t.summ1.bc,Write4,px \
; RUN:  -r %t.summ1.bc,Write8,px \
; RUN:  -r %t.summ1.bc,WriteAndReturn8,px \
; RUN:    2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,LTO

; RUN: llvm-dis %t-newpm.lto.index.bc -o - | FileCheck --check-prefix=INDEX %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

declare void @Write1(i8* %p)
declare void @Write4(i8* %p)
declare void @Write4_2(i8* %p, i8* %q)
declare void @Write8(i8* %p)
declare dso_local i8* @WriteAndReturn8(i8* %p)
declare dso_local void @ExternalCall(i8* %p)
declare void @PreemptableWrite1(i8* %p)
declare void @InterposableWrite1(i8* %p)
declare i8* @ReturnDependent(i8* %p)
declare void @Rec2(i8* %p)
declare void @RecursiveNoOffset(i32* %p, i32 %size, i32* %acc)
declare void @RecursiveWithOffset(i32 %size, i32* %acc)
declare void @Write1SameModule(i8* %p)
declare void @Write1DiffModule(i8* %p)
declare void @Write1Private(i8* %p)
declare void @Write1Weak(i8* %p)

; Basic out-of-bounds.
define void @f1() #0 {
; CHECK-LABEL: @f1 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write8(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,8), @Write8(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @Write8(i8* %x1)
  ret void
}

; Basic in-bounds.
define void @f2() #0 {
; CHECK-LABEL: @f2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,1), @Write1(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @Write1(i8* %x1)
  ret void
}

; Another basic in-bounds.
define void @f3() #0 {
; CHECK-LABEL: @f3 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write4(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,4), @Write4(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @Write4(i8* %x1)
  ret void
}

; In-bounds with offset.
define void @f4() #0 {
; CHECK-LABEL: @f4 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write1(arg0, [1,2)){{$}}
; GLOBAL-NEXT: x[4]: [1,2), @Write1(arg0, [1,2)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 1
  call void @Write1(i8* %x2)
  ret void
}

; Out-of-bounds with offset.
define void @f5() #0 {
; CHECK-LABEL: @f5 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: empty-set, @Write4(arg0, [1,2)){{$}}
; GLOBAL-NEXT: [1,5), @Write4(arg0, [1,2)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 1
  call void @Write4(i8* %x2)
  ret void
}

; External call.
define void @f6() #0 {
; CHECK-LABEL: @f6 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @ExternalCall(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: full-set, @ExternalCall(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @ExternalCall(i8* %x1)
  ret void
}

; Call to dso_preemptable function
define void @PreemptableCall() #0 {
; CHECK-LABEL: @PreemptableCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @PreemptableWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: full-set, @PreemptableWrite1(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @PreemptableWrite1(i8* %x1)
  ret void
}

; Call to function with interposable linkage
define void @InterposableCall() #0 {
; CHECK-LABEL: @InterposableCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @InterposableWrite1(arg0, [0,1)){{$}}
; NOLTO-NEXT: x[4]: full-set, @InterposableWrite1(arg0, [0,1)){{$}}
; LTO-NEXT: x[4]: [0,1), @InterposableWrite1(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @InterposableWrite1(i8* %x1)
  ret void
}

; Call to function with private linkage
define void @PrivateCall() #0 {
; CHECK-LABEL: @PrivateCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @PrivateWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,1), @PrivateWrite1(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  call void @PrivateWrite1(i8* %x1)
  ret void
}

define private void @PrivateWrite1(i8* %p) #0 {
; CHECK-LABEL: @PrivateWrite1{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:
entry:
  store i8 0, i8* %p, align 1
  ret void
}

; Caller returns a dependent value.
; FIXME: alloca considered unsafe even if the return value is unused.
define void @f7() #0 {
; CHECK-LABEL: @f7 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @ReturnDependent(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: full-set, @ReturnDependent(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x1 = bitcast i32* %x to i8*
  %x2 = call i8* @ReturnDependent(i8* %x1)
  ret void
}

define void @f8left() #0 {
; CHECK-LABEL: @f8left dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [2,3)){{$}}
; GLOBAL-NEXT: x[8]: [0,4), @Rec2(arg0, [2,3)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 2
; 2 + [-2, 2) = [0, 4) => OK
  call void @Rec2(i8* %x2)
  ret void
}

define void @f8right() #0 {
; CHECK-LABEL: @f8right dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [6,7)){{$}}
; GLOBAL-NEXT: x[8]: [4,8), @Rec2(arg0, [6,7)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 6
; 6 + [-2, 2) = [4, 8) => OK
  call void @Rec2(i8* %x2)
  ret void
}

define void @f8oobleft() #0 {
; CHECK-LABEL: @f8oobleft dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [1,2)){{$}}
; GLOBAL-NEXT: x[8]: [-1,3), @Rec2(arg0, [1,2)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 1
; 1 + [-2, 2) = [-1, 3) => NOT OK
  call void @Rec2(i8* %x2)
  ret void
}

define void @f8oobright() #0 {
; CHECK-LABEL: @f8oobright dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [7,8)){{$}}
; GLOBAL-NEXT: x[8]: [5,9), @Rec2(arg0, [7,8)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 7
; 7 + [-2, 2) = [5, 9) => NOT OK
  call void @Rec2(i8* %x2)
  ret void
}

define void @TwoArguments() #0 {
; CHECK-LABEL: @TwoArguments dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg1, [0,1)), @Write4_2(arg0, [4,5)){{$}}
; GLOBAL-NEXT: x[8]: [0,8), @Write4_2(arg1, [0,1)), @Write4_2(arg0, [4,5)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 4
  call void @Write4_2(i8* %x2, i8* %x1)
  ret void
}

define void @TwoArgumentsOOBOne() #0 {
; CHECK-LABEL: @TwoArgumentsOOBOne dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg1, [0,1)), @Write4_2(arg0, [5,6)){{$}}
; GLOBAL-NEXT: x[8]: [0,9), @Write4_2(arg1, [0,1)), @Write4_2(arg0, [5,6)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = bitcast i64* %x to i8*
  %x2 = getelementptr i8, i8* %x1, i64 5
  call void @Write4_2(i8* %x2, i8* %x1)
  ret void
}

define void @TwoArgumentsOOBOther() #0 {
; CHECK-LABEL: @TwoArgumentsOOBOther dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg1, [-1,0)), @Write4_2(arg0, [4,5)){{$}}
; GLOBAL-NEXT: x[8]: [-1,8), @Write4_2(arg1, [-1,0)), @Write4_2(arg0, [4,5)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x0 = bitcast i64* %x to i8*
  %x1 = getelementptr i8, i8* %x0, i64 -1
  %x2 = getelementptr i8, i8* %x0, i64 4
  call void @Write4_2(i8* %x2, i8* %x1)
  ret void
}

define void @TwoArgumentsOOBBoth() #0 {
; CHECK-LABEL: @TwoArgumentsOOBBoth dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg1, [-1,0)), @Write4_2(arg0, [5,6)){{$}}
; GLOBAL-NEXT: x[8]: [-1,9), @Write4_2(arg1, [-1,0)), @Write4_2(arg0, [5,6)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x0 = bitcast i64* %x to i8*
  %x1 = getelementptr i8, i8* %x0, i64 -1
  %x2 = getelementptr i8, i8* %x0, i64 5
  call void @Write4_2(i8* %x2, i8* %x1)
  ret void
}

define i32 @TestRecursiveNoOffset(i32* %p, i32 %size) #0 {
; CHECK-LABEL: @TestRecursiveNoOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @RecursiveNoOffset(arg0, [0,1)){{$}}
; GLOBAL-NEXT: p[]: full-set, @RecursiveNoOffset(arg0, [0,1)){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: sum[4]: [0,4), @RecursiveNoOffset(arg2, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %sum = alloca i32, align 4
  %0 = bitcast i32* %sum to i8*
  store i32 0, i32* %sum, align 4
  call void @RecursiveNoOffset(i32* %p, i32 %size, i32* %sum)
  %1 = load i32, i32* %sum, align 4
  ret i32 %1
}

define void @TestRecursiveWithOffset(i32 %size) #0 {
; CHECK-LABEL: @TestRecursiveWithOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: sum[64]: empty-set, @RecursiveWithOffset(arg1, [0,1)){{$}}
; GLOBAL-NEXT: sum[64]: full-set, @RecursiveWithOffset(arg1, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %sum = alloca i32, i64 16, align 4
  call void @RecursiveWithOffset(i32 %size, i32* %sum)
  ret void
}

; FIXME: IPA should detect that access is safe
define void @TestUpdateArg() #0 {
; CHECK-LABEL: @TestUpdateArg dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[16]: empty-set, @WriteAndReturn8(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[16]: full-set, @WriteAndReturn8(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, i64 16, align 4
  %0 = call i8* @WriteAndReturn8(i8* %x)
  ret void
}

define void @TestCrossModuleOnce() #0 {
; CHECK-DAG: @TestCrossModuleOnce dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: y[1]: empty-set, @Write1SameModule(arg0, [0,1)){{$}}
; GLOBAL-NEXT: y[1]: [0,1), @Write1SameModule(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %y = alloca i8, align 4
  call void @Write1SameModule(i8* %y)
  ret void
}

; FIXME: LTO should match NOLTO
define void @TestCrossModuleTwice() #0 {
; CHECK-DAG: @TestCrossModuleTwice dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: z[1]: empty-set, @Write1DiffModule(arg0, [0,1)){{$}}
; NOLTO-NEXT: z[1]: [0,1), @Write1DiffModule(arg0, [0,1)){{$}}
; LTO-NEXT: z[1]: full-set, @Write1DiffModule(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %z = alloca i8, align 4
  call void @Write1DiffModule(i8* %z)
  ret void
}

define void @TestCrossModuleConflict() #0 {
; CHECK-DAG: @TestCrossModuleConflict dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @Write1Private(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[1]: [-1,0), @Write1Private(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  call void @Write1Private(i8* %x)
  ret void
}

; FIXME: LTO should match NOLTO
define void @TestCrossModuleWeak() #0 {
; CHECK-DAG: @TestCrossModuleWeak dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @Write1Weak(arg0, [0,1)){{$}}
; NOLTO-NEXT: x[1]: [1,2), @Write1Weak(arg0, [0,1)){{$}}
; LTO-NEXT: x[1]: full-set, @Write1Weak(arg0, [0,1)){{$}}
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  call void @Write1Weak(i8* %x)
  ret void
}

define private dso_local void @Private(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 1
  store i8 0, i8* %p1, align 1
  ret void
}

define dso_local void @Write1Module0(i8* %p) #0 {
entry:
  store i8 0, i8* %p, align 1
  ret void
}

define dso_local void @Weak(i8* %p) #0 {
entry:
  %p1 = getelementptr i8, i8* %p, i64 1
  store i8 0, i8* %p1, align 1
  ret void
}

; The rest is from Inputs/ipa.ll

; CHECK-LABEL: @Write1{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @Write4{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @Write4_2{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: q[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @Write8{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,8){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @WriteAndReturn8{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @PreemptableWrite1 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @InterposableWrite1 interposable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @ReturnDependent{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @Rec0{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @Write4(arg0, [2,3)){{$}}
; GLOBAL-NEXT: p[]: [2,6)
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @Rec1{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @Rec0(arg0, [1,2)){{$}}
; GLOBAL-NEXT: p[]: [3,7)
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @Rec2{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @Rec1(arg0, [-5,-4)){{$}}
; GLOBAL-NEXT: p[]: [-2,2)
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @RecursiveNoOffset{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: [0,4), @RecursiveNoOffset(arg0, [4,5)){{$}}
; GLOBAL-NEXT: p[]: full-set, @RecursiveNoOffset(arg0, [4,5)){{$}}
; CHECK-NEXT: acc[]: [0,4), @RecursiveNoOffset(arg2, [0,1)){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @RecursiveWithOffset{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: acc[]: [0,4), @RecursiveWithOffset(arg1, [4,5)){{$}}
; GLOBAL-NEXT: acc[]: full-set, @RecursiveWithOffset(arg1, [4,5)){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-EMPTY:

; CHECK-LABEL: @ReturnAlloca
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: full-set
; CHECK-EMPTY:


; INDEX: ^0 = module:
; INDEX-NEXT: ^1 = module:
; INDEX-NEXT: ^2 = gv: (guid: 357037859923812466, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}))))
; INDEX-NEXT: ^3 = gv: (guid: 402261637236519836, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [1, 1])))))
; INDEX-NEXT: ^4 = gv: (guid: 413541695569076425, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 6,  funcFlags: ({{[^)]+}}), calls: ((callee: ^18)))))
; INDEX-NEXT: ^5 = gv: (guid: 583675441393868004, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^11)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^11, param: 0, offset: [2, 2])))))))
; INDEX-NEXT: ^6 = gv: (guid: 1136387433625471221, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^32)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^32, param: 0, offset: [-5, -5])))))))
; INDEX-NEXT: ^7 = gv: (guid: 2072045998141807037, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^14)))))
; INDEX-NEXT: ^8 = gv: (guid: 2275681413198219603, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 0])))))
; INDEX-NEXT: ^9 = gv: (guid: 2497238192574115968, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 6,  funcFlags: ({{[^)]+}}), calls: ((callee: ^34)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^34, param: 0, offset: [0, 0])))))))
; INDEX-NEXT: ^10 = gv: (guid: 2532896148796215206, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^6)))))
; INDEX-NEXT: ^11 = gv: (guid: 2688347355436776808, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 3])))))
; INDEX-NEXT: ^12 = gv: (guid: 2998357024553461356, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^2)))))
; INDEX-NEXT: ^13 = gv: (guid: 3063826769604044178, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), calls: ((callee: ^39)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^39, param: 0, offset: [0, 0])))))))
; INDEX-NEXT: ^14 = gv: (guid: 3498046014991828871, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 7])))))
; INDEX-NEXT: ^15 = gv: (guid: 3524937277209782828, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^18)))))
; INDEX-NEXT: ^16 = gv: (guid: 4197650231481825559, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^11)))))
; INDEX-NEXT: ^17 = gv: (guid: 4936416805851585180, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^6)))))
; INDEX-NEXT: ^18 = gv: (guid: 5426356728050253384, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 3]), (param: 1, offset: [0, 3])))))
; INDEX-NEXT: ^19 = gv: (guid: 5775385244688528345, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 8, calls: ((callee: ^19)), params: ((param: 1, offset: [0, 3], calls: ((callee: ^19, param: 1, offset: [4, 4])))))))
; INDEX-NEXT: ^20 = gv: (guid: 5825695806885405811, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [1, 1]))), function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [-1, -1])))))
; INDEX-NEXT: ^21 = gv: (guid: 6195939829073041205, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), calls: ((callee: ^31)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^31, param: 0, offset: [0, 0])))))))
; INDEX-NEXT: ^22 = gv: (guid: 6392508476582107374, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^44)))))
; INDEX-NEXT: ^23 = gv: (guid: 7664681152647503879, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^48)))))
; INDEX-NEXT: ^24 = gv: (guid: 7878145294651437475, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^38)))))
; INDEX-NEXT: ^25 = gv: (guid: 8471399308421654326, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^39)))))
; INDEX-NEXT: ^26 = gv: (guid: 8866541111220081334, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^8)))))
; INDEX-NEXT: ^27 = gv: (guid: 9330565726709681411, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^19)))))
; INDEX-NEXT: ^28 = gv: (guid: 9608616742292190323, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^6)))))
; INDEX-NEXT: ^29 = gv: (guid: 9977445152485795981, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 0])))))
; INDEX-NEXT: ^30 = gv: (guid: 10064745020953272174, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^39)))))
; INDEX-NEXT: ^31 = gv: (guid: 10228479262507912607, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [-1, -1])))))
; INDEX-NEXT: ^32 = gv: (guid: 10622720612762034607, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^5)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^5, param: 0, offset: [1, 1])))))))
; INDEX-NEXT: ^33 = gv: (guid: 10879331213637669871, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^21)))))
; INDEX-NEXT: ^34 = gv: (guid: 11254704694495916625, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 11, calls: ((callee: ^34)), params: ((param: 0, offset: [0, 3], calls: ((callee: ^34, param: 0, offset: [4, 4]))), (param: 2, offset: [0, 3], calls: ((callee: ^34, param: 2, offset: [0, 0])))))))
; INDEX-NEXT: ^35 = gv: (guid: 11349614871118095988, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 6,  funcFlags: ({{[^)]+}}), calls: ((callee: ^18)))))
; INDEX-NEXT: ^36 = gv: (guid: 11686717102184386164, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^11)))))
; INDEX-NEXT: ^37 = gv: (guid: 11834966808443348068, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}))))
; INDEX-NEXT: ^38 = gv: (guid: 13174833224364694040, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), calls: ((callee: ^20)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^20, param: 0, offset: [0, 0])))))))
; INDEX-NEXT: ^39 = gv: (guid: 14731732627017339067, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 0])))))
; INDEX-NEXT: ^40 = gv: (guid: 15140994247518074144, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 0])))))
; INDEX-NEXT: ^41 = gv: (guid: 15185358799878470431, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^6)))))
; INDEX-NEXT: ^42 = gv: (guid: 15238722921051206629, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^29)))))
; INDEX-NEXT: ^43 = gv: (guid: 15982699239429404956, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 3,  funcFlags: ({{[^)]+}}), calls: ((callee: ^13)))))
; INDEX-NEXT: ^44 = gv: (guid: 16066523084744374939, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}))))
; INDEX-NEXT: ^45 = gv: (guid: 16200396372539015164, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 5,  funcFlags: ({{[^)]+}}), calls: ((callee: ^18)))))
; INDEX-NEXT: ^46 = gv: (guid: 16531634002570146185, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), params: ((param: 0, offset: [0, 0])))))
; INDEX-NEXT: ^47 = gv: (guid: 16555643409459919887, summaries: (function: (module: ^0,  flags: ({{[^)]+}}), insts: 4,  funcFlags: ({{[^)]+}}), calls: ((callee: ^40)))))
; INDEX-NEXT: ^48 = gv: (guid: 16869210845156067648, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2,  funcFlags: ({{[^)]+}}), calls: ((callee: ^46)), params: ((param: 0, offset: [0, -1], calls: ((callee: ^46, param: 0, offset: [0, 0])))))))
; INDEX-NEXT: ^49 = gv: (guid: 18021476350828399578, summaries: (function: (module: ^1,  flags: ({{[^)]+}}), insts: 2)))
; INDEX-NEXT: ^50 = flags: 1
