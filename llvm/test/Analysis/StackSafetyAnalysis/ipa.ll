; REQUIRES: aarch64-registered-target

; RUN: llvm-as %s -o %t0.bc
; RUN: llvm-as %S/Inputs/ipa.ll -o %t1.bc
; RUN: llvm-link -disable-lazy-loading %t0.bc %t1.bc -o %t.combined.bc

; RUN: opt -S -passes="print<stack-safety-local>" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,LOCAL

; RUN: opt -S -passes="print-stack-safety" -disable-output %t.combined.bc 2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,NOLTO

; Do an end-to-test using the new LTO API
; TODO: Hideous llvm-lto2 invocation, add a --default-symbol-resolution to llvm-lto2?
; RUN: opt -module-summary %s -o %t.summ0.bc
; RUN: opt -module-summary %S/Inputs/ipa.ll -o %t.summ1.bc

; RUN: llvm-dis %t.summ0.bc -o - > %t.ids.txt
; RUN: llvm-dis %t.summ1.bc -o - >> %t.ids.txt

; DEFINE: %{res} = \
; DEFINE:  -r %t.summ0.bc,ExternalCall, \
; DEFINE:  -r %t.summ0.bc,f1,px \
; DEFINE:  -r %t.summ0.bc,f2,px \
; DEFINE:  -r %t.summ0.bc,f3,px \
; DEFINE:  -r %t.summ0.bc,f4,px \
; DEFINE:  -r %t.summ0.bc,f5,px \
; DEFINE:  -r %t.summ0.bc,f6,px \
; DEFINE:  -r %t.summ0.bc,f7,px \
; DEFINE:  -r %t.summ0.bc,f8left,px \
; DEFINE:  -r %t.summ0.bc,f8oobleft,px \
; DEFINE:  -r %t.summ0.bc,f8oobright,px \
; DEFINE:  -r %t.summ0.bc,f8right,px \
; DEFINE:  -r %t.summ0.bc,InterposableCall,px \
; DEFINE:  -r %t.summ0.bc,InterposableWrite1, \
; DEFINE:  -r %t.summ0.bc,PreemptableCall,px \
; DEFINE:  -r %t.summ0.bc,PreemptableWrite1, \
; DEFINE:  -r %t.summ0.bc,PrivateCall,px \
; DEFINE:  -r %t.summ0.bc,Rec2, \
; DEFINE:  -r %t.summ0.bc,RecursiveNoOffset, \
; DEFINE:  -r %t.summ0.bc,RecursiveWithOffset, \
; DEFINE:  -r %t.summ0.bc,ReturnDependent, \
; DEFINE:  -r %t.summ0.bc,TestCrossModuleConflict,px \
; DEFINE:  -r %t.summ0.bc,TestCrossModuleOnce,px \
; DEFINE:  -r %t.summ0.bc,TestCrossModuleTwice,px \
; DEFINE:  -r %t.summ0.bc,TestCrossModuleWeak,px \
; DEFINE:  -r %t.summ0.bc,TestRecursiveNoOffset,px \
; DEFINE:  -r %t.summ0.bc,TestRecursiveWithOffset,px \
; DEFINE:  -r %t.summ0.bc,TestUpdateArg,px \
; DEFINE:  -r %t.summ0.bc,TwoArguments,px \
; DEFINE:  -r %t.summ0.bc,TwoArgumentsOOBBoth,px \
; DEFINE:  -r %t.summ0.bc,TwoArgumentsOOBOne,px \
; DEFINE:  -r %t.summ0.bc,TwoArgumentsOOBOther,px \
; DEFINE:  -r %t.summ0.bc,Weak,x \
; DEFINE:  -r %t.summ0.bc,Write1, \
; DEFINE:  -r %t.summ0.bc,Write1DiffModule,x \
; DEFINE:  -r %t.summ0.bc,Write1Module0,px \
; DEFINE:  -r %t.summ0.bc,Write1Private,x \
; DEFINE:  -r %t.summ0.bc,Write1SameModule,x \
; DEFINE:  -r %t.summ0.bc,Write1Weak,x \
; DEFINE:  -r %t.summ0.bc,Write4_2, \
; DEFINE:  -r %t.summ0.bc,Write4, \
; DEFINE:  -r %t.summ0.bc,Write8, \
; DEFINE:  -r %t.summ0.bc,WriteAndReturn8, \
; DEFINE:  -r %t.summ1.bc,ExternalCall,px \
; DEFINE:  -r %t.summ1.bc,InterposableWrite1,px \
; DEFINE:  -r %t.summ1.bc,PreemptableWrite1,px \
; DEFINE:  -r %t.summ1.bc,Rec0,px \
; DEFINE:  -r %t.summ1.bc,Rec1,px \
; DEFINE:  -r %t.summ1.bc,Rec2,px \
; DEFINE:  -r %t.summ1.bc,RecursiveNoOffset,px \
; DEFINE:  -r %t.summ1.bc,RecursiveWithOffset,px \
; DEFINE:  -r %t.summ1.bc,ReturnAlloca,px \
; DEFINE:  -r %t.summ1.bc,ReturnDependent,px \
; DEFINE:  -r %t.summ1.bc,Weak,x \
; DEFINE:  -r %t.summ1.bc,Write1,px \
; DEFINE:  -r %t.summ1.bc,Write1DiffModule,px \
; DEFINE:  -r %t.summ1.bc,Write1Module0,x \
; DEFINE:  -r %t.summ1.bc,Write1Private,px \
; DEFINE:  -r %t.summ1.bc,Write1SameModule,px \
; DEFINE:  -r %t.summ1.bc,Write1Weak,px \
; DEFINE:  -r %t.summ1.bc,Write4_2,px \
; DEFINE:  -r %t.summ1.bc,Write4,px \
; DEFINE:  -r %t.summ1.bc,Write8,px \
; DEFINE:  -r %t.summ1.bc,WriteAndReturn8,px

; RUN: llvm-lto2 run %t.summ0.bc %t.summ1.bc -o %t.lto -stack-safety-print -stack-safety-run -save-temps -thinlto-threads 1 -O0 \
; RUN:  %{res} \
; RUN:    2>&1 | FileCheck %s --check-prefixes=CHECK,GLOBAL,LTO

; RUN: llvm-lto2 run %t.summ0.bc %t.summ1.bc -o %t.lto -stack-safety-run -thinlto-distributed-indexes -thinlto-threads 1 -O0 %{res}
; RUN: llvm-dis %t.summ1.bc.thinlto.bc -o - >> %t.ids.txt
; RUN: FileCheck --check-prefixes=INDEX %s < %t.ids.txt

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux"

attributes #0 = { noinline sanitize_memtag "target-features"="+mte,+neon" }

declare void @Write1(ptr %p)
declare void @Write4(ptr %p)
declare void @Write4_2(ptr %p, ptr %q)
declare void @Write8(ptr %p)
declare dso_local ptr @WriteAndReturn8(ptr %p)
declare dso_local void @ExternalCall(ptr %p)
declare void @PreemptableWrite1(ptr %p)
declare void @InterposableWrite1(ptr %p)
declare ptr @ReturnDependent(ptr %p)
declare void @Rec2(ptr %p)
declare void @RecursiveNoOffset(ptr %p, i32 %size, ptr %acc)
declare void @RecursiveWithOffset(i32 %size, ptr %acc)
declare void @Write1SameModule(ptr %p)
declare void @Write1DiffModule(ptr %p)
declare void @Write1Private(ptr %p)
declare void @Write1Weak(ptr %p)

; Basic out-of-bounds.
define void @f1() #0 {
; CHECK-LABEL: @f1 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write8(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,8), @Write8(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @Write8(ptr %x)
  ret void
}

; Basic in-bounds.
define void @f2() #0 {
; CHECK-LABEL: @f2 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,1), @Write1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @Write1(ptr %x)
  ret void
}

; Another basic in-bounds.
define void @f3() #0 {
; CHECK-LABEL: @f3 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write4(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,4), @Write4(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @Write4(ptr %x)
  ret void
}

; In-bounds with offset.
define void @f4() #0 {
; CHECK-LABEL: @f4 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @Write1(arg0, [1,2)){{$}}
; GLOBAL-NEXT: x[4]: [1,2), @Write1(arg0, [1,2)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 1
  call void @Write1(ptr %x2)
  ret void
}

; Out-of-bounds with offset.
define void @f5() #0 {
; CHECK-LABEL: @f5 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: empty-set, @Write4(arg0, [1,2)){{$}}
; GLOBAL-NEXT: [1,5), @Write4(arg0, [1,2)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = getelementptr i8, ptr %x, i64 1
  call void @Write4(ptr %x2)
  ret void
}

; External call.
define void @f6() #0 {
; CHECK-LABEL: @f6 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @ExternalCall(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: full-set, @ExternalCall(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @ExternalCall(ptr %x)
  ret void
}

; Call to dso_preemptable function
define void @PreemptableCall() #0 {
; CHECK-LABEL: @PreemptableCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @PreemptableWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: full-set, @PreemptableWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @PreemptableWrite1(ptr %x)
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
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @InterposableWrite1(ptr %x)
  ret void
}

; Call to function with private linkage
define void @PrivateCall() #0 {
; CHECK-LABEL: @PrivateCall dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[4]: empty-set, @PrivateWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[4]: [0,1), @PrivateWrite1(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  call void @PrivateWrite1(ptr %x)
  ret void
}

define private void @PrivateWrite1(ptr %p) #0 {
; CHECK-LABEL: @PrivateWrite1{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %p, align 1
; CHECK-EMPTY:
entry:
  store i8 0, ptr %p, align 1
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
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i32, align 4
  %x2 = call ptr @ReturnDependent(ptr %x)
  ret void
}

define void @f8left() #0 {
; CHECK-LABEL: @f8left dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [2,3)){{$}}
; GLOBAL-NEXT: x[8]: [0,4), @Rec2(arg0, [2,3)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 2
; 2 + [-2, 2) = [0, 4) => OK
  call void @Rec2(ptr %x2)
  ret void
}

define void @f8right() #0 {
; CHECK-LABEL: @f8right dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [6,7)){{$}}
; GLOBAL-NEXT: x[8]: [4,8), @Rec2(arg0, [6,7)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 6
; 6 + [-2, 2) = [4, 8) => OK
  call void @Rec2(ptr %x2)
  ret void
}

define void @f8oobleft() #0 {
; CHECK-LABEL: @f8oobleft dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [1,2)){{$}}
; GLOBAL-NEXT: x[8]: [-1,3), @Rec2(arg0, [1,2)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 1
; 1 + [-2, 2) = [-1, 3) => NOT OK
  call void @Rec2(ptr %x2)
  ret void
}

define void @f8oobright() #0 {
; CHECK-LABEL: @f8oobright dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Rec2(arg0, [7,8)){{$}}
; GLOBAL-NEXT: x[8]: [5,9), @Rec2(arg0, [7,8)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 7
; 7 + [-2, 2) = [5, 9) => NOT OK
  call void @Rec2(ptr %x2)
  ret void
}

define void @TwoArguments() #0 {
; CHECK-LABEL: @TwoArguments dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg0, [4,5)), @Write4_2(arg1, [0,1)){{$}}
; GLOBAL-NEXT: x[8]: [0,8), @Write4_2(arg0, [4,5)), @Write4_2(arg1, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 4
  call void @Write4_2(ptr %x2, ptr %x)
  ret void
}

define void @TwoArgumentsOOBOne() #0 {
; CHECK-LABEL: @TwoArgumentsOOBOne dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg0, [5,6)), @Write4_2(arg1, [0,1)){{$}}
; GLOBAL-NEXT: x[8]: [0,9), @Write4_2(arg0, [5,6)), @Write4_2(arg1, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x2 = getelementptr i8, ptr %x, i64 5
  call void @Write4_2(ptr %x2, ptr %x)
  ret void
}

define void @TwoArgumentsOOBOther() #0 {
; CHECK-LABEL: @TwoArgumentsOOBOther dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg0, [4,5)), @Write4_2(arg1, [-1,0)){{$}}
; GLOBAL-NEXT: x[8]: [-1,8), @Write4_2(arg0, [4,5)), @Write4_2(arg1, [-1,0)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = getelementptr i8, ptr %x, i64 -1
  %x2 = getelementptr i8, ptr %x, i64 4
  call void @Write4_2(ptr %x2, ptr %x1)
  ret void
}

define void @TwoArgumentsOOBBoth() #0 {
; CHECK-LABEL: @TwoArgumentsOOBBoth dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[8]: empty-set, @Write4_2(arg0, [5,6)), @Write4_2(arg1, [-1,0)){{$}}
; GLOBAL-NEXT: x[8]: [-1,9), @Write4_2(arg0, [5,6)), @Write4_2(arg1, [-1,0)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i64, align 4
  %x1 = getelementptr i8, ptr %x, i64 -1
  %x2 = getelementptr i8, ptr %x, i64 5
  call void @Write4_2(ptr %x2, ptr %x1)
  ret void
}

define i32 @TestRecursiveNoOffset(ptr %p, i32 %size) #0 {
; CHECK-LABEL: @TestRecursiveNoOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @RecursiveNoOffset(arg0, [0,1)){{$}}
; GLOBAL-NEXT: p[]: full-set, @RecursiveNoOffset(arg0, [0,1)){{$}}
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: sum[4]: [0,4), @RecursiveNoOffset(arg2, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 0, ptr %sum, align 4
; GLOBAL-NEXT: %load = load i32, ptr %sum, align 4
; CHECK-EMPTY:
entry:
  %sum = alloca i32, align 4
  store i32 0, ptr %sum, align 4
  call void @RecursiveNoOffset(ptr %p, i32 %size, ptr %sum)
  %load = load i32, ptr %sum, align 4
  ret i32 %load
}

define void @TestRecursiveWithOffset(i32 %size) #0 {
; CHECK-LABEL: @TestRecursiveWithOffset dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: sum[64]: empty-set, @RecursiveWithOffset(arg1, [0,1)){{$}}
; GLOBAL-NEXT: sum[64]: full-set, @RecursiveWithOffset(arg1, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %sum = alloca i32, i64 16, align 4
  call void @RecursiveWithOffset(i32 %size, ptr %sum)
  ret void
}

; FIXME: IPA should detect that access is safe
define void @TestUpdateArg() #0 {
; CHECK-LABEL: @TestUpdateArg dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[16]: empty-set, @WriteAndReturn8(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[16]: full-set, @WriteAndReturn8(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8, i64 16, align 4
  %0 = call ptr @WriteAndReturn8(ptr %x)
  ret void
}

define void @TestCrossModuleOnce() #0 {
; CHECK-DAG: @TestCrossModuleOnce dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: y[1]: empty-set, @Write1SameModule(arg0, [0,1)){{$}}
; GLOBAL-NEXT: y[1]: [0,1), @Write1SameModule(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %y = alloca i8, align 4
  call void @Write1SameModule(ptr %y)
  ret void
}

define void @TestCrossModuleTwice() #0 {
; CHECK-DAG: @TestCrossModuleTwice dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: z[1]: empty-set, @Write1DiffModule(arg0, [0,1)){{$}}
; GLOBAL-NEXT: z[1]: [0,1), @Write1DiffModule(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %z = alloca i8, align 4
  call void @Write1DiffModule(ptr %z)
  ret void
}

define void @TestCrossModuleConflict() #0 {
; CHECK-DAG: @TestCrossModuleConflict dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; LOCAL-NEXT: x[1]: empty-set, @Write1Private(arg0, [0,1)){{$}}
; GLOBAL-NEXT: x[1]: [-1,0), @Write1Private(arg0, [0,1)){{$}}
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  call void @Write1Private(ptr %x)
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
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:
entry:
  %x = alloca i8, align 4
  call void @Write1Weak(ptr %x)
  ret void
}

define private dso_local void @Private(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 1
  store i8 0, ptr %p1, align 1
  ret void
}

define dso_local void @Write1Module0(ptr %p) #0 {
entry:
  store i8 0, ptr %p, align 1
  ret void
}

define dso_local void @Weak(ptr %p) #0 {
entry:
  %p1 = getelementptr i8, ptr %p, i64 1
  store i8 0, ptr %p1, align 1
  ret void
}

; The rest is from Inputs/ipa.ll

; CHECK-LABEL: @Write1{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %p, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @Write4{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 0, ptr %p, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @Write4_2{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,4){{$}}
; CHECK-NEXT: q[]: [0,4){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 0, ptr %p, align 1
; GLOBAL-NEXT: store i32 0, ptr %q, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @Write8{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,8){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i64 0, ptr %p, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @WriteAndReturn8{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %p, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @PreemptableWrite1 dso_preemptable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %p, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @InterposableWrite1 interposable{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: [0,1){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i8 0, ptr %p, align 1
; CHECK-EMPTY:

; CHECK-LABEL: @ReturnDependent{{$}}
; CHECK-NEXT: args uses:
; CHECK-NEXT: p[]: full-set{{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:

; CHECK-LABEL: @Rec0{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @Write4(arg0, [2,3)){{$}}
; GLOBAL-NEXT: p[]: [2,6)
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:

; CHECK-LABEL: @Rec1{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @Rec0(arg0, [1,2)){{$}}
; GLOBAL-NEXT: p[]: [3,7)
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:

; CHECK-LABEL: @Rec2{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: empty-set, @Rec1(arg0, [-5,-4)){{$}}
; GLOBAL-NEXT: p[]: [-2,2)
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:

; CHECK-LABEL: @RecursiveNoOffset{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: p[]: [0,4), @RecursiveNoOffset(arg0, [4,5)){{$}}
; GLOBAL-NEXT: p[]: full-set, @RecursiveNoOffset(arg0, [4,5)){{$}}
; CHECK-NEXT: acc[]: [0,4), @RecursiveNoOffset(arg2, [0,1)){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: %load0 = load i32, ptr %p, align 4
; GLOBAL-NEXT: %load1 = load i32, ptr %acc, align 4
; GLOBAL-NEXT: store i32 %add, ptr %acc, align 4
; CHECK-EMPTY:

; CHECK-LABEL: @RecursiveWithOffset{{$}}
; CHECK-NEXT: args uses:
; LOCAL-NEXT: acc[]: [0,4), @RecursiveWithOffset(arg1, [4,5)){{$}}
; GLOBAL-NEXT: acc[]: full-set, @RecursiveWithOffset(arg1, [4,5)){{$}}
; CHECK-NEXT: allocas uses:
; GLOBAL-NEXT: safe accesses:
; GLOBAL-NEXT: store i32 0, ptr %acc, align 4
; CHECK-EMPTY:

; CHECK-LABEL: @ReturnAlloca
; CHECK-NEXT: args uses:
; CHECK-NEXT: allocas uses:
; CHECK-NEXT: x[8]: full-set
; GLOBAL-NEXT: safe accesses:
; CHECK-EMPTY:

; INDEX-LABEL: ^0 = module:
; INDEX-DAG: name: "ReturnDependent"{{.*}} guid = [[ReturnDependent:[-0-9]+]]
; INDEX-DAG: name: "Private"{{.*}} guid = [[Private:[-0-9]+]]
; INDEX-DAG: name: "TwoArgumentsOOBOther"{{.*}} guid = [[TwoArgumentsOOBOther:[-0-9]+]]
; INDEX-DAG: name: "Rec2"{{.*}} guid = [[Rec2:[-0-9]+]]
; INDEX-DAG: name: "f1"{{.*}} guid = [[f1:[-0-9]+]]
; INDEX-DAG: name: "PrivateWrite1"{{.*}} guid = [[PrivateWrite1:[-0-9]+]]
; INDEX-DAG: name: "TestRecursiveNoOffset"{{.*}} guid = [[TestRecursiveNoOffset:[-0-9]+]]
; INDEX-DAG: name: "f8left"{{.*}} guid = [[f8left:[-0-9]+]]
; INDEX-DAG: name: "Write4"{{.*}} guid = [[Write4:[-0-9]+]]
; INDEX-DAG: name: "f7"{{.*}} guid = [[f7:[-0-9]+]]
; INDEX-DAG: name: "Write1SameModule"{{.*}} guid = [[Write1SameModule:[-0-9]+]]
; INDEX-DAG: name: "Write8"{{.*}} guid = [[Write8:[-0-9]+]]
; INDEX-DAG: name: "TwoArgumentsOOBOne"{{.*}} guid = [[TwoArgumentsOOBOne:[-0-9]+]]
; INDEX-DAG: name: "f3"{{.*}} guid = [[f3:[-0-9]+]]
; INDEX-DAG: name: "f8right"{{.*}} guid = [[f8right:[-0-9]+]]
; INDEX-DAG: name: "Write4_2"{{.*}} guid = [[Write4_2:[-0-9]+]]
; INDEX-DAG: name: "RecursiveWithOffset"{{.*}} guid = [[RecursiveWithOffset:[-0-9]+]]
; INDEX-DAG: name: "Weak"{{.*}} guid = [[Weak:[-0-9]+]]
; INDEX-DAG: name: "Write1Private"{{.*}} guid = [[Write1Private:[-0-9]+]]
; INDEX-DAG: name: "TestUpdateArg"{{.*}} guid = [[TestUpdateArg:[-0-9]+]]
; INDEX-DAG: name: "TestCrossModuleTwice"{{.*}} guid = [[TestCrossModuleTwice:[-0-9]+]]
; INDEX-DAG: name: "TestCrossModuleWeak"{{.*}} guid = [[TestCrossModuleWeak:[-0-9]+]]
; INDEX-DAG: name: "f2"{{.*}} guid = [[f2:[-0-9]+]]
; INDEX-DAG: name: "PrivateCall"{{.*}} guid = [[PrivateCall:[-0-9]+]]
; INDEX-DAG: name: "TestRecursiveWithOffset"{{.*}} guid = [[TestRecursiveWithOffset:[-0-9]+]]
; INDEX-DAG: name: "f8oobleft"{{.*}} guid = [[f8oobleft:[-0-9]+]]
; INDEX-DAG: name: "InterposableWrite1"{{.*}} guid = [[InterposableWrite1:[-0-9]+]]
; INDEX-DAG: name: "f4"{{.*}} guid = [[f4:[-0-9]+]]
; INDEX-DAG: name: "TestCrossModuleConflict"{{.*}} guid = [[TestCrossModuleConflict:[-0-9]+]]
; INDEX-DAG: name: "RecursiveNoOffset"{{.*}} guid = [[RecursiveNoOffset:[-0-9]+]]
; INDEX-DAG: name: "TwoArgumentsOOBBoth"{{.*}} guid = [[TwoArgumentsOOBBoth:[-0-9]+]]
; INDEX-DAG: name: "f5"{{.*}} guid = [[f5:[-0-9]+]]
; INDEX-DAG: name: "f6"{{.*}} guid = [[f6:[-0-9]+]]
; INDEX-DAG: name: "Write1Weak"{{.*}} guid = [[Write1Weak:[-0-9]+]]
; INDEX-DAG: name: "Write1"{{.*}} guid = [[Write1:[-0-9]+]]
; INDEX-DAG: name: "PreemptableWrite1"{{.*}} guid = [[PreemptableWrite1:[-0-9]+]]
; INDEX-DAG: name: "f8oobright"{{.*}} guid = [[f8oobright:[-0-9]+]]
; INDEX-DAG: name: "InterposableCall"{{.*}} guid = [[InterposableCall:[-0-9]+]]
; INDEX-DAG: name: "TestCrossModuleOnce"{{.*}} guid = [[TestCrossModuleOnce:[-0-9]+]]
; INDEX-DAG: name: "WriteAndReturn8"{{.*}} guid = [[WriteAndReturn8:[-0-9]+]]
; INDEX-DAG: name: "TwoArguments"{{.*}} guid = [[TwoArguments:[-0-9]+]]
; INDEX-DAG: name: "Write1Module0"{{.*}} guid = [[Write1Module0:[-0-9]+]]
; INDEX-DAG: name: "PreemptableCall"{{.*}} guid = [[PreemptableCall:[-0-9]+]]
; INDEX-DAG: name: "Write1DiffModule"{{.*}} guid = [[Write1DiffModule:[-0-9]+]]
; INDEX-DAG: name: "ExternalCall"{{.*}} guid = [[ExternalCall:[-0-9]+]]
; INDEX-LABEL: = blockcount:

; INDEX-LABEL: ^0 = module:
; INDEX-DAG: name: "ReturnDependent"{{.*}} guid = [[ReturnDependent:[-0-9]+]]
; INDEX-DAG: name: "Rec0"{{.*}} guid = [[Rec0:[-0-9]+]]
; INDEX-DAG: name: "Rec2"{{.*}} guid = [[Rec2:[-0-9]+]]
; INDEX-DAG: name: "Write4"{{.*}} guid = [[Write4:[-0-9]+]]
; INDEX-DAG: name: "Write1SameModule"{{.*}} guid = [[Write1SameModule:[-0-9]+]]
; INDEX-DAG: name: "Write8"{{.*}} guid = [[Write8:[-0-9]+]]
; INDEX-DAG: name: "Write4_2"{{.*}} guid = [[Write4_2:[-0-9]+]]
; INDEX-DAG: name: "RecursiveWithOffset"{{.*}} guid = [[RecursiveWithOffset:[-0-9]+]]
; INDEX-DAG: name: "Weak"{{.*}} guid = [[Weak:[-0-9]+]]
; INDEX-DAG: name: "Write1Private"{{.*}} guid = [[Write1Private:[-0-9]+]]
; INDEX-DAG: name: "InterposableWrite1"{{.*}} guid = [[InterposableWrite1:[-0-9]+]]
; INDEX-DAG: name: "Private"{{.*}} guid = [[Private:[-0-9]+]]
; INDEX-DAG: name: "Rec1"{{.*}} guid = [[Rec1:[-0-9]+]]
; INDEX-DAG: name: "RecursiveNoOffset"{{.*}} guid = [[RecursiveNoOffset:[-0-9]+]]
; INDEX-DAG: name: "Write1Weak"{{.*}} guid = [[Write1Weak:[-0-9]+]]
; INDEX-DAG: name: "Write1"{{.*}} guid = [[Write1:[-0-9]+]]
; INDEX-DAG: name: "PreemptableWrite1"{{.*}} guid = [[PreemptableWrite1:[-0-9]+]]
; INDEX-DAG: name: "WriteAndReturn8"{{.*}} guid = [[WriteAndReturn8:[-0-9]+]]
; INDEX-DAG: name: "Write1Module0"{{.*}} guid = [[Write1Module0:[-0-9]+]]
; INDEX-DAG: name: "Write1DiffModule"{{.*}} guid = [[Write1DiffModule:[-0-9]+]]
; INDEX-DAG: name: "ExternalCall"{{.*}} guid = [[ExternalCall:[-0-9]+]]
; INDEX-DAG: name: "ReturnAlloca"{{.*}} guid = [[ReturnAlloca:[-0-9]+]]
; INDEX-LABEL: = blockcount:

; INDEX-LABEL: ^0 = module:
; INDEX-DAG: guid: [[ReturnDependent]], {{.*}}, funcFlags: ({{.*}}))))
; INDEX-DAG: guid: [[Rec0]], {{.*}}, params: ((param: 0, offset: [2, 5])))))
; INDEX-DAG: guid: [[Rec2]], {{.*}}, params: ((param: 0, offset: [-2, 1])))))
; INDEX-DAG: guid: [[Write4]], {{.*}}, params: ((param: 0, offset: [0, 3])))))
; INDEX-DAG: guid: [[Write1SameModule]], {{.*}}, params: ((param: 0, offset: [0, 0])))))
; INDEX-DAG: guid: [[Write8]], {{.*}}, params: ((param: 0, offset: [0, 7])))))
; INDEX-DAG: guid: [[Write4_2]], {{.*}}, params: ((param: 0, offset: [0, 3]), (param: 1, offset: [0, 3])))))
; INDEX-DAG: guid: [[RecursiveWithOffset]], {{.*}}, calls: ((callee: ^{{[0-9]+}}, tail: 1)))))
; INDEX-DAG: guid: [[Weak]], {{.*}}, funcFlags: ({{.*}}))))
; INDEX-DAG: guid: [[Write1Private]], {{.*}}, params: ((param: 0, offset: [-1, -1])))))
; INDEX-DAG: guid: [[InterposableWrite1]], {{.*}}, params: ((param: 0, offset: [0, 0])))))
; INDEX-DAG: guid: [[Private]], {{.*}}, params: ((param: 0, offset: [-1, -1])))))
; INDEX-DAG: guid: [[Rec1]], {{.*}}, params: ((param: 0, offset: [3, 6])))))
; INDEX-DAG: guid: [[RecursiveNoOffset]], {{.*}}, params: ((param: 2, offset: [0, 3])))))
; INDEX-DAG: guid: [[Write1Weak]], {{.*}}, calls: ((callee: ^{{[0-9]+}})))))
; INDEX-DAG: guid: [[Write1]], {{.*}}, params: ((param: 0, offset: [0, 0])))))
; INDEX-DAG: guid: [[PreemptableWrite1]], {{.*}}, funcFlags: ({{.*}}))))
; INDEX-DAG: guid: [[WriteAndReturn8]], {{.*}}, funcFlags: ({{.*}}))))
; INDEX-DAG: guid: [[Write1DiffModule]], {{.*}}, funcFlags: ({{.*}}))))
; INDEX-DAG: guid: [[ReturnAlloca]], {{.*}}, insts: 2)))
; INDEX-LABEL: blockcount:
