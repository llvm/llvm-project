; RUN: opt -disable-output -licm -print-memoryssa -enable-mssa-loop-dependency=true < %s 2>&1 | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @main()
; CHECK: 5 = MemoryPhi(
; CHECK-NOT: 7 = MemoryPhi(
@v_67 = external dso_local global i32, align 1
@v_76 = external dso_local global i16, align 1
@v_86 = external dso_local global i16 *, align 1

define dso_local void @main() {
entry:
  %v_59 = alloca i16, align 2
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i16 undef, i16* %v_59, align 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br i1 undef, label %if.else568, label %cond.end82

cond.false69:                                     ; No predecessors!
  br label %cond.end82

cond.end82:                                       ; preds = %cond.false69, %cond.true55
  br i1 undef, label %if.else568, label %land.lhs.true87

land.lhs.true87:                                  ; preds = %cond.end82
  br i1 undef, label %if.then88, label %if.else568

if.then88:                                        ; preds = %land.lhs.true87
  store i16 * @v_76, i16 ** @v_86, align 1
  br label %if.end569

if.else568:                                       ; preds = %land.lhs.true87, %cond.end82, %for.end
  store volatile i32 undef, i32 * @v_67, align 1
  br label %if.end569

if.end569:                                        ; preds = %if.else568, %if.then88
  ret void
}


; CHECK-LABEL: @f()
; CHECK: 8 = MemoryPhi(
; CHECK: 7 = MemoryPhi(
; CHECK: 11 = MemoryPhi(
; CHECK: 10 = MemoryPhi(
; CHECK: 9 = MemoryPhi(
define void @f() {
entry:
  %e = alloca i16, align 1
  br label %lbl1

lbl1:                                             ; preds = %if.else, %for.end5, %entry
  store i16 undef, i16* %e, align 1
  %0 = load i16, i16* %e, align 1
  %call = call i16 @g(i16 %0)
  br i1 undef, label %for.end, label %if.else

for.end:                                          ; preds = %if.then
  br i1 true, label %for.cond2, label %lbl2

lbl2:                                             ; preds = %for.body4, %if.end
  br label %for.cond2

for.cond2:                                        ; preds = %lbl3
  br i1 undef, label %for.body4, label %for.end5

for.body4:                                        ; preds = %for.cond2
  br label %lbl2

for.end5:                                         ; preds = %for.cond2
  switch i32 undef, label %unreachable [
    i32 0, label %if.end12
    i32 2, label %lbl1
  ]

if.else:                                          ; preds = %lbl1
  switch i32 undef, label %unreachable [
    i32 0, label %if.end12
    i32 2, label %lbl1
  ]

if.end12:                                         ; preds = %cleanup.cont11s, %cleanup.cont
  call void @llvm.lifetime.end.p0i8(i64 1, i8* undef)
  ret void

unreachable:                                      ; preds = %if.else, %for.end5
  unreachable
}

declare i16 @g(i16)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
