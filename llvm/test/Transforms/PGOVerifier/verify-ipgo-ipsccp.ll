; RUN: opt < %s -verify-ipgo -debug-only=verify-ipgo -force-specialization=true -passes=pgo-instr-gen,ipsccp -S -disable-output 2>&1 | FileCheck --check-prefix=PGOGEN %s
; RUN: opt < %s -verify-ipgo -force-specialization=true -passes=pgo-instr-gen,ipsccp -S -disable-output 2>&1 | FileCheck --check-prefix=PGOGEN-VERIFY %s
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-ipsccp.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -debug-only=verify-ipgo -force-specialization=true -passes=pgo-instr-use,ipsccp -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck --check-prefix=PGOUSE %s
; RUN: llvm-profdata merge %S/Inputs/pgo-instr-use-ipsccp.proftext -o %t.profdata && \
; RUN:     opt < %s -verify-ipgo -force-specialization=true -passes=pgo-instr-use,ipsccp -pgo-test-profile-file=%t.profdata -S -disable-output 2>&1 | FileCheck --check-prefix=PGOUSE-VERIFY %s
; REQUIRES: asserts

source_filename = "proftest.c"

@res = dso_local local_unnamed_addr global [10 x i32] zeroinitializer, align 16
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1

define dso_local void @test(i32 %a, i32 %b) local_unnamed_addr {
entry:
  %cmp4 = icmp slt i32 %b, 0
  %mul = select i1 %cmp4, i32 1, i32 %a
  %result1.0 = mul nsw i32 %b, %mul
  %add = add nsw i32 %a, %result1.0
  store i32 %add, ptr @res, align 16
  ret void
}

define dso_local i32 @main() local_unnamed_addr {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup3, %entry
  %j.0 = phi i32 [ 0, %entry ], [ %inc6, %for.cond.cleanup3 ]
  %cmp = icmp samesign ult i32 %j.0, 1000
  br i1 %cmp, label %for.cond1, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  ret i32 0

for.cond1:                                        ; preds = %for.cond, %for.body4
  %i.0 = phi i64 [ %inc, %for.body4 ], [ 0, %for.cond ]
  %cmp2 = icmp samesign ult i64 %i.0, 1000000
  br i1 %cmp2, label %for.body4, label %for.cond.cleanup3

for.cond.cleanup3:                                ; preds = %for.cond1
  %0 = load i32, ptr @res, align 16
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %0)
  %inc6 = add nuw nsw i32 %j.0, 1
  br label %for.cond

for.body4:                                        ; preds = %for.cond1
  call void @test(i32 10, i32 0)
  call void @test(i32 10, i32 -1)
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond1
}

declare i32 @printf(ptr, ...) local_unnamed_addr
; PGOGEN: *** IPGO Verification After

; PGOGEN-VERIFY: *** IPGO Verification After

; PGOUSE: *** IPGO Verification After

; PGOUSE-VERIFY: *** IPGO Verification After
