; RUN: llc -mtriple=x86_64-linux -no-stack-coloring=false -debug-only=stack-coloring < %s -o /dev/null 2>&1 | FileCheck %s
; REQUIRES: asserts

; Test that volatile stack slots accessed after setjmp are not merged.
; Volatile variables must retain their values across longjmp, so their
; stack slots cannot be reused even if their lifetimes don't overlap.

declare i32 @setjmp(ptr) returns_twice
declare void @baz(ptr)
declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
declare dso_local void @stash(ptr noundef, ptr noundef) local_unnamed_addr 
declare dso_local void @use(ptr noundef) local_unnamed_addr 

; CHECK-LABEL: setjmp_test
; CHECK: Conservative slots : { 1 1 }
; CHECK: Merge 0 slots

define void @setjmp_test(ptr %jump_buffer) {
entry:
  %foo = alloca i32, align 4
  %bar = alloca [100 x i32], align 4

  call void @llvm.lifetime.start.p0(ptr %foo)
  call void @llvm.lifetime.start.p0(ptr %bar)

  %setjmp_result = call i32 @setjmp(ptr %jump_buffer)
  %cmp = icmp eq i32 %setjmp_result, 0
  br i1 %cmp, label %after_setjmp, label %continue

after_setjmp:
  store volatile i32 100, ptr %foo, align 4
  br label %exit

continue:
  store i32 100, ptr %bar, align 4
  call void @baz(ptr %bar)
  call void @llvm.lifetime.end.p0(ptr %foo)
  call void @llvm.lifetime.end.p0(ptr %bar)
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: setjmp_test_2
; CHECK: Conservative slots : { 1 1 }
; CHECK: Merge 0 slots


%struct.T = type { [100 x i32] }

define dso_local void @setjmp_test_2(ptr %jump_buffer) local_unnamed_addr  {
entry:
  %test1 = alloca %struct.T, align 4
  %test2 = alloca %struct.T, align 4
  call void @llvm.lifetime.start.p0(ptr %test1)
  %call = call i32 @setjmp(ptr %jump_buffer)
  %cmp = icmp eq i32 %call, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @stash(ptr %jump_buffer, ptr %test1) 
  br label %if.end

if.else:
  call void @llvm.lifetime.start.p0(ptr %test2) 
  call void @use(ptr %test2) 
  call void @llvm.lifetime.end.p0(ptr %test2) 
  br label %if.end

if.end:
  call void @llvm.lifetime.end.p0(ptr %test1) 
  ret void
}

