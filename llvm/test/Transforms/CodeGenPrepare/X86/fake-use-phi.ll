; RUN: opt < %s -passes='require<profile-summary>,function(codegenprepare)' -S -mtriple=x86_64 | FileCheck %s --implicit-check-not="llvm.fake.use"
;
; When performing return duplication to enable
; tail call optimization we clone fake uses that exist in the to-be-eliminated
; return block into the predecessor blocks. When doing this with fake uses
; of PHI-nodes, they cannot be easily copied, but require the correct operand.
; We are currently not able to do this correctly, so we suppress the cloning
; of such fake uses at the moment.
;
; There should be no fake use of a call result in any of the resulting return
; blocks.

; Fake uses of `this` should be duplicated into both return blocks.
; CHECK: if.then:
; CHECK: @llvm.fake.use({{.*}}this
; CHECK: if.else:
; CHECK: @llvm.fake.use({{.*}}this

; CHECK: declare void @llvm.fake.use

source_filename = "test.ll"

%class.a = type { i8 }

declare i32 @foo(ptr nonnull dereferenceable(1)) local_unnamed_addr
declare i32 @bar(ptr nonnull dereferenceable(1)) local_unnamed_addr

define hidden void @func(ptr nonnull dereferenceable(1) %this) local_unnamed_addr align 2 optdebug {
entry:
  %b = getelementptr inbounds %class.a, ptr %this, i64 0, i32 0
  %0 = load i8, i8* %b, align 1
  %tobool.not = icmp eq i8 %0, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call i32 @foo(ptr nonnull dereferenceable(1) %this)
  %call2 = tail call i32 @bar(ptr nonnull dereferenceable(1) %this)
  br label %if.end

if.else:                                          ; preds = %entry
  %call4 = tail call i32 @bar(ptr nonnull dereferenceable(1) %this)
  %call5 = tail call i32 @foo(ptr nonnull dereferenceable(1) %this)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %call4.sink = phi i32 [ %call4, %if.else ], [ %call, %if.then ]
  notail call void (...) @llvm.fake.use(i32 %call4.sink)
  notail call void (...) @llvm.fake.use(ptr nonnull %this)
  ret void
}
