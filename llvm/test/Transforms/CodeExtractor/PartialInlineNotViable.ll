; RUN: opt < %s -passes=partial-inliner -skip-partial-inlining-cost-analysis -S | FileCheck %s
;

define i32 @callee_indr_branch(i32 %v) {
entry:
  %cmp = icmp sgt i32 %v, 2000
  %addr = select i1 %cmp, i8* blockaddress(@callee_indr_branch, %if.then), i8* blockaddress(@callee_indr_branch, %if.end)
  indirectbr i8* %addr, [ label %if.then, label %if.end]

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %v, 10
  br label %if.then2

if.then2:
  %sub = sub i32 %v, 10
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %v2 = phi i32 [ %v, %entry ], [ %mul, %if.then2 ]
  %add = add nsw i32 %v2, 200
  ret i32 %add
}

declare void @use_fp(i8 *)
declare void @llvm.localescape(...)
declare i8* @llvm.frameaddress(i32)
declare i8* @llvm.localrecover(i8*, i8*, i32)



define i32 @callee_frameescape(i32 %v) {
entry:
  %a = alloca i32
  call void (...) @llvm.localescape(i32* %a)
  %cmp = icmp sgt i32 %v, 2000
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %v, 10
  br label %if.then2

if.then2:
  %sub = sub i32 %v, 10
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %v2 = phi i32 [ %v, %entry ], [ %mul, %if.then2 ]
  %add = add nsw i32 %v2, 200
  ret i32 %add
}


; CHECK-LABEL: @caller
; CHECK: %r1 = call i32 @callee_indr_branch(i32 %v)
; CHECK-NEXT: %r2 = call i32 @callee_frameescape(i32 %v)
define i32 @caller(i32 %v) {
entry:
  %r1 = call i32 @callee_indr_branch(i32 %v)
  %r2 = call i32 @callee_frameescape(i32 %v)
  %res = add i32 %r1, %r2
  ret i32 %res
}

