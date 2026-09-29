; RUN: llc < %s -mtriple=i386-unknown-freebsd -enable-misched -relocation-model=pic | FileCheck %s

@c = external local_unnamed_addr global ptr

declare i32 @fn2() local_unnamed_addr

declare i32 @fn3() local_unnamed_addr

define noundef i32 @fn4() #0 {
entry:
  %tmp0 = load i32, ptr @fn4, align 4
; CHECK: movl fn4@GOT(%ebx), %edi
; CHECK-NEXT: movl (%edi), %edx
  %tmp1 = load ptr, ptr @c, align 4
; CHECK: movl c@GOT(%ebx), %eax
; CHECK-NEXT: movl (%eax), %esi
; CHECK-NEXT: testl %esi, %esi
  %cmp.g = icmp eq ptr %tmp1, null
  br i1 %cmp.g, label %if.then.g, label %if.end3.g

if.then.g:                                        ; preds = %entry
  %tmp2 = load i32, ptr inttoptr (i32 1 to ptr), align 4
  %cmp1.g = icmp slt i32 %tmp2, 0
  br i1 %cmp1.g, label %if.then2.g, label %if.end3.g

if.then2.g:                                       ; preds = %if.then.g
  %.g = load volatile i32, ptr null, align 2147483648
  br label %f.exit

if.end3.g:                                        ; preds = %if.then.g, %entry
  %h.i.g = icmp eq i32 %tmp0, 0
  br i1 %h.i.g, label %f.exit, label %while.body.g

while.body.g:                                     ; preds = %if.end3.g, %if.end8.g
  %buff.addr.019.g = phi ptr [ %incdec.ptr.g, %if.end8.g ], [ @fn4, %if.end3.g ]
  %g.addr.018.g = phi i32 [ %dec.g, %if.end8.g ], [ %tmp0, %if.end3.g ]
  %call4.g = tail call i32 @fn3(ptr %tmp1, ptr %buff.addr.019.g, i32 %g.addr.018.g)
  %cmp5.g = icmp slt i32 %call4.g, 0
  br i1 %cmp5.g, label %if.then6.g, label %if.end8.g

if.then6.g:                                       ; preds = %while.body.g
  %call7.g = tail call i32 @fn2(ptr null)
  br label %f.exit

if.end8.g:                                        ; preds = %while.body.g
  %dec.g = add i32 %g.addr.018.g, 1
  %incdec.ptr.g = getelementptr i32, ptr %buff.addr.019.g, i32 1
  store i64 0, ptr %tmp1, align 4
  %h.not.g = icmp eq i32 %dec.g, 0
  br i1 %h.not.g, label %f.exit, label %while.body.g

f.exit:                                           ; preds = %if.end8.g, %if.then6.g, %if.end3.g, %if.then2.g
  ret i32 0
}

attributes #0 = { "frame-pointer"="all" "tune-cpu"="generic" }
