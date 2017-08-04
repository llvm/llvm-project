; RUN: llc < %s -mtriple=i686-linux   -show-mc-encoding | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK32
; RUN: llc < %s -mtriple=x86_64-linux -show-mc-encoding | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK64
; RUN: llc < %s -mtriple=x86_64-win32 -show-mc-encoding | FileCheck %s --check-prefix=CHECK --check-prefix=WIN64

declare void @foo()
declare void @bar()

define void @f(i32 %x, i32 %y) optsize {
entry:
	%p = icmp eq i32 %x, %y
  br i1 %p, label %bb1, label %bb2
bb1:
  tail call void @foo()
  ret void
bb2:
  tail call void @bar()
  ret void

; CHECK-LABEL: f:
; CHECK: cmp
; CHECK: jne bar
; Check that the asm doesn't just look good, but uses the correct encoding.
; CHECK: encoding: [0x75,A]
; CHECK: jmp foo
}

define void @f_non_leaf(i32 %x, i32 %y) optsize {
entry:
  ; Force %ebx to be spilled on the stack, turning this into
  ; not a "leaf" function for Win64.
  tail call void asm sideeffect "", "~{ebx}"()

	%p = icmp eq i32 %x, %y
  br i1 %p, label %bb1, label %bb2
bb1:
  tail call void @foo()
  ret void
bb2:
  tail call void @bar()
  ret void

; CHECK-LABEL: f_non_leaf:
; WIN64-NOT: je foo
; WIN64-NOT: jne bar
; WIN64: jne
; WIN64: jmp foo
; WIN64: jmp bar
}

declare x86_thiscallcc zeroext i1 @baz(i8*, i32)
define x86_thiscallcc zeroext i1 @BlockPlacementTest(i8* %this, i32 %x) optsize {
entry:
  %and = and i32 %x, 42
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %land.end, label %land.rhs

land.rhs:
  %and6 = and i32 %x, 44
  %tobool7 = icmp eq i32 %and6, 0
  br i1 %tobool7, label %lor.rhs, label %land.end

lor.rhs:
  %call = tail call x86_thiscallcc zeroext i1 @baz(i8* %this, i32 %x) #2
  br label %land.end

land.end:
  %0 = phi i1 [ false, %entry ], [ true, %land.rhs ], [ %call, %lor.rhs ]
  ret i1 %0

; Make sure machine block placement isn't confused by the conditional tail call,
; but sees that it can fall through to the next block.
; CHECK-LABEL: BlockPlacementTest
; CHECK: je baz
; CHECK-NOT: xor
; CHECK: ret
}



%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
declare zeroext i1 @_Z20isValidIntegerSuffixN9__gnu_cxx17__normal_iteratorIPKcSsEES3_(i8*, i8*)

define zeroext i1 @pr31257(%"class.std::basic_string"* nocapture readonly dereferenceable(8) %s) minsize {
; CHECK-LABEL: pr31257
entry:
  %_M_p.i.i = getelementptr inbounds %"class.std::basic_string", %"class.std::basic_string"* %s, i64 0, i32 0, i32 0
  %0 = load i8*, i8** %_M_p.i.i, align 8
  %arrayidx.i.i.i54 = getelementptr inbounds i8, i8* %0, i64 -24
  %_M_length.i.i55 = bitcast i8* %arrayidx.i.i.i54 to i64*
  %1 = load i64, i64* %_M_length.i.i55, align 8
  %add.ptr.i56 = getelementptr inbounds i8, i8* %0, i64 %1
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %it.sroa.0.0 = phi i8* [ %0, %entry ], [ %incdec.ptr.i, %for.inc ]
  %state.0 = phi i32 [ 0, %entry ], [ %state.1, %for.inc ]
  %cmp.i = icmp eq i8* %it.sroa.0.0, %add.ptr.i56
  br i1 %cmp.i, label %5, label %for.body

for.body:                                         ; preds = %for.cond
  switch i32 %state.0, label %for.inc [
    i32 0, label %sw.bb
    i32 1, label %sw.bb14
    i32 2, label %sw.bb22
  ]

sw.bb:                                            ; preds = %for.body
  %2 = load i8, i8* %it.sroa.0.0, align 1
  switch i8 %2, label %if.else [
    i8 43, label %for.inc
    i8 45, label %for.inc
  ]

if.else:                                          ; preds = %sw.bb
  %conv9 = zext i8 %2 to i32
  %isdigittmp45 = add nsw i32 %conv9, -48
  %isdigit46 = icmp ult i32 %isdigittmp45, 10
  br i1 %isdigit46, label %for.inc, label %cleanup.thread.loopexit

sw.bb14:                                          ; preds = %for.body
  %3 = load i8, i8* %it.sroa.0.0, align 1
  %conv16 = zext i8 %3 to i32
  %isdigittmp43 = add nsw i32 %conv16, -48
  %isdigit44 = icmp ult i32 %isdigittmp43, 10
  br i1 %isdigit44, label %for.inc, label %cleanup.thread.loopexit

sw.bb22:                                          ; preds = %for.body
  %4 = load i8, i8* %it.sroa.0.0, align 1
  %conv24 = zext i8 %4 to i32
  %isdigittmp = add nsw i32 %conv24, -48
  %isdigit = icmp ult i32 %isdigittmp, 10
  br i1 %isdigit, label %for.inc, label %if.else28

; Make sure Machine Copy Propagation doesn't delete the mov to %ecx becaue it
; thinks the conditional tail call clobbers it.
; CHECK64-LABEL: .LBB3_11:
; CHECK64:       movzbl  (%rdi), %ecx
; CHECK64-NEXT:  addl    $-48, %ecx
; CHECK64-NEXT:  cmpl    $10, %ecx
; CHECK64-NEXT:  movl    %r9d, %ecx
; CHECK64-NEXT:  jae     _Z20isValidIntegerSuffixN9__gnu_cxx17__normal_iteratorIPKcSsEE

if.else28:                                        ; preds = %sw.bb22
  %call34 = tail call zeroext i1 @_Z20isValidIntegerSuffixN9__gnu_cxx17__normal_iteratorIPKcSsEES3_(i8* nonnull %it.sroa.0.0, i8* %add.ptr.i56)
  br label %cleanup.thread

for.inc:                                          ; preds = %sw.bb, %sw.bb, %sw.bb22, %sw.bb14, %if.else, %for.body
  %state.1 = phi i32 [ %state.0, %for.body ], [ 1, %sw.bb ], [ 2, %if.else ], [ 2, %sw.bb14 ], [ 2, %sw.bb22 ], [ 1, %sw.bb ]
  %incdec.ptr.i = getelementptr inbounds i8, i8* %it.sroa.0.0, i64 1
  br label %for.cond

; <label>:5:                                      ; preds = %for.cond
  %cmp37 = icmp eq i32 %state.0, 2
  br label %cleanup.thread

cleanup.thread.loopexit:                          ; preds = %if.else, %sw.bb14
  br label %cleanup.thread

cleanup.thread:                                   ; preds = %cleanup.thread.loopexit, %if.else28, %5
  %6 = phi i1 [ %cmp37, %5 ], [ %call34, %if.else28 ], [ false, %cleanup.thread.loopexit ]
  ret i1 %6
}
