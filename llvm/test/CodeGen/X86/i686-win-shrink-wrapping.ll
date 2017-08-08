; RUN: llc %s -o - -enable-shrink-wrap=true | FileCheck %s --check-prefix=CHECK --check-prefix=ENABLE
; RUN: llc %s -o - -enable-shrink-wrap=false | FileCheck %s --check-prefix=CHECK --check-prefix=DISABLE
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

%struct.S = type { i32 }

; Check that we do not use a basic block that has EFLAGS as live-in
; if we need to realign the stack.
; PR27531.
; CHECK-LABEL: stackRealignment:
; Prologue code.
; CHECK: pushl
; Make sure we actually perform some stack realignment.
; CHECK: andl ${{[-0-9]+}}, %esp
; This is the end of the entry block.
; The prologue should have happened before that point because past
; this point, EFLAGS is live.
; CHECK: jg
define x86_thiscallcc void @stackRealignment(%struct.S* %this) {
entry:
  %data = alloca [1 x i32], align 4
  %d = alloca double, align 8
  %tmp = bitcast [1 x i32]* %data to i8*
  %arrayinit.begin = getelementptr inbounds [1 x i32], [1 x i32]* %data, i32 0, i32 0
  %x_ = getelementptr inbounds %struct.S, %struct.S* %this, i32 0, i32 0
  %tmp1 = load i32, i32* %x_, align 4
  %cmp = icmp sgt i32 %tmp1, 32
  %cond = select i1 %cmp, i32 42, i32 128
  store i32 %cond, i32* %arrayinit.begin, align 4
  %cmp3 = icmp slt i32 %tmp1, 32
  br i1 %cmp3, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %tmp2 = bitcast double* %d to i8*
  call x86_thiscallcc void @bar(%struct.S* nonnull %this, i32* %arrayinit.begin, double* nonnull %d)
  br label %cleanup

cleanup:                                          ; preds = %if.end, %entry
  ret void
}

; Function Attrs: optsize
declare x86_thiscallcc void @bar(%struct.S*, i32*, double*)
