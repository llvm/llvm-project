; RUN: llc -mtriple=armv7a-linux-androideabi %s -o - | FileCheck %s

@a = local_unnamed_addr global i16 -1, align 2
@b = local_unnamed_addr global i16 0, align 2

; CHECK-LABEL: pr39060:
; CHECK: ldrh
; CHECK: ldrh
; CHECK: sub
; CHECK: uxth
define void @pr39060() local_unnamed_addr #0 {
entry:
  %0 = load i16, i16* @a, align 2
  %1 = load i16, i16* @b, align 2
  %sub = add i16 %1, -1
  %cmp = icmp eq i16 %0, %sub
  br i1 %cmp, label %if.else, label %if.then

if.then:
  tail call void bitcast (void (...)* @f to void ()*)() #2
  br label %if.end

if.else:
  tail call void bitcast (void (...)* @g to void ()*)() #2
  br label %if.end

if.end:
  ret void
}

declare void @f(...) local_unnamed_addr #1

declare void @g(...) local_unnamed_addr #1
