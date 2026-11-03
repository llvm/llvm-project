; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:8:8-i8:8:8-i16:8:8-i32:8:8"
target triple = "msp430"

@g = global i8 0

define zeroext i16 @add(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: add:
; CHECK: add @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = add i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @sub(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: sub:
; CHECK: sub @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = sub i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @or(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: or:
; CHECK: bis @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = or i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @xor(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: xor:
; CHECK: xor @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = xor i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define zeroext i16 @and(ptr nocapture %a, i16 zeroext %n) nounwind readonly {
entry:
  %cmp8 = icmp eq i16 %n, 0                       ; <i1> [#uses=1]
  br i1 %cmp8, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.010 = phi i16 [ 0, %entry ], [ %inc, %for.body ] ; <i16> [#uses=2]
  %sum.09 = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  %arrayidx = getelementptr i16, ptr %a, i16 %i.010   ; <i16*> [#uses=1]
; CHECK-LABEL: and:
; CHECK: and @r{{[0-9]+}}+, r{{[0-9]+}}
  %tmp4 = load i16, ptr %arrayidx                     ; <i16> [#uses=1]
  %add = and i16 %tmp4, %sum.09                   ; <i16> [#uses=2]
  %inc = add i16 %i.010, 1                        ; <i16> [#uses=2]
  %exitcond = icmp eq i16 %inc, %n                ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i16 [ 0, %entry ], [ %add, %for.body ] ; <i16> [#uses=1]
  ret i16 %sum.0.lcssa
}

define ptr @copy8_postinc(ptr %Src, ptr %Dest) nounwind {
; CHECK-LABEL: copy8_postinc:
; CHECK: mov.b @r{{[0-9]+}}+, 0(r{{[0-9]+}})
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 1
  %val = load i8, ptr %Src, align 1
  store i8 %val, ptr %Dest, align 1
  ret ptr %incdec.ptr
}

define ptr @copy16_postinc(ptr %Src, ptr %Dest) nounwind {
; CHECK-LABEL: copy16_postinc:
; CHECK: mov @r{{[0-9]+}}+, 0(r{{[0-9]+}})
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 2
  %val = load i16, ptr %Src, align 2
  store i16 %val, ptr %Dest, align 2
  ret ptr %incdec.ptr
}

; When the loaded value is used for something else, the value still has to be
; loaded into a register.
define ptr @copy8_value_used(ptr %Src, ptr %Dest, ptr %Out) nounwind {
; CHECK-LABEL: copy8_value_used:
; CHECK: mov.b @r{{[0-9]+}}+, r{{[0-9]+}}
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 1
  %val = load i8, ptr %Src, align 1
  store i8 %val, ptr %Dest, align 1
  store i8 %val, ptr %Out, align 1
  ret ptr %incdec.ptr
}

define ptr @copy8_store_between(ptr %Src, ptr %Dest, ptr %Other) nounwind {
; CHECK-LABEL: copy8_store_between:
; CHECK: mov.b @r{{[0-9]+}}+, r{{[0-9]+}}
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 1
  %val = load i8, ptr %Src, align 1
  ; Volatile store between the load of the source pointer and storing the value
  ; to the target address. Folding the addition is not allowed.
  store volatile i8 0, ptr %Other, align 1
  store i8 %val, ptr %Dest, align 1
  ret ptr %incdec.ptr
}

define ptr @copy8_to_global_postinc(ptr %Src, ptr %Dest) nounwind {
; CHECK-LABEL: copy8_to_global_postinc:
; CHECK: mov.b @r{{[0-9]+}}+, &g
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 1
  %val = load i8, ptr %Src, align 1
  store i8 %val, ptr @g, align 1
  ret ptr %incdec.ptr
}

define ptr @copy8_displacement(ptr %Src, ptr %Dest) nounwind {
; CHECK-LABEL: copy8_displacement:
; CHECK: mov.b @r{{[0-9]+}}+, 4(r{{[0-9]+}})
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 1
  %dst = getelementptr inbounds i8, ptr %Dest, i16 4
  %val = load i8, ptr %Src, align 1
  store i8 %val, ptr %dst, align 1
  ret ptr %incdec.ptr
}

; The destination uses the source register. MSP430 computes the destination
; address after the source increment, so folding would store to Src + 1.
define ptr @copy8_same_register(ptr %Src) nounwind {
; CHECK-LABEL: copy8_same_register:
; CHECK-NOT: mov.b @r{{[0-9]+}}+, 0(r{{[0-9]+}})
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %Src, i16 1
  %val = load volatile i8, ptr %Src, align 1
  store volatile i8 %val, ptr %Src, align 1
  ret ptr %incdec.ptr
}
