; RUN: llvm-as < %s | llc -march=arm -disable-arm-if-conversion > %t 
; RUN: grep bne %t
; RUN: grep bge %t
; RUN: grep bhs %t
; RUN: grep blo %t

define void @f1(i32 %a, i32 %b, i32* %v) {
entry:
        %tmp = icmp eq i32 %a, %b               ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}

define void @f2(i32 %a, i32 %b, i32* %v) {
entry:
        %tmp = icmp slt i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}

define void @f3(i32 %a, i32 %b, i32* %v) {
entry:
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %cond_true, label %return

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}

define void @f4(i32 %a, i32 %b, i32* %v) {
entry:
        %tmp = icmp ult i32 %a, %b              ; <i1> [#uses=1]
        br i1 %tmp, label %return, label %cond_true

cond_true:              ; preds = %entry
        store i32 0, i32* %v
        ret void

return:         ; preds = %entry
        ret void
}
