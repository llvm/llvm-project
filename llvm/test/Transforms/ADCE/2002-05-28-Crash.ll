; This testcase is distilled from the GNU rx package.  The loop should be
; removed but causes a problem when ADCE does.  The source function is:
; int rx_bitset_empty (int size, rx_Bitset set) {
;  int x;
;  RX_subset s;
;  s = set[0];
;  set[0] = 1;
;  for (x = rx_bitset_numb_subsets(size) - 1; !set[x]; --x)
;    ;
;  set[0] = s;
;  return !s;
;}
;
; RUN: opt < %s -passes=adce
; RUN: opt < %s -passes=adce -adce-remove-loops -S | FileCheck %s

define i32 @rx_bitset_empty(i32 %size, ptr %set) {
bb1:
        %reg110 = load i32, ptr %set                ; <i32> [#uses=2]
        store i32 1, ptr %set
        %cast112 = sext i32 %size to i64                ; <i64> [#uses=1]
        %reg113 = add i64 %cast112, 31          ; <i64> [#uses=1]
        %reg114 = lshr i64 %reg113, 5           ; <i64> [#uses=2]
        %cast109 = trunc i64 %reg114 to i32             ; <i32> [#uses=1]
        %reg129 = add i32 %cast109, -1          ; <i32> [#uses=1]
        %reg114-idxcast = trunc i64 %reg114 to i32              ; <i32> [#uses=1]
        %reg114-idxcast-offset = add i32 %reg114-idxcast, 1073741823            ; <i32> [#uses=1]
        %reg114-idxcast-offset.upgrd.1 = zext i32 %reg114-idxcast-offset to i64         ; <i64> [#uses=1]
        %reg124 = getelementptr i32, ptr %set, i64 %reg114-idxcast-offset.upgrd.1           ; <ptr> [#uses=1]
        %reg125 = load i32, ptr %reg124             ; <i32> [#uses=1]
        %cond232 = icmp ne i32 %reg125, 0               ; <i1> [#uses=1]
; CHECK: br label %bb3
        br i1 %cond232, label %bb3, label %bb2

bb2:            ; preds = %bb2, %bb1
        %cann-indvar = phi i32 [ 0, %bb1 ], [ %add1-indvar, %bb2 ]              ; <i32> [#uses=2]
        %reg130-scale = mul i32 %cann-indvar, -1                ; <i32> [#uses=1]
        %reg130 = add i32 %reg130-scale, %reg129                ; <i32> [#uses=1]
        %add1-indvar = add i32 %cann-indvar, 1          ; <i32> [#uses=1]
        %reg130-idxcast = bitcast i32 %reg130 to i32            ; <i32> [#uses=1]
        %reg130-idxcast-offset = add i32 %reg130-idxcast, 1073741823            ; <i32> [#uses=1]
        %reg130-idxcast-offset.upgrd.2 = zext i32 %reg130-idxcast-offset to i64         ; <i64> [#uses=1]
        %reg118 = getelementptr i32, ptr %set, i64 %reg130-idxcast-offset.upgrd.2           ; <ptr> [#uses=1]
        %reg119 = load i32, ptr %reg118             ; <i32> [#uses=1]
        %cond233 = icmp eq i32 %reg119, 0               ; <i1> [#uses=1]
        br i1 %cond233, label %bb2, label %bb3

bb3:            ; preds = %bb2, %bb1
        store i32 %reg110, ptr %set
        %cast126 = zext i32 %reg110 to i64              ; <i64> [#uses=1]
        %reg127 = add i64 %cast126, -1          ; <i64> [#uses=1]
        %reg128 = lshr i64 %reg127, 63          ; <i64> [#uses=1]
        %cast120 = trunc i64 %reg128 to i32             ; <i32> [#uses=1]
        ret i32 %cast120
}

