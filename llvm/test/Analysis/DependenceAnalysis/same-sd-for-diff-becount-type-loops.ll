; RUN: opt < %s -disable-output "-passes=print<da>" -aa-pipeline=basic-aa 2>&1 | FileCheck %s

define void @f1() {
; CHECK-LABEL: 'f1'
; CHECK-NEXT:  Src:  store i32 0, ptr null, align 4 --> Dst:  store i32 0, ptr null, align 4
; CHECK-NEXT:    da analyze - consistent output [S]!
; CHECK-NEXT:  Src:  store i32 0, ptr null, align 4 --> Dst:  %2 = load i32, ptr null, align 4
; CHECK-NEXT:    da analyze - consistent flow [|<]!
; CHECK-NEXT:  Src:  %2 = load i32, ptr null, align 4 --> Dst:  %2 = load i32, ptr null, align 4
; CHECK-NEXT:    da analyze - consistent input [S]!
;
entry:
  br label %for.1.header

for.1.header:                                     ; preds = %for.2.end, %entry
  br label %for.1.body

for.1.body:                                       ; preds = %for.1.body, %whiledo
  %0 = phi i32 [ 0, %for.1.header ], [ 1, %for.1.body ]
  store i32 0, ptr null, align 4
  %1 = icmp ult i32 %0, 1
  br i1 %1, label %for.1.body, label %for.1.end

for.1.end:                                        ; preds = %for.1.body
  br label %for.2.body

for.2.body:                                       ; preds = %for.2.body, %for.1.end
  %2 = load i32, ptr null, align 4
  br i1 false, label %for.2.body, label %exit

exit:                                             ; preds = %for.2.body
  ret void
}

define void @f2() {
; CHECK-LABEL: 'f2'
; CHECK-NEXT:  Src:  store i32 0, ptr null, align 4 --> Dst:  store i32 0, ptr null, align 4
; CHECK-NEXT:    da analyze - consistent output [S]!
; CHECK-NEXT:  Src:  store i32 0, ptr null, align 4 --> Dst:  %3 = load i32, ptr null, align 4
; CHECK-NEXT:    da analyze - flow [|<] / assuming 1 loop level(s) fused:  [S|<]!
; CHECK-NEXT:  Src:  %3 = load i32, ptr null, align 4 --> Dst:  %3 = load i32, ptr null, align 4
; CHECK-NEXT:    da analyze - consistent input [S]!
;
entry:
  br label %for.1.header

for.1.header:                                     ; preds = %for.2.end, %entry
  br label %for.1.body

for.1.body:                                       ; preds = %for.1.body, %whiledo
  %0 = phi i32 [ 0, %for.1.header ], [ 1, %for.1.body ]
  store i32 0, ptr null, align 4
  %1 = icmp ult i32 %0, 1
  br i1 %1, label %for.1.body, label %for.1.end

for.1.end:                                        ; preds = %for.1.body
  br label %for.2.body

for.2.body:                                       ; preds = %for.2.body, %for.1.end
  %2 = phi i64 [ 0, %for.1.end ], [ %4, %for.2.body ]
  %3 = load i32, ptr null, align 4
  %4 = add nuw nsw i64 %2, 1
  %5 = icmp ult i64 %4, 2
  br i1 %5, label %for.2.body, label %exit

exit:                                             ; preds = %for.2.body
  ret void
}
