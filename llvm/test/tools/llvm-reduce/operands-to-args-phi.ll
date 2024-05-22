; RUN: llvm-reduce --abort-on-invalid-reduction %s -o %t --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck -check-prefix=RESULT %s < %t

; Make sure an invalid reduction isn't hit from only replacing one of
; the values when a predecessor is listed multiple times in a phi.


; RESULT: define void @fn(i1 %cmp11, i8 %p.2, i8 %k.2, i8 %p.4, i8 %k.0, i8 %k.4, i8 %p.0, i1 %tobool30, i8 %spec.select, i8 %spec.select1)

; RESULT:     for.cond:
; RESULT-NEXT:  %p.01 = phi i8 [ 0, %entry ], [ %p.2, %for.inc ]
; RESULT-NEXT:  %k.02 = phi i8 [ 0, %entry ], [ %k.2, %for.inc ]

; RESULT:     if.end26:
; RESULT-NEXT:  %p.1 = phi i8 [ %p.4, %for.cond35 ], [ %k.0, %if.end ]
; RESULT-NEXT:  %k.1 = phi i8 [ %k.4, %for.cond35 ], [ 0, %if.end ]

; RESULT:     for.inc:
; RESULT-NEXT:  %p.26 = phi i8 [ %spec.select, %if.end26 ], [ poison, %if.then13 ], [ %spec.select, %if.end26 ]
; RESULT-NEXT:  %k.27 = phi i8 [ %spec.select1, %if.end26 ], [ 0, %if.then13 ], [ %spec.select1, %if.end26 ]


; RESULT:      for.cond35:
; RESULT-NEXT:   %p.48 = phi i8 [ 0, %if.then ], [ %k.0, %if.then13 ]
; RESULT-NEXT:   %k.49 = phi i8 [ %k.0, %if.then ], [ 0, %if.then13 ]

define void @fn(i1 %cmp11) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.0 = phi i8 [ 0, %entry ], [ %p.2, %for.inc ]
  %k.0 = phi i8 [ 0, %entry ], [ %k.2, %for.inc ]
  br i1 %cmp11, label %if.then, label %if.end

if.then:                                          ; preds = %for.cond
  br label %for.cond35

if.end:                                           ; preds = %for.cond
  br i1 %cmp11, label %if.then13, label %if.end26

if.then13:                                        ; preds = %if.end
  br i1 %cmp11, label %for.inc, label %for.cond35

if.end26:                                         ; preds = %for.cond35, %if.end
  ; INTERESTING: %p.1 = phi i8
  ; INTERESTING: %k.1 = phi i8

  %p.1 = phi i8 [ %p.4, %for.cond35 ], [ %k.0, %if.end ]
  %k.1 = phi i8 [ %k.4, %for.cond35 ], [ 0, %if.end ]
  %tobool30 = icmp ne i8 %p.0, 0
  %spec.select = select i1 false, i8 0, i8 %p.1
  %spec.select1 = select i1 %tobool30, i8 %k.1, i8 0
  br i1 false, label %for.inc, label %for.inc

; INTERESTING: {{^}}for.inc:
; INTERESTING: phi i8
; INTERESTING-SAME: [ %spec.select{{[0-9]*}}, %if.end26 ]

; INTERESTING: phi i8
; INTERESTING-SAME: [ %spec.select{{[0-9]*}}, %if.end26 ]
for.inc:                                          ; preds = %if.end26, %if.end26, %if.then13
  %p.2 = phi i8 [ %spec.select, %if.end26 ], [ poison, %if.then13 ], [ %spec.select, %if.end26 ]
  %k.2 = phi i8 [ %spec.select1, %if.end26 ], [ 0, %if.then13 ], [ %spec.select1, %if.end26 ]
  %0 = load i32, ptr null, align 4
  br label %for.cond

for.cond35:                                       ; preds = %if.then13, %if.then
  %p.4 = phi i8 [ 0, %if.then ], [ %k.0, %if.then13 ]
  %k.4 = phi i8 [ %k.0, %if.then ], [ 0, %if.then13 ]
  %tobool36 = icmp eq i32 0, 0
  br label %if.end26
}
