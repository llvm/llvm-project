; RUN: llc < %s -mtriple=x86_64-- -mcpu=corei7-avx -mattr=-slow-incdec -disable-lsr -pre-RA-sched=source -enable-misched -verify-machineinstrs | FileCheck %s

; Verify that TEST+JE are scheduled together.
; CHECK: test_je
; CHECK: %loop
; CHECK: test
; CHECK-NEXT: je
define void @test_je() {
entry:
  br label %loop

loop:
  %var = phi i32* [ null, %entry ], [ %next.load, %loop1 ], [ %var, %loop2 ]
  %next.ptr = phi i32** [ null, %entry ], [ %next.ptr, %loop1 ], [ %gep, %loop2 ]
  br label %loop1

loop1:
  %cond = icmp eq i32* %var, null
  %next.load = load i32*, i32** %next.ptr
  br i1 %cond, label %loop, label %loop2

loop2:                                           ; preds = %loop1
  %gep = getelementptr inbounds i32*, i32** %next.ptr, i32 1
  store i32* %next.load, i32** undef
  br label %loop
}

; Verify that DEC+JE are scheduled together.
; CHECK: dec_je
; CHECK: %loop1
; CHECK: dec
; CHECK-NEXT: je
define void @dec_je() {
entry:
  br label %loop

loop:
  %var = phi i32 [ 0, %entry ], [ %next.var, %loop1 ], [ %var2, %loop2 ]
  %next.ptr = phi i32** [ null, %entry ], [ %next.ptr, %loop1 ], [ %gep, %loop2 ]
  br label %loop1

loop1:
  %var2 = sub i32 %var, 1
  %cond = icmp eq i32 %var2, 0
  %next.load = load i32*, i32** %next.ptr
  %next.var = load i32, i32* %next.load
  br i1 %cond, label %loop, label %loop2

loop2:
  %gep = getelementptr inbounds i32*, i32** %next.ptr, i32 1
  store i32* %next.load, i32** undef
  br label %loop
}

; DEC+JS should *not* be scheduled together.
; CHECK: dec_js
; CHECK: %loop1
; CHECK: dec
; CHECK: mov
; CHECK: js
define void @dec_js() {
entry:
  br label %loop2a

loop2a:                                           ; preds = %loop1, %body, %entry
  %var = phi i32 [ 0, %entry ], [ %next.var, %loop1 ], [ %var2, %loop2b ]
  %next.ptr = phi i32** [ null, %entry ], [ %next.ptr, %loop1 ], [ %gep, %loop2b ]
  br label %loop1

loop1:                                            ; preds = %loop2a, %loop2b
  %var2 = sub i32 %var, 1
  %cond = icmp slt i32 %var2, 0
  %next.load = load i32*, i32** %next.ptr
  %next.var = load i32, i32* %next.load
  br i1 %cond, label %loop2a, label %loop2b

loop2b:                                           ; preds = %loop1
  %gep = getelementptr inbounds i32*, i32** %next.ptr, i32 1
  store i32* %next.load, i32** undef
  br label %loop2a
}

; Verify that CMP+JB are scheduled together.
; CHECK: cmp_jb
; CHECK: %loop1
; CHECK: cmp
; CHECK-NEXT: jb
define void @cmp_jb(i32 %n) {
entry:
  br label %loop2a

loop2a:                                           ; preds = %loop1, %body, %entry
  %var = phi i32 [ 0, %entry ], [ %next.var, %loop1 ], [ %var2, %loop2b ]
  %next.ptr = phi i32** [ null, %entry ], [ %next.ptr, %loop1 ], [ %gep, %loop2b ]
  br label %loop1

loop1:                                            ; preds = %loop2a, %loop2b
  %var2 = sub i32 %var, 1
  %cond = icmp ult i32 %var2, %n
  %next.load = load i32*, i32** %next.ptr
  %next.var = load i32, i32* %next.load
  br i1 %cond, label %loop2a, label %loop2b

loop2b:                                           ; preds = %loop1
  %gep = getelementptr inbounds i32*, i32** %next.ptr, i32 1
  store i32* %next.load, i32** undef
  br label %loop2a
}
