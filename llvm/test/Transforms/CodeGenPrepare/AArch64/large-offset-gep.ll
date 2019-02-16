; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

%struct_type = type { [10000 x i32], i32, i32 }

define void @test1(%struct_type** %s, i32 %n) {
; CHECK-LABEL: test1
entry:
  %struct = load %struct_type*, %struct_type** %s
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
; CHECK:     mov     w{{[0-9]+}}, #40000
; CHECK-NOT: mov     w{{[0-9]+}}, #40004
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
; CHECK:     str      w{{[0-9]+}}, [x{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
}

define void @test2(%struct_type* %struct, i32 %n) {
; CHECK-LABEL: test2
entry:
  %cmp = icmp eq %struct_type* %struct, null
  br i1 %cmp, label %while_end, label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
; CHECK:     mov     w{{[0-9]+}}, #40000
; CHECK-NOT: mov     w{{[0-9]+}}, #40004
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp1 = icmp slt i32 %phi, %n
  br i1 %cmp1, label %while_body, label %while_end

while_body:
; CHECK:     str      w{{[0-9]+}}, [x{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
}

define void @test3(%struct_type* %s1, %struct_type* %s2, i1 %cond, i32 %n) {
; CHECK-LABEL: test3
entry:
  br i1 %cond, label %if_true, label %if_end

if_true:
  br label %if_end

if_end:
  %struct = phi %struct_type* [ %s1, %entry ], [ %s2, %if_true ]
  %cmp = icmp eq %struct_type* %struct, null
  br i1 %cmp, label %while_end, label %while_cond

while_cond:
  %phi = phi i32 [ 0, %if_end ], [ %i, %while_body ]
; CHECK:     mov     w{{[0-9]+}}, #40000
; CHECK-NOT: mov     w{{[0-9]+}}, #40004
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  %cmp1 = icmp slt i32 %phi, %n
  br i1 %cmp1, label %while_body, label %while_end

while_body:
; CHECK:     str      w{{[0-9]+}}, [x{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
}

declare %struct_type* @foo()
declare void @foo2()

define void @test4(i32 %n) personality i32 (...)* @__FrameHandler {
; CHECK-LABEL: test4
entry:
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
  %struct = invoke %struct_type* @foo() to label %while_cond_x unwind label %cleanup

while_cond_x:
; CHECK:     mov     w{{[0-9]+}}, #40000
; CHECK-NOT: mov     w{{[0-9]+}}, #40004
  %gep0 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 1
  %gep1 = getelementptr %struct_type, %struct_type* %struct, i64 0, i32 2
  store i32 0, i32* %gep0
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
; CHECK:     str      w{{[0-9]+}}, [x{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void

cleanup:
  %x10 = landingpad { i8*, i32 }
          cleanup
  call void @foo2()
  resume { i8*, i32 } %x10
}

declare i32 @__FrameHandler(...)

define void @test5([65536 x i32]** %s, i32 %n) {
; CHECK-LABEL: test5
entry:
  %struct = load [65536 x i32]*, [65536 x i32]** %s
  br label %while_cond

while_cond:
  %phi = phi i32 [ 0, %entry ], [ %i, %while_body ]
; CHECK:     mov     w{{[0-9]+}}, #14464
; CHECK-NOT: mov     w{{[0-9]+}}, #14468
  %gep0 = getelementptr [65536 x i32], [65536 x i32]* %struct, i64 0, i32 20000
  %gep1 = getelementptr [65536 x i32], [65536 x i32]* %struct, i64 0, i32 20001
  %cmp = icmp slt i32 %phi, %n
  br i1 %cmp, label %while_body, label %while_end

while_body:
; CHECK:     str      w{{[0-9]+}}, [x{{[0-9]+}}, #4]
  %i = add i32 %phi, 1
  store i32 %i, i32* %gep0
  store i32 %phi, i32* %gep1
  br label %while_cond

while_end:
  ret void
}

declare i8* @llvm.strip.invariant.group.p0i8(i8*)

define void @test_invariant_group(i32) {
; CHECK-LABEL: test_invariant_group
  br i1 undef, label %8, label %7

; <label>:2:                                      ; preds = %8, %2
  br i1 undef, label %2, label %7

; <label>:3:                                      ; preds = %8
  %4 = getelementptr inbounds i8, i8* %9, i32 40000
  %5 = bitcast i8* %4 to i64*
  br i1 undef, label %7, label %6

; <label>:6:                                      ; preds = %3
  store i64 1, i64* %5, align 8
  br label %7

; <label>:7:                                      ; preds = %6, %3, %2, %1
  ret void

; <label>:8:                                      ; preds = %1
  %9 = call i8* @llvm.strip.invariant.group.p0i8(i8* nonnull undef)
  %10 = icmp eq i32 %0, 0
  br i1 %10, label %3, label %2
}

