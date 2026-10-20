; RUN: opt -S -dxil-op-lower -dxil-translate-metadata -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; This test make sure LLVM metadata is being translated into DXIL.


; CHECK: define i32 @test_branch(i32 %X)
; CHECK-NOT: hlsl.controlflow.hint
; CHECK: br i1 %cmp, label %if.then, label %if.else, !dx.controlflow.hints [[HINT_BRANCH:![0-9]+]]
define i32 @test_branch(i32 %X) {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !0

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}


; CHECK: define i32 @test_flatten(i32 %X)
; CHECK-NOT: hlsl.controlflow.hint
; CHECK: br i1 %cmp, label %if.then, label %if.else, !dx.controlflow.hints [[HINT_FLATTEN:![0-9]+]]
define i32 @test_flatten(i32 %X) {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else, !hlsl.controlflow.hint !1

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}


; CHECK: define i32 @test_no_attr(i32 %X)
; CHECK-NOT: hlsl.controlflow.hint
; CHECK-NOT: !dx.controlflow.hints
define i32 @test_no_attr(i32 %X) {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %resp, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %resp, align 4
  ret i32 %3
}

; CHECK: define i32 @flatten_switch(i32 %X)
; CHECK-NOT: hlsl.controlflow.hint
; CHECK:      switch i32 %0, label %sw.epilog [
; CHECK-NEXT:   i32 0, label %sw.bb
; CHECK-NEXT:   i32 1, label %sw.bb1
; CHECK-NEXT:   i32 2, label %sw.bb2
; CHECK-NEXT: ], !dx.controlflow.hints [[HINT_FLATTEN]]
define i32 @flatten_switch(i32 %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ], !hlsl.controlflow.hint !1

sw.bb:                                            ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %3 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %resp, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %X.addr, align 4
  %5 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, ptr %resp, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  %6 = load i32, ptr %resp, align 4
  ret i32 %6
}


; CHECK: define i32 @branch_switch(i32 %X)
; CHECK-NOT: hlsl.controlflow.hint
; CHECK:      switch i32 %0, label %sw.epilog [
; CHECK-NEXT:   i32 0, label %sw.bb
; CHECK-NEXT:   i32 1, label %sw.bb1
; CHECK-NEXT:   i32 2, label %sw.bb2
; CHECK-NEXT: ], !dx.controlflow.hints [[HINT_BRANCH]]
define i32 @branch_switch(i32 %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ], !hlsl.controlflow.hint !0

sw.bb:                                            ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %3 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %resp, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %X.addr, align 4
  %5 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, ptr %resp, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  %6 = load i32, ptr %resp, align 4
  ret i32 %6
}


; CHECK: define i32 @no_attr_switch(i32 %X)
; CHECK-NOT: hlsl.controlflow.hint
; CHECK-NOT: !dx.controlflow.hints
define i32 @no_attr_switch(i32 %X) #0 {
entry:
  %X.addr = alloca i32, align 4
  %resp = alloca i32, align 4
  store i32 %X, ptr %X.addr, align 4
  %0 = load i32, ptr %X.addr, align 4
  switch i32 %0, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
  ]

sw.bb:                                            ; preds = %entry
  %1 = load i32, ptr %X.addr, align 4
  %sub = sub nsw i32 0, %1
  store i32 %sub, ptr %resp, align 4
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %2 = load i32, ptr %X.addr, align 4
  %3 = load i32, ptr %X.addr, align 4
  %add = add nsw i32 %2, %3
  store i32 %add, ptr %resp, align 4
  br label %sw.epilog

sw.bb2:                                           ; preds = %entry
  %4 = load i32, ptr %X.addr, align 4
  %5 = load i32, ptr %X.addr, align 4
  %mul = mul nsw i32 %4, %5
  store i32 %mul, ptr %resp, align 4
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb2, %sw.bb1, %sw.bb
  %6 = load i32, ptr %resp, align 4
  ret i32 %6
}

; CHECK-NOT: hlsl.controlflow.hint
; CHECK: [[HINT_BRANCH]] = !{!"dx.controlflow.hints", i32 1}
; CHECK: [[HINT_FLATTEN]] = !{!"dx.controlflow.hints", i32 2}
!0 = !{!"hlsl.controlflow.hint", i32 1}
!1 = !{!"hlsl.controlflow.hint", i32 2}
