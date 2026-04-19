; RUN: opt -S -passes=early-cse < %s | FileCheck %s

%struct.anon = type { i32, i32 }

declare void @foo() #2

@d = global %struct.anon zeroinitializer, align 4
@c = global %struct.anon zeroinitializer, align 4

define i32 @test_different() #0 {
; CHECK-LABEL: @test_different
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 false, label %{{.+}}, label %{{.+}}
entry:
  %cmp = icmp eq ptr @c, @d
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

define i32 @test_different_first() #0 {
; CHECK-LABEL: @test_different_first
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 false, label %{{.+}}, label %{{.+}}
entry:
  %cmp = icmp eq ptr @c, @d
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

define i32 @test_different_offset() #0 {
; CHECK-LABEL: @test_different_offset
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 false, label %{{.+}}, label %{{.+}}
entry:
  %cmp = icmp eq ptr @c, getelementptr inbounds nuw (i8, ptr @d, i64 4)
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

define i32 @test_same() #0 {
; CHECK-LABEL: @test_same
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 true, label %{{.+}}, label %{{.+}}
entry:
  %cmp = icmp eq ptr @c, @c
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

define i32 @test_same_first() #0 {
; CHECK-LABEL: @test_same_first
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 true, label %{{.+}}, label %{{.+}}
entry:
  %cmp = icmp eq ptr @c, @c
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}

define i32 @test_same_offset() #0 {
; CHECK-LABEL: @test_same_offset
; CHECK-NEXT: entry:
; CHECK-NEXT: br i1 false, label %{{.+}}, label %{{.+}}
entry:
  %cmp = icmp eq ptr @c, getelementptr inbounds nuw (i8, ptr @c, i64 4)
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @foo()
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret i32 0
}
