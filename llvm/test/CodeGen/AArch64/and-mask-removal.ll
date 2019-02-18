; RUN: llc -mtriple=arm64-apple-darwin  < %s  | FileCheck %s

@board = common global [400 x i8] zeroinitializer, align 1
@next_string = common global i32 0, align 4
@string_number = common global [400 x i32] zeroinitializer, align 4

; Function Attrs: nounwind ssp
define void @new_position(i32 %pos) {
entry:
  %idxprom = sext i32 %pos to i64
  %arrayidx = getelementptr inbounds [400 x i8], [400 x i8]* @board, i64 0, i64 %idxprom
  %tmp = load i8, i8* %arrayidx, align 1
  %.off = add i8 %tmp, -1
  %switch = icmp ult i8 %.off, 2
  br i1 %switch, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %tmp1 = load i32, i32* @next_string, align 4
  %arrayidx8 = getelementptr inbounds [400 x i32], [400 x i32]* @string_number, i64 0, i64 %idxprom
  store i32 %tmp1, i32* %arrayidx8, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
; CHECK-LABEL: new_position
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test8_0(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 74
  %1 = icmp ult i8 %0, -20
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_0
; CHECK: and
; CHECK: ret
}

define zeroext i1 @test8_1(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 246
  %1 = icmp uge i8 %0, 90
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_1
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test8_2(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 227
  %1 = icmp ne i8 %0, 179
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_2
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test8_3(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 201
  %1 = icmp eq i8 %0, 154
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_3
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test8_4(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, -79
  %1 = icmp ne i8 %0, -40
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_4
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test8_5(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 133
  %1 = icmp uge i8 %0, -105
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_5
; CHECK: and
; CHECK: ret
}

define zeroext i1 @test8_6(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, -58
  %1 = icmp uge i8 %0, 155
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_6
; CHECK: and
; CHECK: ret
}

define zeroext i1 @test8_7(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 225
  %1 = icmp ult i8 %0, 124
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_7
; CHECK-NOT: and
; CHECK: ret
}



define zeroext i1 @test8_8(i8 zeroext %x)  align 2 {
entry:
  %0 = add i8 %x, 190
  %1 = icmp uge i8 %0, 1
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test8_8
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test16_0(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, -46989
  %1 = icmp ne i16 %0, -41903
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_0
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test16_2(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, 16882
  %1 = icmp ule i16 %0, -24837
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_2
; CHECK: mov	[[CST:w[0-9]+]], #16882
; CHECK: add	[[ADD:w[0-9]+]], w0, [[CST]]
; CHECK: cmp	{{.*}}, [[ADD]], uxth
; CHECK: ret
}

define zeroext i1 @test16_3(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, 29283
  %1 = icmp ne i16 %0, 16947
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_3
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test16_4(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, -35551
  %1 = icmp uge i16 %0, 15677
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_4
; CHECK: mov	[[CST:w[0-9]+]], #29985
; CHECK: add	[[ADD:w[0-9]+]], w0, [[CST]]
; CHECK: cmp	{{.*}}, [[ADD]], uxth
; CHECK: ret
}

define zeroext i1 @test16_5(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, -25214
  %1 = icmp ne i16 %0, -1932
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_5
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test16_6(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, -32194
  %1 = icmp uge i16 %0, -41215
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_6
; CHECK-NOT: and
; CHECK: ret
}

define zeroext i1 @test16_7(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, 9272
  %1 = icmp uge i16 %0, -42916
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_7
; CHECK: mov	[[CST:w[0-9]+]], #9272
; CHECK: add	[[ADD:w[0-9]+]], w0, [[CST]]
; CHECK: cmp	{{.*}}, [[ADD]], uxth
; CHECK: ret
}

define zeroext i1 @test16_8(i16 zeroext %x)  align 2 {
entry:
  %0 = add i16 %x, -63749
  %1 = icmp ne i16 %0, 6706
  br i1 %1, label %ret_true, label %ret_false
ret_false:
  ret i1 false
ret_true:
  ret i1 true
; CHECK-LABEL: test16_8
; CHECK-NOT: and
; CHECK: ret
}

