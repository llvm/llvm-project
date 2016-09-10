; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s

declare void @bar()

define void @test1(i32* nocapture %X) nounwind minsize {
entry:
  %tmp1 = load i32, i32* %X, align 4
  %and = and i32 %tmp1, 255
  %cmp = icmp eq i32 %and, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
; CHECK-LABEL: test1:
; CHECK: cmpb $47, (%{{rdi|rcx}})
}

define void @test2(i32 %X) nounwind minsize {
entry:
  %and = and i32 %X, 255
  %cmp = icmp eq i32 %and, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
; CHECK-LABEL: test2:
; CHECK: cmpb $47, %{{dil|cl}}
}

define void @test3(i32 %X) nounwind minsize {
entry:
  %and = and i32 %X, 255
  %cmp = icmp eq i32 %and, 255
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
; CHECK-LABEL: test3:
; CHECK: cmpb $-1, %{{dil|cl}}
}

; PR16083
define i1 @test4(i64 %a, i32 %b) {
entry:
  %tobool = icmp ne i32 %b, 0
  br i1 %tobool, label %lor.end, label %lor.rhs

lor.rhs:                                          ; preds = %entry
  %and = and i64 0, %a
  %tobool1 = icmp ne i64 %and, 0
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %p = phi i1 [ true, %entry ], [ %tobool1, %lor.rhs ]
  ret i1 %p
}

@x = global { i8, i8, i8, i8, i8, i8, i8, i8 } { i8 1, i8 0, i8 0, i8 0, i8 1, i8 0, i8 0, i8 1 }, align 4

; PR16551
define void @test5(i32 %X) nounwind minsize {
entry:
  %bf.load = load i56, i56* bitcast ({ i8, i8, i8, i8, i8, i8, i8, i8 }* @x to i56*), align 4
  %bf.lshr = lshr i56 %bf.load, 32
  %bf.cast = trunc i56 %bf.lshr to i32
  %cmp = icmp ne i32 %bf.cast, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void

; CHECK-LABEL: test5:
; CHECK-NOT: cmpl $1,{{.*}}x+4
; CHECK: ret
}

; CHECK-LABEL: test2_1:
; CHECK: movzbl
; CHECK: cmpl $256
; CHECK: je bar
define void @test2_1(i32 %X) nounwind minsize {
entry:
  %and = and i32 %X, 255
  %cmp = icmp eq i32 %and, 256
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_1:
; CHECK: cmpb $1, %{{dil|cl}}
define void @test_sext_i8_icmp_1(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_47:
; CHECK: cmpb $47, %{{dil|cl}}
define void @test_sext_i8_icmp_47(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, 47
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_127:
; CHECK: cmpb $127, %{{dil|cl}}
define void @test_sext_i8_icmp_127(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, 127
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_neg1:
; CHECK: cmpb $-1, %{{dil|cl}}
define void @test_sext_i8_icmp_neg1(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, -1
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_neg2:
; CHECK: cmpb $-2, %{{dil|cl}}
define void @test_sext_i8_icmp_neg2(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, -2
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_neg127:
; CHECK: cmpb $-127, %{{dil|cl}}
define void @test_sext_i8_icmp_neg127(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, -127
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_neg128:
; CHECK: cmpb $-128, %{{dil|cl}}
define void @test_sext_i8_icmp_neg128(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, -128
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}

; CHECK-LABEL: test_sext_i8_icmp_255:
; CHECK: movb $1,
; CHECK: testb
; CHECK: je bar
define void @test_sext_i8_icmp_255(i8 %x) nounwind minsize {
entry:
  %sext = sext i8 %x to i32
  %cmp = icmp eq i32 %sext, 255
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @bar() nounwind
  br label %if.end

if.end:
  ret void
}
