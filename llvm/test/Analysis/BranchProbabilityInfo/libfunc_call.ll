; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare i32 @strcmp(ptr, ptr)
declare i32 @strncmp(ptr, ptr, i32)
declare i32 @strcasecmp(ptr, ptr)
declare i32 @strncasecmp(ptr, ptr, i32)
declare i32 @memcmp(ptr, ptr)
declare i32 @bcmp(ptr, ptr)
declare i32 @nonstrcmp(ptr, ptr)


; Check that the result of strcmp is considered more likely to be nonzero than
; zero, and equally likely to be (nonzero) positive or negative.

define i32 @test_strcmp_eq(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strcmp_eq'
entry:
  %val = call i32 @strcmp(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge %entry -> %else probability is 0x50000000 / 0x80000000 = 62.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_strcmp_eq5(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strcmp_eq5'
entry:
  %val = call i32 @strcmp(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 5
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge %entry -> %else probability is 0x50000000 / 0x80000000 = 62.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_strcmp_ne(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strcmp_ne'
entry:
  %val = call i32 @strcmp(ptr %p, ptr %q)
  %cond = icmp ne i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge %entry -> %else probability is 0x30000000 / 0x80000000 = 37.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_strcmp_sgt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strcmp_sgt'
entry:
  %val = call i32 @strcmp(ptr %p, ptr %q)
  %cond = icmp sgt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %else probability is 0x40000000 / 0x80000000 = 50.00%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_strcmp_slt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strcmp_slt'
entry:
  %val = call i32 @strcmp(ptr %p, ptr %q)
  %cond = icmp slt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %else probability is 0x40000000 / 0x80000000 = 50.00%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}


; Similarly check other library functions that have the same behaviour

define i32 @test_strncmp_sgt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strncmp_sgt'
entry:
  %val = call i32 @strncmp(ptr %p, ptr %q, i32 4)
  %cond = icmp sgt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %else probability is 0x40000000 / 0x80000000 = 50.00%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_strcasecmp_sgt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strcasecmp_sgt'
entry:
  %val = call i32 @strcasecmp(ptr %p, ptr %q)
  %cond = icmp sgt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %else probability is 0x40000000 / 0x80000000 = 50.00%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_strncasecmp_sgt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_strncasecmp_sgt'
entry:
  %val = call i32 @strncasecmp(ptr %p, ptr %q, i32 4)
  %cond = icmp sgt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %else probability is 0x40000000 / 0x80000000 = 50.00%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_memcmp_sgt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_memcmp_sgt'
entry:
  %val = call i32 @memcmp(ptr %p, ptr %q)
  %cond = icmp sgt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x40000000 / 0x80000000 = 50.00%
; CHECK: edge %entry -> %else probability is 0x40000000 / 0x80000000 = 50.00%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}


; Check that for the result of a call to a non-library function the default
; heuristic is applied, i.e. positive more likely than negative, nonzero more
; likely than zero.

define i32 @test_nonstrcmp_eq(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_nonstrcmp_eq'
entry:
  %val = call i32 @nonstrcmp(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge %entry -> %else probability is 0x50000000 / 0x80000000 = 62.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_nonstrcmp_ne(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_nonstrcmp_ne'
entry:
  %val = call i32 @nonstrcmp(ptr %p, ptr %q)
  %cond = icmp ne i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge %entry -> %else probability is 0x30000000 / 0x80000000 = 37.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_nonstrcmp_sgt(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_nonstrcmp_sgt'
entry:
  %val = call i32 @nonstrcmp(ptr %p, ptr %q)
  %cond = icmp sgt i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge %entry -> %else probability is 0x30000000 / 0x80000000 = 37.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}


define i32 @test_bcmp_eq(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_bcmp_eq'
entry:
  %val = call i32 @bcmp(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge %entry -> %else probability is 0x50000000 / 0x80000000 = 62.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}

define i32 @test_bcmp_eq5(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_bcmp_eq5'
entry:
  %val = call i32 @bcmp(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 5
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x30000000 / 0x80000000 = 37.50%
; CHECK: edge %entry -> %else probability is 0x50000000 / 0x80000000 = 62.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}



define i32 @test_bcmp_ne(ptr %p, ptr %q) {
; CHECK: Printing analysis {{.*}} for function 'test_bcmp_ne'
entry:
  %val = call i32 @bcmp(ptr %p, ptr %q)
  %cond = icmp ne i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x50000000 / 0x80000000 = 62.50%
; CHECK: edge %entry -> %else probability is 0x30000000 / 0x80000000 = 37.50%

then:
  br label %exit
; CHECK: edge %then -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

else:
  br label %exit
; CHECK: edge %else -> %exit probability is 0x80000000 / 0x80000000 = 100.00% [HOT edge]

exit:
  %result = phi i32 [ 0, %then ], [ 1, %else ]
  ret i32 %result
}
