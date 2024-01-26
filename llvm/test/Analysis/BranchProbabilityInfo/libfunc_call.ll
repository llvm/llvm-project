; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

declare i32 @strcmp(ptr, ptr)
declare i32 @strncmp(ptr, ptr, i32)
declare i32 @strcasecmp(ptr, ptr)
declare i32 @strncasecmp(ptr, ptr, i32)
declare i32 @memcmp(ptr, ptr)
declare i32 @bcmp(ptr, ptr)
declare i32 @nonstrcmp(ptr, ptr)
declare i64 @fread(ptr, i64, i64, ptr)
declare i64 @fwrite(ptr, i64, i64, ptr)
declare i64 @read(i32, ptr, i64)
declare i64 @write(i32, ptr, i64)
declare i32 @chmod(ptr, i32)
declare i32 @chown(ptr, i32, i32)
declare i32 @closedir(ptr)
declare i32 @fclose(ptr)
declare i32 @ferror(ptr)
declare i32 @fflush(ptr)
declare i32 @fseek(ptr, i64, i32)
declare i32 @fseeko(ptr, i64, i32)
declare i32 @fstat(i32, ptr)
declare i32 @fstatvfs(i32, ptr)
declare i32 @ftrylockfile(ptr)
declare i32 @lchown(ptr)
declare i32 @lstat(ptr, ptr)
declare i32 @mkdir(ptr, i32)
declare i32 @remove(ptr)
declare i32 @rename(ptr, ptr)
declare i32 @rmdir(ptr)
declare i32 @setvbuf(ptr, ptr, i32, i64)
declare i32 @stat(ptr, ptr)
declare i32 @statvfs(ptr, ptr)
declare i32 @unlink(ptr)
declare i32 @unsetenv(ptr)
declare i32 @utime(ptr, ptr)
declare i32 @utimes(ptr, ptr)


; Check that the result of strcmp is considered more likely to be nonzero than
; zero, and equally likely to be (nonzero) positive or negative.

define i32 @test_strcmp_eq(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strcmp_eq'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strcmp_eq5'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strcmp_ne'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strcmp_sgt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strcmp_slt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strncmp_sgt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strcasecmp_sgt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_strncasecmp_sgt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_memcmp_sgt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_nonstrcmp_eq'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_nonstrcmp_ne'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_nonstrcmp_sgt'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_bcmp_eq'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_bcmp_eq5'
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
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_bcmp_ne'
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

define i32 @test_fread_eq_zero(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_eq_zero'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp eq i64 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_fread_ne_zero(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_ne_zero'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp ne i64 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fread_slt_one(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_slt_one'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp slt i64 %val, 1
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_fread_eq_count(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_eq_count'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp eq i64 %val, %count
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fread_ne_count(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_ne_count'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp ne i64 %val, %count
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_fread_eq_const_count(i64 %size, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_eq_const_count'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 10, ptr %q)
  %cond = icmp eq i64 %val, 10
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fread_ne_const_count(i64 %size, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_ne_const_count'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 10, ptr %q)
  %cond = icmp ne i64 %val, 10
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_fread_slt_count(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_slt_count'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp slt i64 %val, %count
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_fread_slt_const_count(i64 %size, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fread_slt_const_count'
entry:
  %val = call i64 @fread(ptr %p, i64 %size, i64 10, ptr %q)
  %cond = icmp slt i64 %val, 10
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

; fwrite only tests eq predicate
define i32 @test_fwrite_eq_zero(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fwrite_eq_zero'
entry:
  %val = call i64 @fwrite(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp eq i64 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_fwrite_eq_count(i64 %size, i64 %count, ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fwrite_eq_count'
entry:
  %val = call i64 @fwrite(ptr %p, i64 %size, i64 %count, ptr %q)
  %cond = icmp eq i64 %val, %count
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

; read only tests eq predicate
define i32 @test_read_eq_minus_one(i32 %handle, i64 %count, ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_read_eq_minus_one'
entry:
  %val = call i64 @read(i32 %handle, ptr %p, i64 %count)
  %cond = icmp eq i64 %val, -1
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_read_eq_count(i32 %handle, i64 %count, ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_read_eq_count'
entry:
  %val = call i64 @read(i32 %handle, ptr %p, i64 %count)
  %cond = icmp eq i64 %val, %count
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

; write only tests eq predicate
define i32 @test_write_eq_minus_one(i32 %handle, i64 %count, ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_write_eq_minus_one'
entry:
  %val = call i64 @write(i32 %handle, ptr %p, i64 %count)
  %cond = icmp eq i64 %val, -1
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_write_eq_count(i32 %handle, i64 %count, ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_write_eq_count'
entry:
  %val = call i64 @write(i32 %handle, ptr %p, i64 %count)
  %cond = icmp eq i64 %val, %count
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_chmod_eq_zero(ptr %p, i32 %mod) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_chmod_eq_zero'
entry:
  %val = call i32 @chmod(ptr %p, i32 %mod)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_chmod_ne_zero(ptr %p, i32 %mod) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_chmod_ne_zero'
entry:
  %val = call i32 @chmod(ptr %p, i32 %mod)
  %cond = icmp ne i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x0020c49c / 0x80000000 = 0.10%
; CHECK: edge %entry -> %else probability is 0x7fdf3b64 / 0x80000000 = 99.90%

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

define i32 @test_chown_eq_zero(ptr %p, i32 %owner, i32 %group) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_chown_eq_zero'
entry:
  %val = call i32 @chown(ptr %p, i32 %owner, i32 %group)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_closedir_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_closedir_eq_zero'
entry:
  %val = call i32 @closedir(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fclose_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fclose_eq_zero'
entry:
  %val = call i32 @fclose(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_ferror_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_ferror_eq_zero'
entry:
  %val = call i32 @ferror(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fflush_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fflush_eq_zero'
entry:
  %val = call i32 @fflush(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fseek_eq_zero(ptr %p, i64 %offset, i32 %whence) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fseek_eq_zero'
entry:
  %val = call i32 @fseek(ptr %p, i64 %offset, i32 %whence)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fseeko_eq_zero(ptr %p, i64 %offset, i32 %whence) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fseeko_eq_zero'
entry:
  %val = call i32 @fseeko(ptr %p, i64 %offset, i32 %whence)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fstat_eq_zero(i32 %filedes, ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fstat_eq_zero'
entry:
  %val = call i32 @fstat(i32 %filedes, ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_fstatvfs_eq_zero(i32 %filedes, ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_fstatvfs_eq_zero'
entry:
  %val = call i32 @fstatvfs(i32 %filedes, ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_ftrylockfile_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_ftrylockfile_eq_zero'
entry:
  %val = call i32 @ftrylockfile(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_lchown_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_lchown_eq_zero'
entry:
  %val = call i32 @lchown(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_lstat_eq_zero(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_lstat_eq_zero'
entry:
  %val = call i32 @lstat(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_mkdir_eq_zero(ptr %p, i32 %mode) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_mkdir_eq_zero'
entry:
  %val = call i32 @mkdir(ptr %p, i32 %mode)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_remove_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_remove_eq_zero'
entry:
  %val = call i32 @remove(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_rename_eq_zero(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_rename_eq_zero'
entry:
  %val = call i32 @rename(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_rmdir_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_rmdir_eq_zero'
entry:
  %val = call i32 @rmdir(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_setvbuf_eq_zero(ptr %p, ptr %q, i32 %mode, i64 %size) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_setvbuf_eq_zero'
entry:
  %val = call i32 @setvbuf(ptr %p, ptr %q, i32 %mode, i64 %size)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_stat_eq_zero(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_stat_eq_zero'
entry:
  %val = call i32 @stat(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_statvfs_eq_zero(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_statvfs_eq_zero'
entry:
  %val = call i32 @statvfs(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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


define i32 @test_unlink_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_unlink_eq_zero'
entry:
  %val = call i32 @unlink(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_unsetenv_eq_zero(ptr %p) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_unsetenv_eq_zero'
entry:
  %val = call i32 @unsetenv(ptr %p)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_utime_eq_zero(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_utime_eq_zero'
entry:
  %val = call i32 @utime(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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

define i32 @test_utimes_eq_zero(ptr %p, ptr %q) {
; CHECK-LABEL: Printing analysis {{.*}} for function 'test_utimes_eq_zero'
entry:
  %val = call i32 @utimes(ptr %p, ptr %q)
  %cond = icmp eq i32 %val, 0
  br i1 %cond, label %then, label %else
; CHECK: edge %entry -> %then probability is 0x7fdf3b64 / 0x80000000 = 99.90%
; CHECK: edge %entry -> %else probability is 0x0020c49c / 0x80000000 = 0.10%

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