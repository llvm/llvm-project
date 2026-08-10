; RUN: llc -mtriple=amdgcn-amd-amdhsa -stop-after=finalize-isel -debug-only=isel -o /dev/null %s 2>&1 | FileCheck %s

define i64 @i64_test(i64 %i) nounwind readnone {
  %loc = alloca i64, addrspace(5)
  %j = load i64, ptr addrspace(5) %loc
  %r = add i64 %i, %j
  ret i64 %r
}

define i64 @i32_test(i32 %i) nounwind readnone {
  %loc = alloca i32, addrspace(5)
  %j = load i32, ptr addrspace(5) %loc
  %r = add i32 %i, %j
  %ext = zext i32 %r to i64
  ret i64 %ext
}

define i64 @i16_test(i16 %i) nounwind readnone {
  %loc = alloca i16, addrspace(5)
  %j = load i16, ptr addrspace(5) %loc
  %r = add i16 %i, %j
  %ext = zext i16 %r to i64
  ret i64 %ext
}

define i64 @i8_test(i8 %i) nounwind readnone {
  %loc = alloca i8, addrspace(5)
  %j = load i8, ptr addrspace(5) %loc
  %r = add i8 %i, %j
  %ext = zext i8 %r to i64
  ret i64 %ext
}
