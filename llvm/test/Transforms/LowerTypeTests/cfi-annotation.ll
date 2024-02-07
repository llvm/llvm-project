; RUN: opt -passes=lowertypetests %s -o %t.o
; RUN: llvm-dis %t.o -o - | FileCheck %s --check-prefix=CHECK-foobar
; CHECK-foobar: {{llvm.global.annotations = .*[foo|bar], .*[foo|bar],}}
; RUN: llvm-dis %t.o -o - | FileCheck %s --check-prefix=CHECK-cfi
; CHECK-cfi-NOT: {{llvm.global.annotations = .*cfi.*}}

target triple = "aarch64-none-linux-gnu"

@fptr = global ptr null, align 8
@.src = private unnamed_addr constant [7 x i8] c"test.c\00", align 1
@anon.eb4aa7a5d41c72267995d92d93c37bde.0 = private unnamed_addr constant { i16, i16, [14 x i8] } { i16 -1, i16 0, [14 x i8] c"'void (void)'\00" }
@.str = private unnamed_addr constant [30 x i8] c"annotation_string_literal_bar\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [7 x i8] c"test.c\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [30 x i8] c"annotation_string_literal_foo\00", section "llvm.metadata"
@llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @bar, ptr @.str, ptr @.str.1, i32 3, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @foo, ptr @.str.2, ptr @.str.1, i32 2, ptr null }], section "llvm.metadata"

define void @bar() {
entry:
  ret void
}

define void @test(i32 noundef %x) {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  %1 = zext i1 %cmp to i64
  %cond = select i1 %cmp, ptr @foo, ptr @bar
  store ptr %cond, ptr @fptr, align 8
  %2 = load ptr, ptr @fptr, align 8
  %3 = call i1 @llvm.type.test(ptr %2, metadata !"_ZTSFvvE"), !nosanitize !{}
  br i1 %3, label %cont, label %trap, !nosanitize !{}

trap:
  call void @llvm.ubsantrap(i8 2) #4, !nosanitize !{}
  unreachable, !nosanitize !{}

cont:
  call void %2()
  ret void
}

declare void @foo(...)
declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.ubsantrap(i8 immarg)
