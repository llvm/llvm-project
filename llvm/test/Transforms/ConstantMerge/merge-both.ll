; RUN: opt -S < %s -passes=constmerge | FileCheck %s
; Test that in one run var2 is merged into var4 and var6 is merged into var8.
; Test that we merge @var6 and @var8 into one with the higher alignment

declare void @zed(ptr, ptr)

%struct.foobar = type { i32 }

@var1 = internal constant %struct.foobar { i32 2 }
@var2 = private unnamed_addr constant %struct.foobar { i32 2 }
@var3 = internal constant %struct.foobar { i32 2 }
@var4 = private unnamed_addr constant %struct.foobar { i32 2 }

; CHECK:      %struct.foobar = type { i32 }
; CHECK-NOT: @
; CHECK: @var1 = internal constant %struct.foobar { i32 2 }
; CHECK-NEXT: @var3 = internal constant %struct.foobar { i32 2 }
; CHECK-NEXT: @var4 = private unnamed_addr constant %struct.foobar { i32 2 }

declare void @helper(ptr)
@var5 = internal constant [16 x i8] c"foo1bar2foo3bar\00", align 16
@var6 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00", align 1
@var7 = internal constant [16 x i8] c"foo1bar2foo3bar\00"
@var8 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00"

; CHECK: @var5 = internal constant [16 x i8] c"foo1bar2foo3bar\00", align 16
; CHECK-NEXT: @var7 = internal constant [16 x i8] c"foo1bar2foo3bar\00"
; CHECK-NEXT: @var8 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00", align 1

@var4a = alias %struct.foobar, ptr @var4
@llvm.used = appending global [1 x ptr] [ptr @var4a], section "llvm.metadata"

define i32 @main() {
entry:
  call void @zed(ptr @var1, ptr @var2)
  call void @zed(ptr @var3, ptr @var4)
  call void @helper(ptr @var5)
  call void @helper(ptr @var6)
  call void @helper(ptr @var7)
  call void @helper(ptr @var8)
  ret i32 0
}
