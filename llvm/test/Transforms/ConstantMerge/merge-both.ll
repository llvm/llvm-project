; RUN: opt -S < %s -passes=constmerge | FileCheck %s
; Test that in one run var3 is merged into var2 and var1 into var4.
; Test that we merge @var5 and @var6 into one with the higher alignment

declare void @zed(ptr, ptr)

%struct.foobar = type { i32 }

@var1 = internal constant %struct.foobar { i32 2 }
@var2 = unnamed_addr constant %struct.foobar { i32 2 }
@var3 = internal constant %struct.foobar { i32 2 }
@var4 = unnamed_addr constant %struct.foobar { i32 2 }

; CHECK:      %struct.foobar = type { i32 }
; CHECK-NOT: @
; CHECK: @var2 = constant %struct.foobar { i32 2 }
; CHECK-NEXT: @var4 = constant %struct.foobar { i32 2 }

declare void @helper(ptr)
@var5 = internal constant [16 x i8] c"foo1bar2foo3bar\00", align 16
@var6 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00", align 1
@var7 = internal constant [16 x i8] c"foo1bar2foo3bar\00"
@var8 = private unnamed_addr constant [16 x i8] c"foo1bar2foo3bar\00"

; CHECK-NEXT: @var7 = internal constant [16 x i8] c"foo1bar2foo3bar\00"
; CHECK-NEXT: @var8 = private constant [16 x i8] c"foo1bar2foo3bar\00", align 16

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

